# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import random
import time
import torch
import argparse
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import sys
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

from data.config import cfg
from layers.modules import MultiBoxLoss, EnhanceLoss
from models.factory import build_net, basenet_factory
from models.enhancer import RetinexNet
from utils.DarkISP import Low_Illumination_Degrading
from PIL import Image

# Import your custom dataset
from data.people_dataset import PeopleDetection, detection_collate

# Force unbuffered stdout so prints appear immediately
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

parser = argparse.ArgumentParser(description='DSFD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training (increased from 4)')
parser.add_argument('--model', default='dark', type=str, choices=['dark', 'vgg', 'resnet50', 'resnet101', 'resnet152'], help='model for training')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading (increased from 0)')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--multigpu', default=True, type=bool, help='Use mutil Gpu training')
parser.add_argument('--save_folder', default='weights/', help='Directory for saving checkpoint models')
parser.add_argument('--local_rank', type=int, default=0, help='local rank for dist')
# === NEW: Training optimization arguments ===
parser.add_argument('--amp', action='store_true', default=True, help='Use Automatic Mixed Precision (AMP) training')
parser.add_argument('--no_amp', action='store_true', help='Disable AMP even if --amp is set')
parser.add_argument('--val_interval', default=5, type=int, help='Run validation every N epochs (default: 5)')
parser.add_argument('--early_stop_patience', default=15, type=int, help='Early stopping patience in epochs (0 to disable)')
parser.add_argument('--cosine_lr', action='store_true', help='Use cosine annealing LR scheduler instead of step decay')
parser.add_argument('--grad_accum_steps', default=1, type=int, help='Gradient accumulation steps (increase effective batch size without more VRAM)')
parser.add_argument('--compile', action='store_true', help='Use torch.compile for PyTorch 2.0+ (can give 10-30%% speedup)')
parser.add_argument('--prefetch', action='store_true', default=True, help='Use CUDA prefetching for data loading')

args = parser.parse_args()

# Handle --no_amp flag
if args.no_amp:
    args.amp = False

# --- FIX: Correctly detect rank for torchrun ---
if 'LOCAL_RANK' in os.environ:
    args.local_rank = int(os.environ['LOCAL_RANK'])

local_rank = args.local_rank
# -----------------------------------------------

if torch.cuda.is_available():
    if args.cuda:
        import torch.distributed as dist
        gpu_num = torch.cuda.device_count()
        if local_rank == 0:
            print('Using {} gpus'.format(gpu_num))

        # Safe device setting
        if 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
            torch.cuda.set_device(rank % gpu_num)
        else:
            torch.cuda.set_device(local_rank)

        dist.init_process_group('nccl')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using CUDA.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

save_folder = os.path.join(args.save_folder, args.model)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Load Dataset
train_dataset = PeopleDetection(cfg.params.img_train_path, image_sets='train')
val_dataset = PeopleDetection(cfg.params.img_val_path, image_sets='valid')

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               collate_fn=detection_collate,
                               sampler=train_sampler,
                               pin_memory=True,
                               persistent_workers=args.num_workers > 0,
                               prefetch_factor=2 if args.num_workers > 0 else None)

val_batchsize = args.batch_size
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True)
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=args.num_workers,
                             collate_fn=detection_collate,
                             sampler=val_sampler,
                             pin_memory=True,
                             persistent_workers=args.num_workers > 0,
                             prefetch_factor=2 if args.num_workers > 0 else None)

min_loss = np.inf


class CUDAPrefetcher:
    """Prefetch data to GPU using a separate CUDA stream for overlap."""
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        first = True
        batch = None
        for next_batch in self.loader:
            with torch.cuda.stream(self.stream):
                next_images = next_batch[0].cuda(non_blocking=True)
                next_targets = [ann.cuda(non_blocking=True) for ann in next_batch[1]]
                next_extra = next_batch[2] if len(next_batch) > 2 else None
            if not first:
                yield batch
            else:
                first = False
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = (next_images, next_targets, next_extra)
        if batch is not None:
            yield batch

    def __len__(self):
        return len(self.loader)


def train():
    per_epoch_size = len(train_dataset) // (args.batch_size * torch.cuda.device_count())
    start_epoch = 0
    iteration = 0
    step_index = 0

    basenet = basenet_factory(args.model)
    dsfd_net = build_net('train', cfg.NUM_CLASSES, args.model)
    net = dsfd_net
    net_enh = RetinexNet()

    decomp_path = args.save_folder + 'decomp.pth'
    if os.path.exists(decomp_path):
        net_enh.load_state_dict(torch.load(decomp_path))

    if args.resume:
        if local_rank == 0:
            print('Resuming training, loading {}...'.format(args.resume))
        start_epoch = net.load_weights(args.resume)
        iteration = start_epoch * per_epoch_size
    else:
        base_weight_path = args.save_folder + basenet
        if os.path.exists(base_weight_path):
            base_weights = torch.load(base_weight_path)
            if local_rank == 0:
                print('Load base network {}'.format(base_weight_path))
            if args.model == 'vgg' or args.model == 'dark':
                net.vgg.load_state_dict(base_weights)
            else:
                net.resnet.load_state_dict(base_weights)

    if not args.resume:
        if local_rank == 0:
            print('Initializing weights...')
        net.extras.apply(net.weights_init)
        net.fpn_topdown.apply(net.weights_init)
        net.fpn_latlayer.apply(net.weights_init)
        net.fpn_fem.apply(net.weights_init)
        net.loc_pal1.apply(net.weights_init)
        net.conf_pal1.apply(net.weights_init)
        net.loc_pal2.apply(net.weights_init)
        net.conf_pal2.apply(net.weights_init)
        net.ref.apply(net.weights_init)

    lr = args.lr * np.round(np.sqrt(args.batch_size / 4 * torch.cuda.device_count()),4)
    param_group = []
    param_group += [{'params': dsfd_net.vgg.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.extras.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_topdown.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_latlayer.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_fem.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.loc_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.conf_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.loc_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.conf_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.ref.parameters(), 'lr': lr / 10.}]

    optimizer = optim.SGD(param_group, lr=lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # === NEW: Cosine annealing LR scheduler (optional) ===
    scheduler = None
    if args.cosine_lr:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.EPOCHES, eta_min=lr * 0.01)

    if args.cuda:
        if args.multigpu:
            net = torch.nn.parallel.DistributedDataParallel(net.cuda(), find_unused_parameters=False)
            net_enh = torch.nn.parallel.DistributedDataParallel(net_enh.cuda(), find_unused_parameters=False)
        cudnn.benchmark = True

    # === NEW: torch.compile for PyTorch 2.0+ ===
    if args.compile and hasattr(torch, 'compile'):
        if local_rank == 0:
            print('Compiling model with torch.compile...')
        net = torch.compile(net)
        net_enh = torch.compile(net_enh)

    # === NEW: AMP GradScaler ===
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    if local_rank == 0 and args.amp:
        print('Using Automatic Mixed Precision (AMP) training')

    criterion = MultiBoxLoss(cfg, args.cuda)
    criterion_enhance = EnhanceLoss()

    if local_rank == 0:
        print('Using the specified args:')
        print(args)
        print(f'Effective batch size: {args.batch_size * args.grad_accum_steps * torch.cuda.device_count()}')
        print('Starting training loop...')

    for step in cfg.LR_STEPS:
        if iteration > step:
            step_index += 1
            if not args.cosine_lr:
                adjust_learning_rate(optimizer, args.gamma, step_index)

    net_enh.eval()
    net.train()

    # === NEW: Early stopping state ===
    early_stop_counter = 0

    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        train_sampler.set_epoch(epoch)  # Important for proper shuffling with DDP
        if local_rank == 0:
            print(f"Epoch {epoch}/{cfg.EPOCHES} started.")

        # Use prefetcher if enabled
        loader = CUDAPrefetcher(train_loader) if args.prefetch else train_loader

        for batch_idx, batch_data in enumerate(loader):
            if args.prefetch:
                images, targets_list, _ = batch_data
                images = images / 255.
                targetss = [ann for ann in targets_list]
            else:
                images, targets, _ = batch_data
                images = images.cuda(non_blocking=True) / 255.
                targetss = [ann.cuda(non_blocking=True) for ann in targets]

            # Low illumination degradation (already on GPU, no loop needed for stack)
            img_dark = torch.stack(
                [Low_Illumination_Degrading(images[i])[0] for i in range(images.shape[0])], dim=0)

            if not args.cosine_lr and iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()

            # === AMP autocast for forward pass ===
            with torch.amp.autocast('cuda', enabled=args.amp):
                R_dark_gt, I_dark = net_enh(img_dark)
                R_light_gt, I_light = net_enh(images)

                out, out2, loss_mutual = net(img_dark, images, I_dark.detach(), I_light.detach())
                R_dark, R_light, R_dark_2, R_light_2 = out2

                loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targetss)
                loss_l_pa12, loss_c_pal2 = criterion(out[3:], targetss)

                loss_enhance = criterion_enhance([R_dark, R_light, R_dark_2, R_light_2, I_dark.detach(), I_light.detach()], images, img_dark) * 0.1
                loss_enhance2 = F.l1_loss(R_dark, R_dark_gt.detach()) + F.l1_loss(R_light, R_light_gt.detach()) + (
                            1. - ssim(R_dark, R_dark_gt.detach())) + (1. - ssim(R_light, R_light_gt.detach()))

                loss = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2 + loss_enhance2 + loss_enhance + loss_mutual
                # Scale loss for gradient accumulation
                loss = loss / args.grad_accum_steps

            # === AMP backward pass ===
            scaler.scale(loss).backward()

            # Gradient accumulation: only step every N batches
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=35, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            losses += loss.item() * args.grad_accum_steps  # Undo scaling for logging

            if iteration % 5 == 0:
                tloss = losses / (batch_idx + 1)
                if local_rank == 0:
                    print(f'Iter {iteration} || Time: {(t1 - t0):.4f}s || Loss: {tloss:.4f} || LR: {optimizer.param_groups[0]["lr"]}')

            if iteration != 0 and iteration % 5000 == 0:
                if local_rank == 0:
                    print('Saving state, iter:', iteration)
                    file = 'dsfd_' + repr(iteration) + '.pth'
                    torch.save(dsfd_net.state_dict(),
                               os.path.join(save_folder, file))
            iteration += 1

        # === Cosine LR scheduler step ===
        if args.cosine_lr and scheduler is not None:
            scheduler.step()

        # === NEW: Validate every N epochs instead of every epoch ===
        if (epoch + 1) % args.val_interval == 0 or epoch == cfg.EPOCHES - 1:
            improved = val(epoch, net, dsfd_net, net_enh, criterion)

            # === NEW: Early stopping ===
            if args.early_stop_patience > 0:
                if improved:
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if local_rank == 0:
                        print(f'Early stopping: {early_stop_counter}/{args.early_stop_patience}')
                    if early_stop_counter >= args.early_stop_patience:
                        if local_rank == 0:
                            print(f'Early stopping triggered at epoch {epoch}')
                        break

        if iteration >= cfg.MAX_STEPS:
            break

def val(epoch, net, dsfd_net, net_enh, criterion):
    net.eval()
    step = 0
    losses = torch.tensor(0.).cuda()
    t1 = time.time()

    with torch.no_grad():
        for batch_idx, (images, targets, img_paths) in enumerate(val_loader):
            images = images.cuda(non_blocking=True) / 255.
            targets = [ann.cuda(non_blocking=True) for ann in targets]

            img_dark = torch.stack([Low_Illumination_Degrading(images[i])[0] for i in range(images.shape[0])], dim=0)

            with torch.amp.autocast('cuda', enabled=args.amp):
                if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                     out, R = net.module.test_forward(img_dark)
                else:
                     out, R = net.test_forward(img_dark)

                loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)
                loss = loss_l_pa12 + loss_c_pal2

            losses += loss.item()
            step += 1

    dist.reduce(losses, 0, op=dist.ReduceOp.SUM)
    tloss = losses / step / torch.cuda.device_count()
    t2 = time.time()

    if local_rank == 0:
        print('Validation Timer: %.4f' % (t2 - t1))
        print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    improved = tloss < min_loss
    if improved:
        if local_rank == 0:
            print('Saving best state,epoch', epoch)
            torch.save(dsfd_net.state_dict(), os.path.join(save_folder, 'dsfd.pth'))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': dsfd_net.state_dict(),
    }
    if local_rank == 0:
        torch.save(states, os.path.join(save_folder, 'dsfd_checkpoint.pth'))

    net.train()
    return improved

def adjust_learning_rate(optimizer, gamma, step):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma

if __name__ == '__main__':
    train()
