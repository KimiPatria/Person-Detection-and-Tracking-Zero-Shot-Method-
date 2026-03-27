# -*- coding: utf-8 -*-
"""
Training script — DAI-Net with YOLOv8n backbone.
Zero-shot low-light human detection (CCTV deployment target).

Combined loss
-------------
  L = λ_box · L_CIoU  +  λ_cls · L_BCE              (detection)
    + L_enhance2                                       (reflectance L1 + SSIM)
    + L_enhance  (EnhanceLoss)  × 0.1                 (full Retinex coherence)
    + L_mutual   (KL alignment)                        (zero-shot)

Optimisations vs original
-------------------------
  1. DarkISP moved to DataLoader workers → parallel CPU pre-computation
  2. Backbone stem-only for auxiliary passes → ~3.6× backbone FLOP reduction
  3. Vectorised build_targets → eliminates Python triple-loop
  4. EMA (Exponential Moving Average) → better generalisation
  5. Linear warmup + cosine annealing → stable early training
  6. Larger batch size (32) → better GPU utilisation on 5090
  7. Reduced epochs (60) → diminishing returns beyond ~50
  8. Validation every 3 epochs → less overhead

AMP / bf16 note
---------------
  The main network forward and detection loss run in bf16 (Blackwell native).
  EnhanceLoss is kept in fp32 because it uses hardcoded float32 convolution
  kernels internally — R_maps are cast to .float() before that block.
  No GradScaler is needed for bf16; one is used automatically for fp16 fallback.

Usage
-----
  Single GPU (default — vast.ai single-GPU rental):
      python train_yolo.py

  Multi-GPU (torchrun):
      torchrun --nproc_per_node=<N> train_yolo.py --multigpu True
"""

from __future__ import division, absolute_import, print_function

import os
import sys
import copy
import math
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure as ssim

from data.config import cfg
from layers.modules import EnhanceLoss
from models.factory import build_net
from models.enhancer import RetinexNet
from models.DAINet_yolov8 import compute_detection_loss
from utils.DarkISP import Low_Illumination_Degrading
from data.people_dataset import PeopleDetection


# ── Unbuffered stdout (shows logs immediately in vast.ai terminal) ────────────
class Unbuffered:
    def __init__(self, s):   self.s = s
    def write(self, d):      self.s.write(d);  self.s.flush()
    def writelines(self, d): self.s.writelines(d); self.s.flush()
    def __getattr__(self, a): return getattr(self.s, a)

sys.stdout = Unbuffered(sys.stdout)

# ── TF32 on Ampere+ / Blackwell gives free ~8 % throughput boost ─────────────
torch.set_float32_matmul_precision('high')


# ═══════════════════════════════════════════════════════════════════════════
# EMA (Exponential Moving Average)
# ═══════════════════════════════════════════════════════════════════════════

class ModelEMA:
    """YOLOv8-style EMA with decay warmup for better generalisation."""
    def __init__(self, model, decay=0.9999, tau=2000):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.tau = tau
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        with torch.no_grad():
            for k, v in self.ema.state_dict().items():
                if v.is_floating_point():
                    v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


# ═══════════════════════════════════════════════════════════════════════════
# DarkISP pre-computation in DataLoader workers
# ═══════════════════════════════════════════════════════════════════════════

class DarkAugDataset(data.Dataset):
    """Wraps PeopleDetection to pre-compute DarkISP in parallel CPU workers."""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target, path = self.base[idx]
        img_norm = img.float() / 255.0
        img_dark, _ = Low_Illumination_Degrading(img_norm)
        return img, img_dark, target, path


def dark_collate(batch):
    """Collate function for DarkAugDataset (adds img_dark to batch)."""
    imgs, darks, targets, paths = [], [], [], []
    for img, dark, tgt, path in batch:
        imgs.append(img)
        darks.append(dark)
        targets.append(torch.FloatTensor(tgt))
        paths.append(path)
    return torch.stack(imgs), torch.stack(darks), targets, paths


# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='DAI-Net YOLOv8n Training')
parser.add_argument('--batch_size',   default=32,    type=int,
                    help='Per-GPU batch size (5090 32 GB: 32 recommended)')
parser.add_argument('--resume',       default=None,  type=str,
                    help='Checkpoint path to resume from')
parser.add_argument('--num_workers',  default=8,     type=int,
                    help='DataLoader workers (8 recommended for parallel DarkISP)')
parser.add_argument('--cuda',         default=True,  type=bool)
parser.add_argument('--lr',           default=1e-3,  type=float)
parser.add_argument('--weight_decay', default=5e-4,  type=float)
parser.add_argument('--multigpu',     default=False, type=bool,
                    help='Enable DDP (use with torchrun for multi-GPU)')
parser.add_argument('--save_folder',  default='weights/', type=str)
parser.add_argument('--local_rank',   default=0,     type=int)
parser.add_argument('--lambda_box',   default=5.0,   type=float)
parser.add_argument('--lambda_cls',   default=1.0,   type=float)
parser.add_argument('--amp',          default=True,  type=lambda x: x.lower() != 'false',
                    help='Mixed precision training. bf16 on Blackwell, fp16 fallback '
                         '(disable with --amp false)')
parser.add_argument('--warmup_epochs', default=3,    type=int,
                    help='Linear LR warmup epochs')
parser.add_argument('--val_interval', default=3,     type=int,
                    help='Validate every N epochs (saves time)')
args = parser.parse_args()

# ── Distributed / device setup ───────────────────────────────────────────────
if 'LOCAL_RANK' in os.environ:
    args.local_rank = int(os.environ['LOCAL_RANK'])
local_rank = args.local_rank

use_cuda = torch.cuda.is_available() and args.cuda

if use_cuda:
    import torch.distributed as dist
    gpu_num = torch.cuda.device_count()
    if args.multigpu:
        if 'RANK' in os.environ:
            torch.cuda.set_device(int(os.environ['RANK']) % gpu_num)
        else:
            torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
    else:
        torch.cuda.set_device(0)
    cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# ── AMP dtype selection ───────────────────────────────────────────────────────
# RTX 5090 (Blackwell) natively supports bf16 — prefer it.
# Older GPUs fall back to fp16 + GradScaler.
if args.amp and use_cuda:
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = (amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    if local_rank == 0:
        print(f'[INFO] AMP enabled — using {amp_dtype}')
else:
    amp_dtype = torch.float32
    use_scaler = False
    scaler = None

# ── Save directory ───────────────────────────────────────────────────────────
save_folder = os.path.join(args.save_folder, 'yolo_dark')
os.makedirs(save_folder, exist_ok=True)

# ── Dataset (wrapped with DarkAugDataset for parallel DarkISP) ───────────────
train_dataset = DarkAugDataset(PeopleDetection(cfg.params.img_train_path, image_sets='train'))
val_dataset   = DarkAugDataset(PeopleDetection(cfg.params.img_val_path,   image_sets='valid'))

if args.multigpu and use_cuda:
    train_sampler = data.distributed.DistributedSampler(train_dataset, shuffle=True)
    val_sampler   = data.distributed.DistributedSampler(val_dataset,   shuffle=False)
else:
    train_sampler = None
    val_sampler   = None

# persistent_workers=True avoids re-spawning workers every epoch on Linux.
# prefetch_factor pre-loads batches so the GPU is never starved.
_worker_kwargs = dict(
    num_workers=args.num_workers,
    persistent_workers=(args.num_workers > 0),
    prefetch_factor=(2 if args.num_workers > 0 else None),
)

train_loader = data.DataLoader(
    train_dataset, args.batch_size,
    shuffle=(train_sampler is None),
    collate_fn=dark_collate,
    sampler=train_sampler,
    pin_memory=use_cuda,
    **_worker_kwargs,
)
val_loader = data.DataLoader(
    val_dataset, args.batch_size,
    shuffle=False,
    collate_fn=dark_collate,
    sampler=val_sampler,
    pin_memory=use_cuda,
    **_worker_kwargs,
)

min_val_loss = float('inf')


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train():
    global min_val_loss

    # ── Build model ──────────────────────────────────────────────────────
    net = build_net('train', num_classes=1, model='yolo_dark')

    # ── Load pretrained YOLOv8n COCO backbone ───────────────────────────
    BACKBONE_PT = './weights/yolov8n.pt'
    if not os.path.exists(BACKBONE_PT):
        if local_rank == 0:
            print('[INFO] Downloading yolov8n.pt from Ultralytics...')
            import urllib.request
            os.makedirs(os.path.dirname(BACKBONE_PT), exist_ok=True)
            urllib.request.urlretrieve(
                'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt',
                BACKBONE_PT)
            print('[INFO] Download complete.')
    if os.path.exists(BACKBONE_PT) and not args.resume:
        net.load_pretrained_backbone(BACKBONE_PT)

    # ── Frozen RetinexNet (provides pseudo GT R and I maps) ──────────────
    net_enh = RetinexNet()
    decomp_path = os.path.join(args.save_folder, 'decomp.pth')
    if os.path.exists(decomp_path):
        net_enh.load_state_dict(torch.load(decomp_path, map_location='cpu'))
        if local_rank == 0:
            print(f'[INFO] Loaded RetinexNet from {decomp_path}')
    else:
        if local_rank == 0:
            print('[WARN] decomp.pth not found — RetinexNet using random weights.')
    net_enh.eval()
    for p in net_enh.parameters():
        p.requires_grad_(False)

    # ── Optionally resume ─────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = net.load_weights(args.resume)
        if local_rank == 0:
            print(f'[INFO] Resumed from {args.resume}, starting at epoch {start_epoch}')

    # ── Move to device ────────────────────────────────────────────────────
    net     = net.to(device)
    net_enh = net_enh.to(device)

    if args.multigpu and use_cuda:
        net     = nn.parallel.DistributedDataParallel(net,     find_unused_parameters=True)
        net_enh = nn.parallel.DistributedDataParallel(net_enh, find_unused_parameters=False)

    # ── Unwrap for EMA / checkpointing ────────────────────────────────────
    core = net.module if isinstance(net, nn.parallel.DistributedDataParallel) else net

    # ── EMA ───────────────────────────────────────────────────────────────
    ema = ModelEMA(core, decay=0.9999)

    # ── Optimiser: AdamW ─────────────────────────────────────────────────
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── LR schedule: linear warmup → cosine annealing ────────────────────
    warmup_epochs = args.warmup_epochs
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.EPOCHES - warmup_epochs, 1), eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])

    # Advance scheduler if resuming
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    # ── Enhance loss (fp32 — uses hardcoded float32 conv kernels) ─────────
    criterion_enhance = EnhanceLoss()

    if local_rank == 0:
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f'[INFO] Model      : DAI-Net YOLOv8n')
        print(f'[INFO] Parameters : {n_params / 1e6:.2f} M')
        print(f'[INFO] Device     : {device}  |  AMP dtype: {amp_dtype}')
        print(f'[INFO] Batch size : {args.batch_size}')
        print(f'[INFO] Train set  : {len(train_dataset)} images')
        print(f'[INFO] Val set    : {len(val_dataset)} images')
        print(f'[INFO] Epochs     : {cfg.EPOCHES}  (warmup: {warmup_epochs})')
        print(f'[INFO] EMA enabled (decay=0.9999)')
        print(f'[INFO] Val every  : {args.val_interval} epochs')
        print(f'[INFO] Starting training …')

    for epoch in range(start_epoch, cfg.EPOCHES):
        net.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        t_epoch    = time.time()

        for batch_idx, (images, img_dark, targets, _) in enumerate(train_loader):

            # ── Inputs (DarkISP already applied by DataLoader workers) ────
            images   = images.to(device, dtype=torch.float32) / 255.0
            img_dark = img_dark.to(device, dtype=torch.float32)

            # ── RetinexNet: frozen, fp32, no grad ─────────────────────────
            with torch.no_grad():
                R_dark_gt,  I_dark  = net_enh(img_dark)   # fp32
                R_light_gt, I_light = net_enh(images)     # fp32

            gt_list = [t.to(device) for t in targets]

            # ── Forward + detection loss in AMP (bf16 / fp16) ─────────────
            t0 = time.time()
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=args.amp):
                preds, R_maps, loss_mutual = net(
                    img_dark, images, I_dark.detach(), I_light.detach())
                loss_box, loss_cls = compute_detection_loss(preds, gt_list, device)

            # ── Enhance losses: fp32 (EnhanceLoss has float32 conv kernels) ─
            # Cast R_maps out of bf16 — gradients still flow back through .float()
            R_dark, R_light, R_dark_2, R_light_2 = [r.float() for r in R_maps]
            Rdg = R_dark_gt.detach();  Rlg = R_light_gt.detach()

            loss_enhance2 = (
                F.l1_loss(R_dark,  Rdg) + F.l1_loss(R_light, Rlg) +
                (1.0 - ssim(R_dark,  Rdg)) + (1.0 - ssim(R_light, Rlg))
            )
            loss_enhance = criterion_enhance(
                [R_dark, R_light, R_dark_2, R_light_2,
                 I_dark.detach().float(), I_light.detach().float()],
                images, img_dark
            ) * 0.1

            # ── Combined loss (fp32) ───────────────────────────────────────
            loss = (args.lambda_box * loss_box.float()
                    + args.lambda_cls * loss_cls.float()
                    + 0.3 * loss_enhance2
                    + loss_enhance
                    + 0.5 * loss_mutual.float())

            optimizer.zero_grad()
            if use_scaler:                          # fp16 path
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
            else:                                   # bf16 / fp32 path
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()

            # ── Update EMA ─────────────────────────────────────────────────
            ema.update(core)

            epoch_loss += loss.item()
            t1 = time.time()

            if batch_idx % 10 == 0 and local_rank == 0:
                print(
                    f'Epoch {epoch:03d} | Iter {batch_idx:04d} | '
                    f'Loss {loss.item():.4f} '
                    f'(box {loss_box.item():.3f}  '
                    f'cls {loss_cls.item():.3f}  '
                    f'enh {loss_enhance2.item():.3f}  '
                    f'kl {loss_mutual.item():.3f}) | '
                    f'{t1 - t0:.3f}s/iter | '
                    f'LR {optimizer.param_groups[0]["lr"]:.2e}'
                )

        scheduler.step()

        if local_rank == 0:
            avg     = epoch_loss / max(len(train_loader), 1)
            elapsed = time.time() - t_epoch
            print(f'[Epoch {epoch:03d}] avg_loss={avg:.4f}  elapsed={elapsed:.1f}s')

        # ── Validation (every N epochs + final epoch) ─────────────────────
        run_val = ((epoch + 1) % args.val_interval == 0) or (epoch + 1 == cfg.EPOCHES)
        if run_val:
            val_loss = validate(ema.ema, net_enh, criterion_enhance)
            if local_rank == 0:
                print(f'[Epoch {epoch:03d}] val_loss={val_loss:.4f}')

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    # Save EMA weights as best model (better generalisation)
                    torch.save(ema.ema.state_dict(),
                               os.path.join(save_folder, 'dsfd.pth'))
                    print(f'[INFO] Best model saved  (val_loss={val_loss:.4f}, EMA)')

        # ── Checkpoints ───────────────────────────────────────────────────
        if local_rank == 0:
            torch.save({'epoch': epoch + 1, 'weight': core.state_dict()},
                       os.path.join(save_folder, 'yolo_checkpoint.pth'))

            if (epoch + 1) % 10 == 0:
                torch.save(ema.ema.state_dict(),
                           os.path.join(save_folder, f'yolo_epoch{epoch + 1:03d}.pth'))


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate(net_eval, net_enh, criterion_enhance):
    """Run validation using the provided model (typically EMA)."""
    net_eval.eval()
    total_loss = 0.0
    steps      = 0

    with torch.no_grad():
        for images, img_dark, targets, _ in val_loader:
            images   = images.to(device, dtype=torch.float32) / 255.0
            img_dark = img_dark.to(device, dtype=torch.float32)

            R_dark_gt,  I_dark  = net_enh(img_dark)
            R_light_gt, I_light = net_enh(images)

            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=args.amp):
                preds, R_maps, loss_mutual = net_eval(
                    img_dark, images, I_dark.detach(), I_light.detach())
                gt_list   = [t.to(device) for t in targets]
                loss_box, loss_cls = compute_detection_loss(preds, gt_list, device)

            R_dark, R_light, R_dark_2, R_light_2 = [r.float() for r in R_maps]
            Rdg = R_dark_gt;  Rlg = R_light_gt

            loss_enhance2 = (
                F.l1_loss(R_dark,  Rdg) + F.l1_loss(R_light, Rlg) +
                (1.0 - ssim(R_dark,  Rdg)) + (1.0 - ssim(R_light, Rlg))
            )
            loss_enhance = criterion_enhance(
                [R_dark, R_light, R_dark_2, R_light_2,
                 I_dark.detach().float(), I_light.detach().float()],
                images, img_dark
            ) * 0.1

            loss = (args.lambda_box * loss_box.float()
                    + args.lambda_cls * loss_cls.float()
                    + 0.3 * loss_enhance2 + loss_enhance + 0.5 * loss_mutual.float())
            total_loss += loss.item()
            steps      += 1

    return total_loss / max(steps, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    train()
