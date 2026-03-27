#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import time

import argparse
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from models.factory import build_net
from torchvision.utils import make_grid
import glob

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.benckmark = True
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def tensor_to_image(tensor):
    grid = make_grid(tensor)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return ndarr

def to_chw_bgr(image):
    """
    Transpose image from HWC to CHW and from RBG to BGR.
    Args:
        image (np.array): an image with HWC and RBG layout.
    """
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    image = image[[2, 1, 0], :, :]
    return image

def letterbox(img, target_size=640):
    """Resize image preserving aspect ratio and pad to target_size×target_size."""
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=img.dtype)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, pad_left, pad_top


def detect_face(img, tmp_shrink):
    orig_h, orig_w = img.shape[:2]
    image, scale, pad_left, pad_top = letterbox(img, 640)

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x = x / 255.
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))

    if use_cuda:
        x = x.cuda()

    y = net.test_forward(x)[0]
    detections = y.data.cpu().numpy()
    # test_forward returns normalised coords relative to padded 640×640
    lb_scale = np.array([640, 640, 640, 640])

    boxes=[]
    scores = []
    for i in range(detections.shape[1]):
      j = 0
      while ((j < detections.shape[2]) and detections[0, i, j, 0] > 0.0):
        pt = detections[0, i, j, 1:] * lb_scale  # to padded-pixel coords
        score = detections[0, i, j, 0]
        # undo letterbox: remove padding, then undo scale
        x1 = (pt[0] - pad_left) / scale
        y1 = (pt[1] - pad_top) / scale
        x2 = (pt[2] - pad_left) / scale
        y2 = (pt[3] - pad_top) / scale
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        j += 1

    det_conf = np.array(scores)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0,0,0,0,0.001]])

    det = np.column_stack((boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], det_conf))

    return det


def flip_test(image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def multi_scale_test(image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image, st)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s,detect_face(image, 0.75)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(image, bt)

    # enlarge small iamge x times for small face
    if max_im_shrink > 1.5:
        det_b = np.row_stack((det_b,detect_face(image, 1.5)))
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink: # and bt <= 2:
            det_b = np.row_stack((det_b, detect_face(image, bt)))
            bt *= 2

        det_b = np.row_stack((det_b, detect_face(image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b


def multi_scale_test_pyramid(image, max_shrink):
    det_b = detect_face(image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(image, st[i])
            # enlarge only detect small face
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b


def bbox_vote(det_):
    order_ = det_[:, 4].ravel().argsort()[::-1]
    det_ = det_[order_, :]
    dets_ = np.zeros((0, 5),dtype=np.float32)
    while det_.shape[0] > 0:
        # IOU
        area_ = (det_[:, 2] - det_[:, 0] + 1) * (det_[:, 3] - det_[:, 1] + 1)
        xx1 = np.maximum(det_[0, 0], det_[:, 0])
        yy1 = np.maximum(det_[0, 1], det_[:, 1])
        xx2 = np.minimum(det_[0, 2], det_[:, 2])
        yy2 = np.minimum(det_[0, 3], det_[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o_ = inter / (area_[0] + area_[:] - inter)

        # get needed merge det and delete these det
        merge_index_ = np.where(o_ >= 0.5)[0]
        det_accu_ = det_[merge_index_, :]
        det_ = np.delete(det_, merge_index_, 0)

        if merge_index_.shape[0] <= 1:
            continue
        det_accu_[:, 0:4] = det_accu_[:, 0:4] * np.tile(det_accu_[:, -1:], (1, 4))
        max_score_ = np.max(det_accu_[:, 4])
        det_accu_sum_ = np.zeros((1, 5))
        det_accu_sum_[:, 0:4] = np.sum(det_accu_[:, 0:4], axis=0) / np.sum(det_accu_[:, -1:])
        det_accu_sum_[:, 4] = max_score_
        try:
            dets_ = np.row_stack((dets_, det_accu_sum_))
        except:
            dets_ = det_accu_sum_

    dets_ = dets_[0:750, :]
    return dets_


_WEIGHT_DEFAULTS = {
    'dark':      'weights/dsfd.pth',
    'yolo_dark': 'weights/yolo_dark/dsfd.pth',
}

def load_models(model_type='yolo_dark', weights_path=None):
    print(f'build network: {model_type}')
    weights_path = weights_path or _WEIGHT_DEFAULTS[model_type]
    num_classes  = 1 if model_type == 'yolo_dark' else 2
    net = build_net('test', num_classes=num_classes, model=model_type)
    net.eval()
    ckpt = torch.load(weights_path, map_location='cuda' if use_cuda else 'cpu')
    if isinstance(ckpt, dict) and 'weight' in ckpt:
        net.load_state_dict(ckpt['weight'])
    else:
        net.load_state_dict(ckpt)
    if use_cuda:
        net = net.cuda()
    return net

def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def voc_ap(rec, prec):
    # 11-point interpolated average precision (Standard Pascal VOC)
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.
    return ap

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        # If you have multiple classes, you can filter by name here:
        # if obj.find('name').text != 'person': continue
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        bboxes.append({'bbox': [xmin, ymin, xmax, ymax], 'matched': False})
    return bboxes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DAI-Net test')
    parser.add_argument('--model',   default='yolo_dark',
                        choices=['dark', 'yolo_dark'],
                        help='Model architecture (default: yolo_dark)')
    parser.add_argument('--weights', default=None,
                        help='Path to .pth weights file (auto-detected if omitted)')
    cli = parser.parse_args()

    ''' Parameters '''
    USE_MULTI_SCALE = False
    MY_SHRINK = 1

    # Define your paths here
    images_dir      = './dataset/roboflow/test/images/'
    annotations_dir = './dataset/roboflow/test/annotations/'
    save_path       = './result/'

    def load_images():
        imglist = glob.glob(os.path.join(images_dir, '*.jpg'))
        return imglist

    ''' Main Test & Evaluation '''
    net = load_models(cli.model, cli.weights)
    img_list = load_images()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Dictionaries to store everything for mAP
    all_detections = []    # Will store: [image_id, confidence, xmin, ymin, xmax, ymax]
    all_ground_truths = {} # Will store: {image_id: [{'bbox': [...], 'matched': False}, ...]}
    total_gt_boxes = 0

    now = 0
    print('Processing and Evaluating: {} images'.format(img_list.__len__()))
    
    for img_path in img_list:
        image_id = Path(img_path).stem
        
        # 1. Load Ground Truth XML
        xml_path = os.path.join(annotations_dir, image_id + '.xml')
        if os.path.exists(xml_path):
            gt_boxes = parse_voc_xml(xml_path)
            all_ground_truths[image_id] = gt_boxes
            total_gt_boxes += len(gt_boxes)
        else:
            all_ground_truths[image_id] = []

        # 2. Load and Detect Image
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        image = np.array(image)

        start_time = time.time()

        if USE_MULTI_SCALE:
            max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.5 
            max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink
            with torch.no_grad():
                det0 = detect_face(image, MY_SHRINK)
                det1 = flip_test(image, MY_SHRINK)
                [det2, det3] = multi_scale_test(image, max_im_shrink)
                det4 = multi_scale_test_pyramid(image, max_im_shrink)
            det = np.row_stack((det0, det1, det2, det3, det4))
            dets = bbox_vote(det)
        else:
            with torch.no_grad():
                dets = detect_face(image, MY_SHRINK)

        inference_time = time.time() - start_time
        fps = 1.0 / inference_time

        # 3. Store Detections for mAP
        for i in range(dets.shape[0]):
            score = dets[i][4]
            # Optional: Filter out very low confidence boxes early
            if score > 0.05: 
                xmin, ymin, xmax, ymax = dets[i][0:4]
                all_detections.append([image_id, score, xmin, ymin, xmax, ymax])

        now += 1
        print('Processed: {}/{} | FPS: {:.2f}'.format(now, len(img_list), fps), end='\r')

    print('\n\n--- Evaluation Results ---')
    
    # Sort all detections by confidence (highest first)
    all_detections.sort(key=lambda x: x[1], reverse=True)

    nd = len(all_detections)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # 4. Calculate True Positives (TP) and False Positives (FP)
    for d in range(nd):
        detection = all_detections[d]
        image_id = detection[0]
        bb = detection[2:]
        
        ovmax = -np.inf
        kmax = -1
        
        gt_boxes = all_ground_truths.get(image_id, [])
        
        for k, gt in enumerate(gt_boxes):
            bbgt = gt['bbox']
            iou = calculate_iou(bb, bbgt)
            if iou > ovmax:
                ovmax = iou
                kmax = k

        # Standard IoU threshold is 0.5
        if ovmax >= 0.5:
            if not gt_boxes[kmax]['matched']:
                tp[d] = 1.
                gt_boxes[kmax]['matched'] = True
            else:
                fp[d] = 1. # Multiple detections for same object
        else:
            fp[d] = 1.

    # 5. Compute Precision, Recall, and mAP
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    
    rec = tp / float(total_gt_boxes) if total_gt_boxes > 0 else 0
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    
    ap = voc_ap(rec, prec)
    
    print(f"Total Ground Truth Objects: {total_gt_boxes}")
    print(f"Total Detections Made: {nd}")
    print(f"mAP @ IoU=0.50: {ap * 100:.2f}%")
