# -*- coding: utf-8 -*-
"""
Per-Category Evaluation Script — DAI-Net
=========================================
Evaluates detection accuracy broken down by object category.

Categories
----------
  Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog,
  Motorbike, People, Table

Metrics per category
--------------------
  • AP @ IoU = 0.50  (Pascal VOC 11-point interpolation)
  • Precision / Recall @ best-F1 threshold
  • Number of GT objects and detections

Usage
-----
    python evaluate_per_category.py
    python evaluate_per_category.py --model yolo_dark
    python evaluate_per_category.py --model dark --weights ./weights/dsfd.pth

Output
------
    result/<tag>_<NNN>_<timestamp>/figures/per_category_metrics.png
    result/<tag>_<NNN>_<timestamp>/reports/per_category_summary.txt
"""

from __future__ import division, absolute_import, print_function

import os
import glob
import time
import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

import argparse
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from models.factory import build_net

# ─── Device ──────────────────────────────────────────────────────────────────
use_cuda = torch.cuda.is_available()
device_name = "CUDA" if use_cuda else "CPU"

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.benchmark = True
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# ─── Paths ───────────────────────────────────────────────────────────────────
IMAGES_DIR      = './dataset/roboflow/test/images/'
ANNOTATIONS_DIR = './dataset/roboflow/test/annotations/'
WEIGHTS_PATH    = './weights/dsfd.pth'
RESULTS_ROOT    = './result/'

# ─── Experiment naming ───────────────────────────────────────────────────────
EXPERIMENT_TAG = 'per_category'

# ─── Inference settings ──────────────────────────────────────────────────────
USE_MULTI_SCALE = True
MY_SHRINK       = 1.0
CONF_THRESH     = 0.01
IOU_THRESH      = 0.50

# ─── Category definitions ────────────────────────────────────────────────────
CATEGORIES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
    'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table',
]

# Map annotation class names (lowercase) to canonical category names.
# Extend this mapping if your annotations use different naming.
CLASS_NAME_MAP = {
    'bicycle':    'Bicycle',
    'bike':       'Bicycle',
    'boat':       'Boat',
    'bottle':     'Bottle',
    'bus':        'Bus',
    'car':        'Car',
    'cat':        'Cat',
    'chair':      'Chair',
    'cup':        'Cup',
    'dog':        'Dog',
    'motorbike':  'Motorbike',
    'motorcycle': 'Motorbike',
    'people':     'People',
    'person':     'People',
    'table':      'Table',
    'diningtable':'Table',
    'dining table':'Table',
}


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def to_chw_bgr(image):
    """HWC RGB  →  CHW BGR."""
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
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


def detect_face(net, img, shrink):
    base = 640
    target_size = int(base * shrink)
    image, lb_scale, pad_left, pad_top = letterbox(img, target_size)

    x = to_chw_bgr(image).astype('float32') / 255.
    x = x[[2, 1, 0], :, :]
    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    y = net.test_forward(x)[0]
    detections = y.data.cpu().numpy()
    px_scale = np.array([target_size, target_size, target_size, target_size])

    boxes, scores = [], []
    for i in range(detections.shape[1]):
        j = 0
        while j < detections.shape[2] and detections[0, i, j, 0] > 0.0:
            pt    = detections[0, i, j, 1:] * px_scale
            score = detections[0, i, j, 0]
            x1 = (pt[0] - pad_left) / lb_scale
            y1 = (pt[1] - pad_top) / lb_scale
            x2 = (pt[2] - pad_left) / lb_scale
            y2 = (pt[3] - pad_top) / lb_scale
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            j += 1

    if len(boxes) == 0:
        return np.array([[0, 0, 0, 0, 0.001]])

    det = np.column_stack((np.array(boxes), np.array(scores)))
    return det


def flip_test(net, image, shrink):
    """Horizontal flip TTA."""
    image_f = cv2.flip(image, 1)
    det_f = detect_face(net, image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def multi_scale_test(net, image, max_im_shrink):
    """Adaptive multi-scale."""
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(net, image, st)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s, detect_face(net, image, 0.75)))
    index = np.where(
        np.maximum(det_s[:, 2] - det_s[:, 0] + 1,
                   det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]

    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(net, image, bt)

    if max_im_shrink > 1.5:
        det_b = np.row_stack((det_b, detect_face(net, image, 1.5)))
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, detect_face(net, image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(net, image, max_im_shrink)))

    if bt > 1:
        index = np.where(
            np.minimum(det_b[:, 2] - det_b[:, 0] + 1,
                       det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(
            np.maximum(det_b[:, 2] - det_b[:, 0] + 1,
                       det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b


def multi_scale_test_pyramid(net, image, max_shrink):
    """Fixed-pyramid multi-scale at [0.25, 1.25, 1.75, 2.25]."""
    det_b = detect_face(net, image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1,
                   det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if st[i] <= max_shrink:
            det_temp = detect_face(net, image, st[i])
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
    """Weighted NMS merge: overlapping boxes averaged by confidence."""
    order_ = det_[:, 4].ravel().argsort()[::-1]
    det_ = det_[order_, :]
    dets_ = np.zeros((0, 5), dtype=np.float32)
    while det_.shape[0] > 0:
        area_ = (det_[:, 2] - det_[:, 0] + 1) * (det_[:, 3] - det_[:, 1] + 1)
        xx1 = np.maximum(det_[0, 0], det_[:, 0])
        yy1 = np.maximum(det_[0, 1], det_[:, 1])
        xx2 = np.minimum(det_[0, 2], det_[:, 2])
        yy2 = np.minimum(det_[0, 3], det_[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o_ = inter / (area_[0] + area_[:] - inter)

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
        except Exception:
            dets_ = det_accu_sum_

    dets_ = dets_[0:750, :]
    return dets_


def parse_voc_xml_with_categories(xml_path):
    """Parse VOC XML and return list of dicts with 'bbox', 'category', 'matched'."""
    tree = ET.parse(xml_path)
    objects = []
    for obj in tree.getroot().findall('object'):
        name = obj.find('name').text.lower().strip()
        category = CLASS_NAME_MAP.get(name, None)
        if category is None:
            # Skip objects whose class is not in our 12 categories
            continue
        b = obj.find('bndbox')
        objects.append({
            'bbox': [float(b.find('xmin').text),
                     float(b.find('ymin').text),
                     float(b.find('xmax').text),
                     float(b.find('ymax').text)],
            'category': category,
            'matched': False,
        })
    return objects


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def voc_ap(rec, prec):
    """Pascal VOC 11-point interpolated Average Precision."""
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        p = np.max(prec[rec >= t]) if np.sum(rec >= t) > 0 else 0.0
        ap += p / 11.0
    return ap


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_type):
    print(f'[INFO] Building model: {model_type}')
    num_classes = 1 if model_type == 'yolo_dark' else 2
    net = build_net('test', num_classes=num_classes, model=model_type)
    net.eval()
    ckpt = torch.load(WEIGHTS_PATH,
                      map_location='cuda' if use_cuda else 'cpu')
    if isinstance(ckpt, dict) and 'weight' in ckpt:
        net.load_state_dict(ckpt['weight'])
    else:
        net.load_state_dict(ckpt)
    if use_cuda:
        net = net.cuda()
    print(f'[INFO] Weights loaded from {WEIGHTS_PATH}')
    print(f'[INFO] Running on: {device_name}')
    return net


# ═══════════════════════════════════════════════════════════════════════════
# Inference loop
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(net):
    img_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, '*.jpg')))
    n = len(img_paths)
    print(f'[INFO] Found {n} test images.')

    all_detections  = []   # [img_id, score, x1, y1, x2, y2]
    all_gts         = {}   # {img_id: [{'bbox': [...], 'category': str, 'matched': False}]}
    fps_list        = []

    for idx, img_path in enumerate(img_paths, 1):
        img_id   = Path(img_path).stem
        xml_path = os.path.join(ANNOTATIONS_DIR, img_id + '.xml')

        # ground truth with category labels
        if os.path.exists(xml_path):
            gts = parse_voc_xml_with_categories(xml_path)
            all_gts[img_id] = gts
        else:
            all_gts[img_id] = []

        # load image
        img = np.array(Image.open(img_path).convert('RGB'))

        # inference
        t0 = time.time()
        with torch.no_grad():
            if USE_MULTI_SCALE:
                max_im_shrink = 2.0
                det0 = detect_face(net, img, MY_SHRINK)
                det1 = flip_test(net, img, MY_SHRINK)
                [det2, det3] = multi_scale_test(net, img, max_im_shrink)
                det4 = multi_scale_test_pyramid(net, img, max_im_shrink)
                det = np.row_stack((det0, det1, det2, det3, det4))
                dets = bbox_vote(det)
            else:
                dets = detect_face(net, img, MY_SHRINK)
        fps = 1.0 / (time.time() - t0)
        fps_list.append(fps)

        # collect detections above threshold
        for i in range(dets.shape[0]):
            score = float(dets[i, 4])
            if score > CONF_THRESH:
                all_detections.append(
                    [img_id, score,
                     float(dets[i, 0]), float(dets[i, 1]),
                     float(dets[i, 2]), float(dets[i, 3])])

        print(f'\r[INFO] Processed {idx}/{n} | FPS {fps:.1f}', end='')

    # Count GT per category
    gt_counts = defaultdict(int)
    for gts in all_gts.values():
        for gt in gts:
            gt_counts[gt['category']] += 1
    total_gt = sum(gt_counts.values())

    print(f'\n[INFO] Total GT boxes   : {total_gt}')
    print(f'[INFO] Total detections : {len(all_detections)}')
    print(f'[INFO] GT per category  :')
    for cat in CATEGORIES:
        print(f'         {cat:12s}: {gt_counts[cat]}')

    return all_detections, all_gts, gt_counts, fps_list


# ═══════════════════════════════════════════════════════════════════════════
# Per-category evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_category(category, all_detections, all_gts, gt_count):
    """
    Evaluate detections against ground truth for a single category.

    Since the model produces class-agnostic detections, each detection is
    matched against ground truth boxes of the given category. A detection
    is a TP if it overlaps (IoU >= threshold) with an unmatched GT box of
    that category; otherwise it is a FP.

    Returns: (recall_array, precision_array, ap, best_f1, best_p, best_r, best_thr)
    """
    if gt_count == 0:
        return (np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0)

    # Reset matched flags
    for gts in all_gts.values():
        for gt in gts:
            gt['matched'] = False

    # Sort detections by confidence (descending)
    sorted_dets = sorted(all_detections, key=lambda x: x[1], reverse=True)
    nd = len(sorted_dets)

    tp_raw = np.zeros(nd)
    fp_raw = np.zeros(nd)
    conf   = np.array([d[1] for d in sorted_dets])

    for d_idx, det in enumerate(sorted_dets):
        img_id = det[0]
        bb     = det[2:]
        gts    = all_gts.get(img_id, [])

        # Only match against GT boxes of this category
        best_iou, best_k = -np.inf, -1
        for k, gt in enumerate(gts):
            if gt['category'] != category:
                continue
            iou = calculate_iou(bb, gt['bbox'])
            if iou > best_iou:
                best_iou, best_k = iou, k

        if best_iou >= IOU_THRESH:
            if not gts[best_k]['matched']:
                tp_raw[d_idx] = 1.
                gts[best_k]['matched'] = True
            else:
                fp_raw[d_idx] = 1.
        else:
            fp_raw[d_idx] = 1.

    cum_tp = np.cumsum(tp_raw)
    cum_fp = np.cumsum(fp_raw)

    rec  = cum_tp / float(gt_count)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)
    ap   = voc_ap(rec, prec)

    # Compute best-F1 at discrete confidence thresholds
    best_f1, best_p, best_r, best_thr = 0.0, 0.0, 0.0, 0.0
    for thr in np.linspace(0.01, 0.95, 95):
        # Reset matched flags
        for gts in all_gts.values():
            for gt in gts:
                gt['matched'] = False

        tp = fp = 0
        for det in sorted_dets:
            if det[1] < thr:
                continue
            img_id = det[0]
            bb     = det[2:]
            gts    = all_gts.get(img_id, [])

            best_iou_t, best_k_t = -np.inf, -1
            for k, gt in enumerate(gts):
                if gt['category'] != category:
                    continue
                iou = calculate_iou(bb, gt['bbox'])
                if iou > best_iou_t:
                    best_iou_t, best_k_t = iou, k

            if best_iou_t >= IOU_THRESH:
                if not gts[best_k_t]['matched']:
                    tp += 1
                    gts[best_k_t]['matched'] = True
                else:
                    fp += 1
            else:
                fp += 1

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / gt_count  if gt_count  > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_p, best_r, best_thr = f1, p, r, thr

    # Reset matched flags for next caller
    for gts in all_gts.values():
        for gt in gts:
            gt['matched'] = False

    return rec, prec, ap, best_f1, best_p, best_r, best_thr


def evaluate_all_categories(all_detections, all_gts, gt_counts):
    """Evaluate each category and return per-category results dict."""
    results = {}
    for cat in CATEGORIES:
        gt_count = gt_counts.get(cat, 0)
        print(f'[INFO] Evaluating category: {cat:12s} (GT={gt_count})')
        rec, prec, ap, best_f1, best_p, best_r, best_thr = \
            evaluate_category(cat, all_detections, all_gts, gt_count)
        results[cat] = {
            'gt_count':  gt_count,
            'ap':        ap,
            'best_f1':   best_f1,
            'precision':  best_p,
            'recall':     best_r,
            'best_thr':  best_thr,
            'rec_curve': rec,
            'prec_curve': prec,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Overall (all-class) evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_overall(all_detections, all_gts, gt_counts):
    """Compute overall AP treating all categories together."""
    total_gt = sum(gt_counts.values())
    if total_gt == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    for gts in all_gts.values():
        for gt in gts:
            gt['matched'] = False

    sorted_dets = sorted(all_detections, key=lambda x: x[1], reverse=True)
    nd = len(sorted_dets)

    tp_raw = np.zeros(nd)
    fp_raw = np.zeros(nd)

    for d_idx, det in enumerate(sorted_dets):
        img_id = det[0]
        bb     = det[2:]
        gts    = all_gts.get(img_id, [])

        best_iou, best_k = -np.inf, -1
        for k, gt in enumerate(gts):
            iou = calculate_iou(bb, gt['bbox'])
            if iou > best_iou:
                best_iou, best_k = iou, k

        if best_iou >= IOU_THRESH:
            if not gts[best_k]['matched']:
                tp_raw[d_idx] = 1.
                gts[best_k]['matched'] = True
            else:
                fp_raw[d_idx] = 1.
        else:
            fp_raw[d_idx] = 1.

    cum_tp = np.cumsum(tp_raw)
    cum_fp = np.cumsum(fp_raw)
    rec  = cum_tp / float(total_gt)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)
    overall_ap = voc_ap(rec, prec)

    # mAP = mean of per-category APs (only categories with GT > 0)
    return overall_ap, rec, prec, total_gt, nd

    for gts in all_gts.values():
        for gt in gts:
            gt['matched'] = False


# ═══════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════

BRAND_COLOR = '#1a73e8'
WARN_COLOR  = '#ea4335'
FONT_TITLE  = 13
FONT_LABEL  = 11
FONT_TICK   = 9

CATEGORY_COLORS = [
    '#4285f4', '#ea4335', '#fbbc05', '#34a853', '#ff6d01', '#46bdc6',
    '#7baaf7', '#f07b72', '#fdd663', '#57bb8a', '#ff9e80', '#78d9e0',
]


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.4)


def build_figure(cat_results, fps_list):
    """Build a publication-ready figure with per-category metrics."""

    # Prepare data sorted by AP (descending)
    cats_with_gt = [c for c in CATEGORIES if cat_results[c]['gt_count'] > 0]
    cats_sorted = sorted(cats_with_gt, key=lambda c: cat_results[c]['ap'], reverse=True)
    cats_no_gt  = [c for c in CATEGORIES if cat_results[c]['gt_count'] == 0]

    # ── layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), facecolor='#f8f9fa')
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            left=0.06, right=0.97,
                            top=0.88, bottom=0.08,
                            hspace=0.40, wspace=0.35)

    ax_ap   = fig.add_subplot(gs[0, 0])    # AP bar chart
    ax_f1   = fig.add_subplot(gs[0, 1])    # F1 bar chart
    ax_pr   = fig.add_subplot(gs[0, 2])    # Precision / Recall bar chart
    ax_gt   = fig.add_subplot(gs[1, 0])    # GT count bar chart
    ax_fps  = fig.add_subplot(gs[1, 1])    # FPS distribution
    ax_tab  = fig.add_subplot(gs[1, 2])    # Summary table
    ax_tab.axis('off')

    # ── 1. AP per category ───────────────────────────────────────────────
    if cats_sorted:
        aps = [cat_results[c]['ap'] * 100 for c in cats_sorted]
        colors = [CATEGORY_COLORS[CATEGORIES.index(c)] for c in cats_sorted]
        bars = ax_ap.barh(range(len(cats_sorted)), aps, color=colors,
                          edgecolor='white', linewidth=0.5)
        ax_ap.set_yticks(range(len(cats_sorted)))
        ax_ap.set_yticklabels(cats_sorted)
        ax_ap.set_xlim([0, 105])
        ax_ap.invert_yaxis()
        for i, v in enumerate(aps):
            ax_ap.text(v + 1, i, f'{v:.1f}%', va='center',
                       fontsize=FONT_TICK, fontweight='bold')
    _style_ax(ax_ap, 'AP @ IoU=0.50 per Category', 'AP (%)', '')

    # ── 2. F1 per category ───────────────────────────────────────────────
    if cats_sorted:
        f1s = [cat_results[c]['best_f1'] * 100 for c in cats_sorted]
        colors = [CATEGORY_COLORS[CATEGORIES.index(c)] for c in cats_sorted]
        ax_f1.barh(range(len(cats_sorted)), f1s, color=colors,
                   edgecolor='white', linewidth=0.5)
        ax_f1.set_yticks(range(len(cats_sorted)))
        ax_f1.set_yticklabels(cats_sorted)
        ax_f1.set_xlim([0, 105])
        ax_f1.invert_yaxis()
        for i, v in enumerate(f1s):
            ax_f1.text(v + 1, i, f'{v:.1f}%', va='center',
                       fontsize=FONT_TICK, fontweight='bold')
    _style_ax(ax_f1, 'Best F1-Score per Category', 'F1 (%)', '')

    # ── 3. Precision & Recall per category ────────────────────────────────
    if cats_sorted:
        precs = [cat_results[c]['precision'] * 100 for c in cats_sorted]
        recs  = [cat_results[c]['recall'] * 100 for c in cats_sorted]
        y_pos = np.arange(len(cats_sorted))
        bar_h = 0.35
        ax_pr.barh(y_pos - bar_h/2, precs, bar_h, label='Precision',
                   color=BRAND_COLOR, alpha=0.85, edgecolor='white')
        ax_pr.barh(y_pos + bar_h/2, recs, bar_h, label='Recall',
                   color=WARN_COLOR, alpha=0.85, edgecolor='white')
        ax_pr.set_yticks(y_pos)
        ax_pr.set_yticklabels(cats_sorted)
        ax_pr.set_xlim([0, 105])
        ax_pr.invert_yaxis()
        ax_pr.legend(fontsize=FONT_TICK, loc='lower right')
    _style_ax(ax_pr, 'Precision & Recall @ Best F1', 'Score (%)', '')

    # ── 4. GT count per category ──────────────────────────────────────────
    all_cats = cats_sorted + cats_no_gt
    gt_vals  = [cat_results[c]['gt_count'] for c in all_cats]
    colors   = [CATEGORY_COLORS[CATEGORIES.index(c)] for c in all_cats]
    ax_gt.barh(range(len(all_cats)), gt_vals, color=colors,
               edgecolor='white', linewidth=0.5)
    ax_gt.set_yticks(range(len(all_cats)))
    ax_gt.set_yticklabels(all_cats)
    ax_gt.invert_yaxis()
    for i, v in enumerate(gt_vals):
        ax_gt.text(v + 0.5, i, str(v), va='center',
                   fontsize=FONT_TICK, fontweight='bold')
    _style_ax(ax_gt, 'Ground Truth Objects per Category', 'Count', '')

    # ── 5. FPS distribution ──────────────────────────────────────────────
    fps_arr  = np.array(fps_list)
    mean_fps = fps_arr.mean()
    bins = min(30, max(10, len(fps_arr) // 5))
    ax_fps.hist(fps_arr, bins=bins, color=BRAND_COLOR,
                alpha=0.75, edgecolor='white', linewidth=0.5)
    ax_fps.axvline(mean_fps, color=WARN_COLOR, lw=2,
                   linestyle='--', label=f'Mean = {mean_fps:.2f} FPS')
    ax_fps.legend(fontsize=FONT_LABEL, framealpha=0.7)
    _style_ax(ax_fps, f'Per-Image FPS Distribution ({device_name})',
              'FPS', 'Count')

    # ── 6. Summary table ─────────────────────────────────────────────────
    # Compute mAP (mean of per-category APs where GT > 0)
    ap_values = [cat_results[c]['ap'] for c in cats_with_gt]
    mAP = np.mean(ap_values) if ap_values else 0.0

    rows = [
        ['Category',    'GT', 'AP(%)',  'F1(%)', 'Prec(%)', 'Rec(%)'],
    ]
    for cat in CATEGORIES:
        r = cat_results[cat]
        rows.append([
            cat,
            str(r['gt_count']),
            f"{r['ap']*100:.1f}" if r['gt_count'] > 0 else '-',
            f"{r['best_f1']*100:.1f}" if r['gt_count'] > 0 else '-',
            f"{r['precision']*100:.1f}" if r['gt_count'] > 0 else '-',
            f"{r['recall']*100:.1f}" if r['gt_count'] > 0 else '-',
        ])
    rows.append(['mAP (all)', '', f'{mAP*100:.1f}', '', '', ''])

    row_h = 1.0 / len(rows)
    ncols = len(rows[0])
    col_w = [0.22, 0.10, 0.14, 0.14, 0.14, 0.14]
    x_start = 0.02

    for r_idx, row in enumerate(rows):
        y = 1.0 - (r_idx + 0.5) * row_h
        if r_idx == 0:
            bg = '#e8f0fe'
        elif r_idx == len(rows) - 1:
            bg = '#e6f4ea'
        else:
            bg = '#ffffff' if r_idx % 2 else '#f1f3f4'

        total_w = sum(col_w)
        rect = FancyBboxPatch((x_start, y - row_h * 0.5),
                              total_w, row_h,
                              boxstyle='round,pad=0.002',
                              facecolor=bg, edgecolor='#dadce0',
                              linewidth=0.6,
                              transform=ax_tab.transAxes, clip_on=False)
        ax_tab.add_patch(rect)

        cx = x_start
        for c_idx, cell in enumerate(row):
            fw = 'bold' if r_idx == 0 or r_idx == len(rows) - 1 else 'normal'
            color = '#202124'
            if r_idx > 0 and r_idx < len(rows) - 1 and c_idx == 2:
                color = BRAND_COLOR  # highlight AP values
            ax_tab.text(cx + 0.01, y, cell,
                        transform=ax_tab.transAxes,
                        va='center', ha='left',
                        fontsize=8, fontweight=fw, color=color)
            cx += col_w[c_idx]

    ax_tab.set_title('Per-Category Results',
                     fontsize=FONT_TITLE, fontweight='bold', pad=10)

    # ── Main title ───────────────────────────────────────────────────────
    fig.suptitle(
        'DAI-Net Per-Category Evaluation  |  Detection Accuracy by Object Class',
        fontsize=16, fontweight='bold', color='#202124', y=0.96
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Text report
# ═══════════════════════════════════════════════════════════════════════════

def save_text_report(cat_results, fps_list, all_detections, out_path):
    fps_arr = np.array(fps_list)
    total_gt = sum(r['gt_count'] for r in cat_results.values())
    cats_with_gt = [c for c in CATEGORIES if cat_results[c]['gt_count'] > 0]
    ap_values = [cat_results[c]['ap'] for c in cats_with_gt]
    mAP = np.mean(ap_values) if ap_values else 0.0

    lines = [
        '=' * 72,
        'DAI-Net Per-Category Evaluation — Detection Accuracy by Object Class',
        '=' * 72,
        '',
        f'  {"Category":<14s} {"GT":>5s} {"AP(%)":>8s} {"F1(%)":>8s} '
        f'{"Prec(%)":>8s} {"Rec(%)":>8s} {"Thr":>6s}',
        '-' * 72,
    ]

    for cat in CATEGORIES:
        r = cat_results[cat]
        if r['gt_count'] > 0:
            lines.append(
                f'  {cat:<14s} {r["gt_count"]:>5d} {r["ap"]*100:>8.2f} '
                f'{r["best_f1"]*100:>8.2f} {r["precision"]*100:>8.2f} '
                f'{r["recall"]*100:>8.2f} {r["best_thr"]:>6.2f}'
            )
        else:
            lines.append(
                f'  {cat:<14s} {r["gt_count"]:>5d} {"N/A":>8s} '
                f'{"N/A":>8s} {"N/A":>8s} {"N/A":>8s} {"N/A":>6s}'
            )

    lines += [
        '-' * 72,
        f'  {"mAP (all)":<14s} {total_gt:>5d} {mAP*100:>8.2f}',
        '',
        f'  Mean FPS                 : {fps_arr.mean():.2f}',
        f'  Median FPS               : {np.median(fps_arr):.2f}',
        f'  Total test images        : {len(fps_list)}',
        f'  Total GT objects         : {total_gt}',
        f'  Total detections (raw)   : {len(all_detections)}',
        '',
        f'  Device                   : {device_name}',
        f'  Multi-scale              : {USE_MULTI_SCALE}',
        f'  IoU threshold            : {IOU_THRESH}',
        f'  Confidence threshold     : {CONF_THRESH}',
        '=' * 72,
    ]
    report = '\n'.join(lines)
    print('\n' + report)
    with open(out_path, 'w') as f:
        f.write(report + '\n')
    print(f'[INFO] Text report saved → {out_path}')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def make_run_dir(results_root, tag):
    os.makedirs(results_root, exist_ok=True)
    existing = [
        d for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d))
        and d.startswith(tag + '_')
    ]
    numbers = []
    for name in existing:
        parts = name.split('_')
        if len(parts) >= 2 and parts[2].isdigit():
            numbers.append(int(parts[2]))
    nxt = (max(numbers) + 1) if numbers else 1

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name  = f'{tag}_{nxt:03d}_{timestamp}'
    run_dir   = os.path.join(results_root, run_name)

    figures_dir = os.path.join(run_dir, 'figures')
    reports_dir = os.path.join(run_dir, 'reports')
    os.makedirs(figures_dir)
    os.makedirs(reports_dir)

    print(f'[INFO] Run directory → {run_dir}')
    return run_dir, figures_dir, reports_dir


if __name__ == '__main__':
    _WEIGHT_DEFAULTS = {
        'dark':      './weights/dsfd.pth',
        'yolo_dark': './weights/yolo_dark/dsfd.pth',
    }
    parser = argparse.ArgumentParser(
        description='DAI-Net per-category evaluation')
    parser.add_argument('--model',   default='yolo_dark',
                        choices=['dark', 'yolo_dark'],
                        help='Model architecture (default: yolo_dark)')
    parser.add_argument('--weights', default=None,
                        help='Path to .pth weights file')
    parser.add_argument('--tag',     default=None,
                        help='Experiment tag for result subfolder')
    cli = parser.parse_args()

    WEIGHTS_PATH   = cli.weights or _WEIGHT_DEFAULTS[cli.model]
    EXPERIMENT_TAG = cli.tag or f'per_category_{cli.model}'

    # 0. Create run subfolder
    run_dir, figures_dir, reports_dir = make_run_dir(RESULTS_ROOT, EXPERIMENT_TAG)
    OUT_FIGURE = os.path.join(figures_dir, 'per_category_metrics.png')
    OUT_TXT    = os.path.join(reports_dir, 'per_category_summary.txt')

    # 1. Load model
    net = load_model(cli.model)

    # 2. Run inference
    all_dets, all_gts, gt_counts, fps_list = run_inference(net)

    # 3. Per-category evaluation
    print('[INFO] Running per-category evaluation...')
    cat_results = evaluate_all_categories(all_dets, all_gts, gt_counts)

    # 4. Compute mAP
    cats_with_gt = [c for c in CATEGORIES if cat_results[c]['gt_count'] > 0]
    ap_values = [cat_results[c]['ap'] for c in cats_with_gt]
    mAP = np.mean(ap_values) if ap_values else 0.0
    print(f'\n[INFO] mAP @ IoU=0.50 = {mAP*100:.2f}%')

    # 5. Save text report
    save_text_report(cat_results, fps_list, all_dets, OUT_TXT)

    # 6. Build & save figure
    print('[INFO] Generating per-category figure...')
    fig = build_figure(cat_results, fps_list)
    fig.savefig(OUT_FIGURE, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[INFO] Figure saved  → {OUT_FIGURE}')
    print('[DONE]')
