# -*- coding: utf-8 -*-
"""
EXDARK Evaluation Script — DAI-Net (Zero-Shot Low-Light Object Detection)
==========================================================================
Evaluates the trained model on the EXDARK test split (Train/Val/Test = 3)
with identical metrics to evaluate_baseline.py, plus a per-category AP
breakdown across the 12 EXDARK object classes.

EXDARK annotation format (bbGt version=3)
------------------------------------------
  <Class> <x> <y> <w> <h> ...   (x,y = top-left corner; w,h = box size)

Dataset layout expected
-----------------------
  dataset/EXDARK/
    ExDark/             <-- images, one sub-folder per category
    ExDark_Annno/       <-- annotations, mirrored folder structure
    imageclasslist.txt  <-- split & metadata (col-5: 1=train, 2=val, 3=test)

Metrics produced
----------------
  • Overall Precision-Recall curve  (Pascal VOC 11-point AP, IoU ≥ 0.50)
  • Overall AP & mAP (mean of per-category APs)
  • F1-score / Precision / Recall curve vs. confidence threshold
  • Per-category AP bar chart (12 classes)
  • Per-image FPS distribution
  • Detection confidence score distribution
  • Summary table

Usage
-----
    python evaluate_exdark.py [--model yolo_dark] [--weights PATH] [--tag TAG]
    python evaluate_exdark.py --no_multi_scale     # faster single-scale run

Output
------
    result/exdark_<NNN>_<timestamp>/
        figures/exdark_metrics.png
        reports/exdark_summary.txt
"""

from __future__ import division, absolute_import, print_function

import os
import glob
import time
import datetime
import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from models.factory import build_net

# ─── Device ──────────────────────────────────────────────────────────────────
use_cuda    = torch.cuda.is_available()
device_name = 'CUDA' if use_cuda else 'CPU'

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.benchmark = True
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# ─── Dataset paths ───────────────────────────────────────────────────────────
EXDARK_ROOT    = './dataset/EXDARK'
IMAGES_DIR     = os.path.join(EXDARK_ROOT, 'ExDark')
ANNOS_DIR      = os.path.join(EXDARK_ROOT, 'ExDark_Annno')
CLASSLIST_PATH = os.path.join(EXDARK_ROOT, 'imageclasslist.txt')

RESULTS_ROOT   = './result/'
EXPERIMENT_TAG = 'exdark'

# ─── EXDARK 12 categories (1-indexed as in imageclasslist.txt) ───────────────
CLASSES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
    'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table',
]
CLASS_BY_IDX  = {i + 1: c for i, c in enumerate(CLASSES)}   # int → str
CLASS_BY_NAME = {c: i + 1 for i, c in enumerate(CLASSES)}   # str → int

# ─── Inference settings ──────────────────────────────────────────────────────
USE_MULTI_SCALE = True   # set False via --no_multi_scale for a faster run
MY_SHRINK       = 1.0
CONF_THRESH     = 0.01
IOU_THRESH      = 0.50


# ═══════════════════════════════════════════════════════════════════════════
# Image helpers  (identical to evaluate_baseline.py)
# ═══════════════════════════════════════════════════════════════════════════

def to_chw_bgr(image):
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    image = image[[2, 1, 0], :, :]
    return image


def letterbox(img, target_size=640):
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=img.dtype)
    pad_top  = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, pad_left, pad_top


def detect_face(net, img, shrink):
    base        = 640
    target_size = int(base * shrink)
    image, lb_scale, pad_left, pad_top = letterbox(img, target_size)

    x = to_chw_bgr(image).astype('float32') / 255.
    x = x[[2, 1, 0], :, :]
    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    y          = net.test_forward(x)[0]
    detections = y.data.cpu().numpy()
    px_scale   = np.array([target_size, target_size, target_size, target_size])

    boxes, scores = [], []
    for i in range(detections.shape[1]):
        j = 0
        while j < detections.shape[2] and detections[0, i, j, 0] > 0.0:
            pt    = detections[0, i, j, 1:] * px_scale
            score = detections[0, i, j, 0]
            x1 = (pt[0] - pad_left) / lb_scale
            y1 = (pt[1] - pad_top)  / lb_scale
            x2 = (pt[2] - pad_left) / lb_scale
            y2 = (pt[3] - pad_top)  / lb_scale
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            j += 1

    if len(boxes) == 0:
        return np.array([[0, 0, 0, 0, 0.001]])
    return np.column_stack((np.array(boxes), np.array(scores)))


def flip_test(net, image, shrink):
    image_f = cv2.flip(image, 1)
    det_f   = detect_face(net, image_f, shrink)
    det_t   = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def multi_scale_test(net, image, max_im_shrink):
    st    = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(net, image, st)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s, detect_face(net, image, 0.75)))
    idx   = np.where(
        np.maximum(det_s[:, 2] - det_s[:, 0] + 1,
                   det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[idx, :]

    bt    = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(net, image, bt)
    if max_im_shrink > 1.5:
        det_b = np.row_stack((det_b, detect_face(net, image, 1.5)))
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, detect_face(net, image, bt)))
            bt   *= 2
        det_b = np.row_stack((det_b, detect_face(net, image, max_im_shrink)))

    if bt > 1:
        idx   = np.where(
            np.minimum(det_b[:, 2] - det_b[:, 0] + 1,
                       det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
    else:
        idx   = np.where(
            np.maximum(det_b[:, 2] - det_b[:, 0] + 1,
                       det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
    det_b = det_b[idx, :]
    return det_s, det_b


def multi_scale_test_pyramid(net, image, max_shrink):
    det_b = detect_face(net, image, 0.25)
    idx   = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1,
                   det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
    det_b = det_b[idx, :]
    for s in [1.25, 1.75, 2.25]:
        if s > max_shrink:
            break
        det_t = detect_face(net, image, s)
        if s > 1:
            idx   = np.where(
                np.minimum(det_t[:, 2] - det_t[:, 0] + 1,
                           det_t[:, 3] - det_t[:, 1] + 1) < 100)[0]
        else:
            idx   = np.where(
                np.maximum(det_t[:, 2] - det_t[:, 0] + 1,
                           det_t[:, 3] - det_t[:, 1] + 1) > 30)[0]
        det_b = np.row_stack((det_b, det_t[idx, :]))
    return det_b


def bbox_vote(det_):
    order_ = det_[:, 4].ravel().argsort()[::-1]
    det_   = det_[order_, :]
    dets_  = np.zeros((0, 5), dtype=np.float32)
    while det_.shape[0] > 0:
        area_  = (det_[:, 2] - det_[:, 0] + 1) * (det_[:, 3] - det_[:, 1] + 1)
        xx1    = np.maximum(det_[0, 0], det_[:, 0])
        yy1    = np.maximum(det_[0, 1], det_[:, 1])
        xx2    = np.minimum(det_[0, 2], det_[:, 2])
        yy2    = np.minimum(det_[0, 3], det_[:, 3])
        inter  = np.maximum(0., xx2 - xx1 + 1) * np.maximum(0., yy2 - yy1 + 1)
        o_     = inter / (area_[0] + area_[:] - inter)

        merge_idx = np.where(o_ >= 0.5)[0]
        det_accu  = det_[merge_idx, :]
        det_       = np.delete(det_, merge_idx, 0)

        if merge_idx.shape[0] <= 1:
            continue
        det_accu[:, 0:4] *= np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        merged    = np.zeros((1, 5))
        merged[0, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        merged[0, 4]   = max_score
        try:
            dets_ = np.row_stack((dets_, merged))
        except Exception:
            dets_ = merged

    return dets_[:750, :]


# ═══════════════════════════════════════════════════════════════════════════
# EXDARK-specific helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_test_split():
    """
    Parse imageclasslist.txt and return metadata for test images (split == 3).

    Returns
    -------
    list of dict with keys:
        img_path   : absolute path to image
        anno_path  : absolute path to annotation .txt
        category   : class name string
        light      : illumination level (1-10)
    """
    records = []
    with open(CLASSLIST_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Name'):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            fname    = parts[0]
            cls_idx  = int(parts[1])
            light    = int(parts[2])
            split    = int(parts[4])
            if split != 3:
                continue
            category  = CLASS_BY_IDX.get(cls_idx)
            if category is None:
                continue
            img_path  = os.path.join(IMAGES_DIR, category, fname)
            anno_path = os.path.join(ANNOS_DIR,  category, fname + '.txt')
            if os.path.exists(img_path):
                records.append(dict(
                    img_path=img_path,
                    anno_path=anno_path,
                    category=category,
                    light=light,
                ))
    return records


def parse_exdark_anno(anno_path):
    """
    Parse an ExDark bbGt annotation file.

    Format per data line:
        <Class> <x> <y> <w> <h> ...

    where (x, y) is the top-left corner and (w, h) is the box size.
    Returns a list of dicts: {class, bbox [x1, y1, x2, y2], matched}.
    """
    bboxes = []
    if not os.path.exists(anno_path):
        return bboxes
    with open(anno_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_name = parts[0]
            x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            if w <= 0 or h <= 0:
                continue
            bboxes.append({
                'class':   cls_name,
                'bbox':    [x, y, x + w, y + h],
                'matched': False,
            })
    return bboxes


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0., xB - xA) * max(0., yB - yA)
    if inter == 0:
        return 0.
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def voc_ap(rec, prec):
    """Pascal VOC 11-point interpolated Average Precision."""
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        ap += (np.max(prec[rec >= t]) if np.sum(rec >= t) > 0 else 0.) / 11.
    return ap


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_type, weights_path):
    print(f'[INFO] Building model : {model_type}')
    num_classes = 1 if model_type == 'yolo_dark' else 2
    net = build_net('test', num_classes=num_classes, model=model_type)
    net.eval()
    ckpt = torch.load(weights_path, map_location='cuda' if use_cuda else 'cpu')
    if isinstance(ckpt, dict) and 'weight' in ckpt:
        net.load_state_dict(ckpt['weight'])
    else:
        net.load_state_dict(ckpt)
    if use_cuda:
        net = net.cuda()
    print(f'[INFO] Weights loaded : {weights_path}')
    print(f'[INFO] Running on     : {device_name}')
    return net


# ═══════════════════════════════════════════════════════════════════════════
# Inference loop
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(net, records):
    """
    Run inference over all test images.

    Returns
    -------
    all_detections : list of [img_id, score, x1, y1, x2, y2]
    all_gts        : dict {img_id: [{'class', 'bbox', 'matched'}]}
    total_gt       : int
    fps_list       : list of float
    """
    n               = len(records)
    all_detections  = []
    all_gts         = {}
    total_gt        = 0
    fps_list        = []

    for idx, rec in enumerate(records, 1):
        img_path  = rec['img_path']
        anno_path = rec['anno_path']
        img_id    = Path(img_path).name   # e.g. '2015_00001.png'

        gts = parse_exdark_anno(anno_path)
        all_gts[img_id] = gts
        total_gt       += len(gts)

        img = np.array(Image.open(img_path).convert('RGB'))

        t0 = time.time()
        with torch.no_grad():
            if USE_MULTI_SCALE:
                max_im_shrink = 2.0
                det0 = detect_face(net, img, MY_SHRINK)
                det1 = flip_test(net, img, MY_SHRINK)
                [det2, det3] = multi_scale_test(net, img, max_im_shrink)
                det4 = multi_scale_test_pyramid(net, img, max_im_shrink)
                det  = np.row_stack((det0, det1, det2, det3, det4))
                dets = bbox_vote(det)
            else:
                dets = detect_face(net, img, MY_SHRINK)
        fps = 1.0 / (time.time() - t0)
        fps_list.append(fps)

        for i in range(dets.shape[0]):
            score = float(dets[i, 4])
            if score > CONF_THRESH:
                all_detections.append(
                    [img_id, score,
                     float(dets[i, 0]), float(dets[i, 1]),
                     float(dets[i, 2]), float(dets[i, 3])])

        print(f'\r[INFO] {idx}/{n} | FPS {fps:.1f} | dets {len(all_detections)}', end='')

    print(f'\n[INFO] Total GT boxes   : {total_gt}')
    print(f'[INFO] Total detections : {len(all_detections)}')
    return all_detections, all_gts, total_gt, fps_list


# ═══════════════════════════════════════════════════════════════════════════
# Overall evaluation (class-agnostic — pools all GT regardless of category)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_overall(all_detections, all_gts, total_gt):
    """PR curve & AP treating all categories as one class."""
    all_detections.sort(key=lambda x: x[1], reverse=True)
    nd     = len(all_detections)
    tp_raw = np.zeros(nd)
    fp_raw = np.zeros(nd)

    for d_idx, det in enumerate(all_detections):
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
    rec    = cum_tp / float(total_gt) if total_gt > 0 else np.zeros(nd)
    prec   = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)
    ap     = voc_ap(rec, prec)

    # reset for subsequent callers
    for gts in all_gts.values():
        for gt in gts:
            gt['matched'] = False

    return rec, prec, ap


# ═══════════════════════════════════════════════════════════════════════════
# Per-category evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_category(category, all_detections, all_gts):
    """
    Compute AP for a single EXDARK category.
    Detections are matched only against GT boxes of the given category.
    """
    total_gt_cat = sum(
        sum(1 for gt in gts if gt['class'] == category)
        for gts in all_gts.values()
    )
    if total_gt_cat == 0:
        return 0., 0., 0., 0., 0.

    # Reset matched flags (category-scoped)
    for gts in all_gts.values():
        for gt in gts:
            gt['matched'] = False

    nd     = len(all_detections)
    tp_raw = np.zeros(nd)
    fp_raw = np.zeros(nd)

    for d_idx, det in enumerate(all_detections):
        img_id   = det[0]
        bb       = det[2:]
        cat_gts  = [gt for gt in all_gts.get(img_id, []) if gt['class'] == category]

        best_iou, best_k = -np.inf, -1
        for k, gt in enumerate(cat_gts):
            iou = calculate_iou(bb, gt['bbox'])
            if iou > best_iou:
                best_iou, best_k = iou, k

        if best_iou >= IOU_THRESH:
            if not cat_gts[best_k]['matched']:
                tp_raw[d_idx] = 1.
                cat_gts[best_k]['matched'] = True
            else:
                fp_raw[d_idx] = 1.
        else:
            fp_raw[d_idx] = 1.

    cum_tp = np.cumsum(tp_raw)
    cum_fp = np.cumsum(fp_raw)
    rec    = cum_tp / float(total_gt_cat)
    prec   = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)
    ap     = voc_ap(rec, prec)

    # Best F1 for this category
    f1_vals = 2 * prec * rec / np.maximum(prec + rec, np.finfo(np.float64).eps)
    best_f1  = float(np.max(f1_vals)) if len(f1_vals) > 0 else 0.
    best_idx = int(np.argmax(f1_vals))
    best_p   = float(prec[best_idx])
    best_r   = float(rec[best_idx])

    return ap, best_f1, best_p, best_r, total_gt_cat


def evaluate_all_categories(all_detections, all_gts):
    """Return per-category AP dict and mAP."""
    # detections must already be sorted by confidence (done in evaluate_overall)
    results = {}
    for cat in CLASSES:
        ap, f1, p, r, n_gt = evaluate_category(cat, all_detections, all_gts)
        results[cat] = dict(ap=ap, f1=f1, precision=p, recall=r, n_gt=n_gt)
        print(f'  {cat:<12} AP={ap*100:.1f}%  F1={f1:.3f}  GT={n_gt}')

    valid_aps = [v['ap'] for v in results.values() if v['n_gt'] > 0]
    mean_ap   = float(np.mean(valid_aps)) if valid_aps else 0.
    return results, mean_ap


# ═══════════════════════════════════════════════════════════════════════════
# F1 sweep
# ═══════════════════════════════════════════════════════════════════════════

def compute_f1_curve(all_detections, all_gts, total_gt):
    thresholds       = np.linspace(0.01, 0.95, 95)
    f1_list, p_list, r_list = [], [], []

    for thr in thresholds:
        for gts in all_gts.values():
            for gt in gts:
                gt['matched'] = False

        tp = fp = 0
        for det in all_detections:
            if det[1] < thr:
                continue
            img_id = det[0]; bb = det[2:]
            gts    = all_gts.get(img_id, [])
            best_iou, best_k = -np.inf, -1
            for k, gt in enumerate(gts):
                iou = calculate_iou(bb, gt['bbox'])
                if iou > best_iou:
                    best_iou, best_k = iou, k
            if best_iou >= IOU_THRESH:
                if not gts[best_k]['matched']:
                    tp += 1; gts[best_k]['matched'] = True
                else:
                    fp += 1
            else:
                fp += 1

        p  = tp / (tp + fp)   if (tp + fp) > 0 else 0.
        r  = tp / total_gt    if total_gt  > 0 else 0.
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.
        f1_list.append(f1); p_list.append(p); r_list.append(r)

    for gts in all_gts.values():
        for gt in gts:
            gt['matched'] = False

    return thresholds, np.array(f1_list), np.array(p_list), np.array(r_list)


# ═══════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════

BRAND_COLOR = '#1a73e8'
WARN_COLOR  = '#ea4335'
GREEN_COLOR = '#34a853'
FONT_TITLE  = 13
FONT_LABEL  = 11
FONT_TICK   = 9

CATEGORY_COLORS = [
    '#4285f4', '#ea4335', '#fbbc05', '#34a853',
    '#ff6d00', '#46bdc6', '#7c4dff', '#0d47a1',
    '#e64a19', '#1b5e20', '#880e4f', '#37474f',
]


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.4)


def build_figure(rec, prec, ap, mean_ap,
                 thresholds, f1_arr, p_arr, r_arr,
                 fps_list, all_detections, cat_results, n_test):

    fig = plt.figure(figsize=(20, 13), facecolor='#f8f9fa')
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            left=0.06, right=0.97,
                            top=0.88, bottom=0.09,
                            hspace=0.48, wspace=0.35)

    ax_pr   = fig.add_subplot(gs[0, 0])
    ax_f1   = fig.add_subplot(gs[0, 1])
    ax_fps  = fig.add_subplot(gs[0, 2])
    ax_cat  = fig.add_subplot(gs[1, 0])
    ax_conf = fig.add_subplot(gs[1, 1])
    ax_tab  = fig.add_subplot(gs[1, 2])
    ax_tab.axis('off')

    # ── 1. PR curve ──────────────────────────────────────────────────────
    ax_pr.plot(rec, prec, color=BRAND_COLOR, lw=2,
               label=f'Overall AP = {ap*100:.1f}%')
    ax_pr.fill_between(rec, prec, alpha=0.12, color=BRAND_COLOR)
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) > 0:
            ax_pr.plot(t, np.max(prec[rec >= t]), 'o',
                       color=BRAND_COLOR, markersize=4, alpha=0.7)
    ax_pr.set_xlim([0, 1]); ax_pr.set_ylim([0, 1.05])
    ax_pr.legend(fontsize=FONT_LABEL, framealpha=0.7)
    _style_ax(ax_pr, 'Precision-Recall Curve  (IoU ≥ 0.50)',
              'Recall', 'Precision')

    # ── 2. F1 / P / R vs confidence ──────────────────────────────────────
    best_idx = int(np.argmax(f1_arr))
    best_thr = thresholds[best_idx]
    best_f1  = f1_arr[best_idx]

    ax_f1.plot(thresholds, f1_arr, lw=2,   color=GREEN_COLOR, label='F1')
    ax_f1.plot(thresholds, p_arr,  lw=1.5, color=BRAND_COLOR,
               linestyle='--', label='Precision')
    ax_f1.plot(thresholds, r_arr,  lw=1.5, color=WARN_COLOR,
               linestyle=':',  label='Recall')
    ax_f1.axvline(best_thr, color=GREEN_COLOR, lw=1.2,
                  linestyle='-.', alpha=0.8,
                  label=f'Best conf = {best_thr:.2f}')
    ax_f1.scatter([best_thr], [best_f1], color=GREEN_COLOR, zorder=5, s=60)
    ax_f1.annotate(f'F1={best_f1:.3f}',
                   xy=(best_thr, best_f1),
                   xytext=(best_thr + 0.06, best_f1 - 0.07),
                   fontsize=9, color=GREEN_COLOR,
                   arrowprops=dict(arrowstyle='->', color=GREEN_COLOR, lw=1))
    ax_f1.set_xlim([0.05, 0.95]); ax_f1.set_ylim([0, 1.05])
    ax_f1.legend(fontsize=9, framealpha=0.7)
    _style_ax(ax_f1, 'F1 / Precision / Recall vs Confidence',
              'Confidence Threshold', 'Score')

    # ── 3. FPS distribution ──────────────────────────────────────────────
    fps_arr  = np.array(fps_list)
    mean_fps = fps_arr.mean()

    bins = min(30, max(10, len(fps_arr) // 10))
    ax_fps.hist(fps_arr, bins=bins, color=BRAND_COLOR,
                alpha=0.75, edgecolor='white', linewidth=0.5)
    ax_fps.axvline(mean_fps, color=WARN_COLOR, lw=2, linestyle='--',
                   label=f'Mean = {mean_fps:.1f} FPS')
    ax_fps.legend(fontsize=FONT_LABEL, framealpha=0.7)
    _style_ax(ax_fps, f'Per-Image FPS Distribution  ({device_name})',
              'FPS', 'Count')

    # ── 4. Per-category AP bars ──────────────────────────────────────────
    sorted_cats = sorted(cat_results.items(), key=lambda kv: kv[1]['ap'])
    cat_names   = [kv[0] for kv in sorted_cats]
    cat_aps     = [kv[1]['ap'] * 100 for kv in sorted_cats]
    colors      = [CATEGORY_COLORS[CLASSES.index(c) % len(CATEGORY_COLORS)]
                   for c in cat_names]

    bars = ax_cat.barh(cat_names, cat_aps, color=colors,
                       edgecolor='white', height=0.65)
    for bar, val in zip(bars, cat_aps):
        ax_cat.text(min(val + 0.5, 99), bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', ha='left',
                    fontsize=8, color='#202124')
    ax_cat.set_xlim([0, 100])
    ax_cat.axvline(mean_ap * 100, color='#202124', lw=1.2, linestyle='--',
                   label=f'mAP = {mean_ap*100:.1f}%')
    ax_cat.legend(fontsize=9, framealpha=0.7)
    _style_ax(ax_cat, 'Per-Category AP (IoU ≥ 0.50)',
              'AP (%)', '')
    ax_cat.grid(axis='x', linestyle='--', alpha=0.4)
    ax_cat.grid(axis='y', visible=False)

    # ── 5. Confidence score histogram ─────────────────────────────────────
    scores = np.array([d[1] for d in all_detections])
    ax_conf.hist(scores, bins=40, color='#fbbc05',
                 alpha=0.85, edgecolor='white', linewidth=0.5)
    ax_conf.axvline(CONF_THRESH, color=WARN_COLOR, lw=1.8, linestyle='--',
                    label=f'Threshold = {CONF_THRESH}')
    ax_conf.legend(fontsize=FONT_LABEL, framealpha=0.7)
    _style_ax(ax_conf, 'Detection Confidence Distribution',
              'Confidence Score', 'Count')

    # ── 6. Summary table ─────────────────────────────────────────────────
    best_p_val = float(p_arr[best_idx])
    best_r_val = float(r_arr[best_idx])

    rows = [
        ['Metric',                     'Value'],
        ['Overall AP (pooled)',         f'{ap*100:.2f}%'],
        ['mAP (mean per-category)',     f'{mean_ap*100:.2f}%'],
        ['Best F1-Score',               f'{best_f1:.4f}'],
        ['Precision @ best F1',         f'{best_p_val:.4f}'],
        ['Recall @ best F1',            f'{best_r_val:.4f}'],
        ['Best confidence threshold',   f'{best_thr:.2f}'],
        ['Mean FPS',                    f'{fps_arr.mean():.2f}'],
        ['Median FPS',                  f'{np.median(fps_arr):.2f}'],
        ['Std FPS',                     f'{fps_arr.std():.2f}'],
        ['Test images',                 str(n_test)],
        ['Total GT boxes',              str(sum(v["n_gt"] for v in cat_results.values()))],
        ['Total detections (≥ thr)',    str(len(all_detections))],
        ['Device',                      device_name],
        ['Multi-scale',                 str(USE_MULTI_SCALE)],
    ]

    col_w   = [0.60, 0.40]
    row_h   = 1.0 / len(rows)
    x_start = 0.02

    for r_idx, row in enumerate(rows):
        y  = 1.0 - (r_idx + 0.5) * row_h
        bg = '#e8f0fe' if r_idx == 0 else ('#ffffff' if r_idx % 2 else '#f1f3f4')
        rect = FancyBboxPatch(
            (x_start, y - row_h * 0.5),
            col_w[0] + col_w[1], row_h,
            boxstyle='round,pad=0.002',
            facecolor=bg, edgecolor='#dadce0', linewidth=0.6,
            transform=ax_tab.transAxes, clip_on=False)
        ax_tab.add_patch(rect)
        fw0 = 'bold' if r_idx == 0 else 'normal'
        fw1 = 'bold' if r_idx in (0, 1, 2) else 'normal'
        c1  = BRAND_COLOR if r_idx in (1, 2) else '#202124'
        ax_tab.text(x_start + 0.01, y, row[0],
                    transform=ax_tab.transAxes, va='center', ha='left',
                    fontsize=FONT_TICK, fontweight=fw0, color='#202124')
        ax_tab.text(x_start + col_w[0] + 0.01, y, row[1],
                    transform=ax_tab.transAxes, va='center', ha='left',
                    fontsize=FONT_TICK, fontweight=fw1, color=c1)

    ax_tab.set_title('EXDARK Summary — DAI-Net',
                     fontsize=FONT_TITLE, fontweight='bold', pad=10)

    fig.suptitle(
        'DAI-Net Evaluation on EXDARK  |  Zero-Shot Low-Light Object Detection  '
        f'(12 classes, {n_test} test images)',
        fontsize=15, fontweight='bold', color='#202124', y=0.96,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Text report
# ═══════════════════════════════════════════════════════════════════════════

def save_text_report(ap, mean_ap, best_f1, best_p, best_r, best_thr,
                     fps_list, all_detections, cat_results, n_test, out_path):
    fps_arr = np.array(fps_list)
    lines   = [
        '=' * 65,
        'DAI-Net Evaluation — EXDARK Dataset (Zero-Shot Low-Light)',
        '=' * 65,
        f'  Overall AP (pooled, IoU=0.50)  : {ap*100:.2f}%',
        f'  mAP (mean of 12 categories)    : {mean_ap*100:.2f}%',
        f'  Best F1-Score                  : {best_f1:.4f}',
        f'  Precision  @ best F1           : {best_p:.4f}',
        f'  Recall     @ best F1           : {best_r:.4f}',
        f'  Confidence @ best F1           : {best_thr:.2f}',
        '',
        f'  Mean FPS                       : {fps_arr.mean():.2f}',
        f'  Median FPS                     : {np.median(fps_arr):.2f}',
        f'  Std  FPS                       : {fps_arr.std():.2f}',
        f'  Min  FPS                       : {fps_arr.min():.2f}',
        f'  Max  FPS                       : {fps_arr.max():.2f}',
        '',
        f'  Test images                    : {n_test}',
        f'  Total GT boxes                 : {sum(v["n_gt"] for v in cat_results.values())}',
        f'  Total detections               : {len(all_detections)}',
        '',
        f'  Device                         : {device_name}',
        f'  Multi-scale                    : {USE_MULTI_SCALE}',
        f'  IoU threshold                  : {IOU_THRESH}',
        f'  Confidence threshold           : {CONF_THRESH}',
        '',
        '-' * 65,
        'Per-Category Results',
        '-' * 65,
        f'  {"Category":<14} {"AP":>7}  {"F1":>6}  {"Prec":>6}  {"Rec":>6}  {"GT":>5}',
    ]
    for cat in CLASSES:
        v = cat_results[cat]
        lines.append(
            f'  {cat:<14} {v["ap"]*100:>6.1f}%  {v["f1"]:>6.3f}  '
            f'{v["precision"]:>6.3f}  {v["recall"]:>6.3f}  {v["n_gt"]:>5}')
    lines.append('=' * 65)

    report = '\n'.join(lines)
    print('\n' + report)
    with open(out_path, 'w') as f:
        f.write(report + '\n')
    print(f'[INFO] Text report saved → {out_path}')


# ═══════════════════════════════════════════════════════════════════════════
# Run directory helper
# ═══════════════════════════════════════════════════════════════════════════

def make_run_dir(results_root, tag):
    os.makedirs(results_root, exist_ok=True)
    existing = [d for d in os.listdir(results_root)
                if os.path.isdir(os.path.join(results_root, d))
                and d.startswith(tag + '_')]
    numbers = []
    for name in existing:
        parts = name.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            numbers.append(int(parts[1]))
    nxt       = (max(numbers) + 1) if numbers else 1
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir   = os.path.join(results_root, f'{tag}_{nxt:03d}_{timestamp}')
    figures_dir = os.path.join(run_dir, 'figures')
    reports_dir = os.path.join(run_dir, 'reports')
    os.makedirs(figures_dir)
    os.makedirs(reports_dir)
    print(f'[INFO] Run directory  → {run_dir}')
    return run_dir, figures_dir, reports_dir


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    _WEIGHT_DEFAULTS = {
        'dark':      './weights/dsfd.pth',
        'yolo_dark': './weights/yolo_dark/dsfd.pth',
    }

    parser = argparse.ArgumentParser(
        description='DAI-Net evaluation on EXDARK dataset')
    parser.add_argument('--model',   default='yolo_dark',
                        choices=['dark', 'yolo_dark'],
                        help='Model architecture (default: yolo_dark)')
    parser.add_argument('--weights', default=None,
                        help='Path to .pth checkpoint (auto-detected if omitted)')
    parser.add_argument('--tag',     default=None,
                        help='Experiment tag for result subfolder')
    parser.add_argument('--no_multi_scale', action='store_true',
                        help='Disable multi-scale inference (faster, lower mAP)')
    cli = parser.parse_args()

    WEIGHTS_PATH   = cli.weights or _WEIGHT_DEFAULTS[cli.model]
    EXPERIMENT_TAG = cli.tag    or f'exdark_{cli.model}'

    if cli.no_multi_scale:
        USE_MULTI_SCALE = False

    # 0. Validate paths
    for p, label in [(IMAGES_DIR, 'ExDark images dir'),
                     (ANNOS_DIR,  'ExDark annotations dir'),
                     (CLASSLIST_PATH, 'imageclasslist.txt'),
                     (WEIGHTS_PATH,   'weights file')]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'[ERROR] {label} not found: {p}')

    # 1. Load test split
    print('[INFO] Loading EXDARK test split …')
    records = load_test_split()
    print(f'[INFO] Test images found : {len(records)}')
    if len(records) == 0:
        raise RuntimeError('[ERROR] No test images found. Check imageclasslist.txt split column.')

    # 2. Create run directory
    run_dir, figures_dir, reports_dir = make_run_dir(RESULTS_ROOT, EXPERIMENT_TAG)
    OUT_FIGURE = os.path.join(figures_dir, 'exdark_metrics.png')
    OUT_TXT    = os.path.join(reports_dir, 'exdark_summary.txt')

    # 3. Load model
    net = load_model(cli.model, WEIGHTS_PATH)

    # 4. Run inference
    all_dets, all_gts, total_gt, fps_list = run_inference(net, records)

    # 5. Overall evaluation (class-agnostic pooled AP)
    print('[INFO] Computing overall PR curve …')
    rec, prec, ap = evaluate_overall(all_dets, all_gts, total_gt)

    # 6. Per-category AP
    print('[INFO] Computing per-category AP …')
    cat_results, mean_ap = evaluate_all_categories(all_dets, all_gts)
    print(f'[INFO] mAP (12 categories) : {mean_ap*100:.2f}%')

    # 7. F1 / P / R curve
    print('[INFO] Computing F1 curve …')
    thresholds, f1_arr, p_arr, r_arr = compute_f1_curve(all_dets, all_gts, total_gt)

    best_idx = int(np.argmax(f1_arr))
    best_thr = float(thresholds[best_idx])
    best_f1  = float(f1_arr[best_idx])
    best_p   = float(p_arr[best_idx])
    best_r   = float(r_arr[best_idx])

    # 8. Save text report
    save_text_report(ap, mean_ap, best_f1, best_p, best_r, best_thr,
                     fps_list, all_dets, cat_results, len(records), OUT_TXT)

    # 9. Build & save figure
    print('[INFO] Generating figure …')
    fig = build_figure(rec, prec, ap, mean_ap,
                       thresholds, f1_arr, p_arr, r_arr,
                       fps_list, all_dets, cat_results, len(records))
    fig.savefig(OUT_FIGURE, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[INFO] Figure saved   → {OUT_FIGURE}')
    print('[DONE]')
