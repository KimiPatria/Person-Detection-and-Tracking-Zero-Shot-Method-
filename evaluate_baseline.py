# -*- coding: utf-8 -*-
"""
Baseline Evaluation Script — DAI-Net (Zero-Shot Low-Light Human Detection)
===========================================================================
Runs inference on the test split, computes comprehensive detection metrics,
and exports a publication-ready figure for thesis documentation.

Metrics produced
----------------
  • Precision-Recall curve  (Pascal VOC 11-point interpolation, IoU ≥ 0.5)
  • mAP @ IoU = 0.50
  • F1-score curve vs. confidence threshold
  • Per-image FPS distribution (histogram + KDE)
  • Confidence score distribution of all detections
  • Summary table: mAP, best-F1, precision / recall at best-F1, mean FPS

Usage
-----
    python evaluate_baseline.py

Output
------
    result/baseline_metrics.png   — full dashboard figure
    result/baseline_results.txt   — plain-text summary
"""

from __future__ import division, absolute_import, print_function

import os
import glob
import time
import datetime
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')   # headless backend — no display required
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
# Each run gets its own subfolder:  result/<EXPERIMENT_TAG>_<NNN>/
#   e.g. result/baseline_001/
# Set EXPERIMENT_TAG to a short label that describes the run.
EXPERIMENT_TAG = 'baseline'

# ─── Inference settings ──────────────────────────────────────────────────────
USE_MULTI_SCALE = True   # single-scale for baseline speed measurement
MY_SHRINK       = 1.0     # scale factor (1.0 = 640×640, 1.5 = 960×960, etc.)
CONF_THRESH     = 0.01    # minimum confidence to keep a detection
IOU_THRESH      = 0.50    # IoU threshold for TP/FP assignment


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
    # test_forward returns normalised coords relative to padded image
    px_scale = np.array([target_size, target_size, target_size, target_size])

    boxes, scores = [], []
    for i in range(detections.shape[1]):
        j = 0
        while j < detections.shape[2] and detections[0, i, j, 0] > 0.0:
            pt    = detections[0, i, j, 1:] * px_scale  # to padded-pixel coords
            score = detections[0, i, j, 0]
            # undo letterbox: remove padding, then undo scale
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
    """Horizontal flip TTA — detect on flipped image, mirror boxes back."""
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
    """Adaptive multi-scale: small shrink for big objects, big shrink for small."""
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


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    bboxes = []
    for obj in tree.getroot().findall('object'):
        b = obj.find('bndbox')
        bboxes.append({
            'bbox': [float(b.find('xmin').text),
                     float(b.find('ymin').text),
                     float(b.find('xmax').text),
                     float(b.find('ymax').text)],
            'matched': False
        })
    return bboxes


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
    all_gts         = {}   # {img_id: [{'bbox': [...], 'matched': False}]}
    total_gt        = 0
    fps_list        = []

    for idx, img_path in enumerate(img_paths, 1):
        img_id   = Path(img_path).stem
        xml_path = os.path.join(ANNOTATIONS_DIR, img_id + '.xml')

        # ground truth
        if os.path.exists(xml_path):
            gts = parse_voc_xml(xml_path)
            all_gts[img_id] = gts
            total_gt += len(gts)
        else:
            all_gts[img_id] = []

        # load image
        img = np.array(Image.open(img_path).convert('RGB'))

        # inference
        t0 = time.time()
        with torch.no_grad():
            if USE_MULTI_SCALE:
                # Cap at 2.0 — largest pass will be 1280×1280, safe for most GPUs
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

    print(f'\n[INFO] Total GT boxes   : {total_gt}')
    print(f'[INFO] Total detections : {len(all_detections)}')
    return all_detections, all_gts, total_gt, fps_list


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(all_detections, all_gts, total_gt):
    """Return precision, recall, AP arrays and raw TP/FP vectors."""
    # sort by descending confidence
    all_detections.sort(key=lambda x: x[1], reverse=True)
    nd = len(all_detections)

    tp_raw = np.zeros(nd)
    fp_raw = np.zeros(nd)
    conf   = np.array([d[1] for d in all_detections])

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

    rec  = cum_tp / float(total_gt) if total_gt > 0 else np.zeros(nd)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)
    ap   = voc_ap(rec, prec)

    return rec, prec, ap, conf, tp_raw, fp_raw


def compute_f1_curve(rec, prec, conf, all_detections, all_gts, total_gt):
    """Compute F1, precision, recall at a range of confidence thresholds."""
    thresholds  = np.linspace(0.01, 0.95, 95)
    f1_list, p_list, r_list = [], [], []

    for thr in thresholds:
        # reset matched flags
        for gts in all_gts.values():
            for gt in gts:
                gt['matched'] = False

        tp = fp = 0
        for det in all_detections:
            if det[1] < thr:
                continue
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
                    tp += 1
                    gts[best_k]['matched'] = True
                else:
                    fp += 1
            else:
                fp += 1

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / total_gt  if total_gt  > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_list.append(f1)
        p_list.append(p)
        r_list.append(r)

    # reset matched flags for subsequent callers
    for gts in all_gts.values():
        for gt in gts:
            gt['matched'] = False

    return thresholds, np.array(f1_list), np.array(p_list), np.array(r_list)


# ═══════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════

BRAND_COLOR = '#1a73e8'   # accent blue
WARN_COLOR  = '#ea4335'   # accent red
FONT_TITLE  = 13
FONT_LABEL  = 11
FONT_TICK   = 9


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.4)


def build_figure(rec, prec, ap, conf,
                 thresholds, f1_arr, p_arr, r_arr,
                 fps_list, all_detections):

    # ── layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), facecolor='#f8f9fa')
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            left=0.06, right=0.97,
                            top=0.88, bottom=0.10,
                            hspace=0.45, wspace=0.35)

    ax_pr   = fig.add_subplot(gs[0, 0])   # Precision-Recall curve
    ax_f1   = fig.add_subplot(gs[0, 1])   # F1 / P / R vs threshold
    ax_fps  = fig.add_subplot(gs[0, 2])   # FPS distribution
    ax_conf = fig.add_subplot(gs[1, 0])   # Confidence score histogram
    ax_tab  = fig.add_subplot(gs[1, 1:])  # Summary table (spans 2 cols)
    ax_tab.axis('off')

    # ── 1. Precision-Recall curve ────────────────────────────────────────
    ax_pr.plot(rec, prec, color=BRAND_COLOR, lw=2, label=f'AP = {ap*100:.1f}%')
    ax_pr.fill_between(rec, prec, alpha=0.12, color=BRAND_COLOR)
    # 11-point interpolated markers
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) > 0:
            p_t = np.max(prec[rec >= t])
            ax_pr.plot(t, p_t, 'o', color=BRAND_COLOR,
                       markersize=4, alpha=0.7)
    ax_pr.set_xlim([0, 1]);  ax_pr.set_ylim([0, 1.05])
    ax_pr.legend(fontsize=FONT_LABEL, framealpha=0.7)
    _style_ax(ax_pr, 'Precision-Recall Curve (IoU ≥ 0.50)',
              'Recall', 'Precision')

    # ── 2. F1 / Precision / Recall vs confidence ─────────────────────────
    best_idx  = int(np.argmax(f1_arr))
    best_thr  = thresholds[best_idx]
    best_f1   = f1_arr[best_idx]

    ax_f1.plot(thresholds, f1_arr, lw=2,   color='#34a853', label='F1')
    ax_f1.plot(thresholds, p_arr,  lw=1.5, color=BRAND_COLOR,
               linestyle='--', label='Precision')
    ax_f1.plot(thresholds, r_arr,  lw=1.5, color=WARN_COLOR,
               linestyle=':', label='Recall')
    ax_f1.axvline(best_thr, color='#34a853', lw=1.2,
                  linestyle='-.', alpha=0.8,
                  label=f'Best conf = {best_thr:.2f}')
    ax_f1.scatter([best_thr], [best_f1],
                  color='#34a853', zorder=5, s=60)
    ax_f1.annotate(f'F1={best_f1:.2f}',
                   xy=(best_thr, best_f1),
                   xytext=(best_thr + 0.06, best_f1 - 0.07),
                   fontsize=9, color='#34a853',
                   arrowprops=dict(arrowstyle='->', color='#34a853', lw=1))
    ax_f1.set_xlim([0.05, 0.95]); ax_f1.set_ylim([0, 1.05])
    ax_f1.legend(fontsize=9, framealpha=0.7)
    _style_ax(ax_f1, 'F1 / Precision / Recall vs Confidence',
              'Confidence Threshold', 'Score')

    # ── 3. FPS distribution ──────────────────────────────────────────────
    fps_arr  = np.array(fps_list)
    mean_fps = fps_arr.mean()
    std_fps  = fps_arr.std()

    bins = min(30, max(10, len(fps_arr) // 5))
    ax_fps.hist(fps_arr, bins=bins, color=BRAND_COLOR,
                alpha=0.75, edgecolor='white', linewidth=0.5)
    ax_fps.axvline(mean_fps, color=WARN_COLOR, lw=2,
                   linestyle='--', label=f'Mean = {mean_fps:.2f} FPS')
    ax_fps.legend(fontsize=FONT_LABEL, framealpha=0.7)
    _style_ax(ax_fps, f'Per-Image FPS Distribution ({device_name})',
              'FPS', 'Count')

    # ── 4. Confidence score histogram ────────────────────────────────────
    scores = np.array([d[1] for d in all_detections])
    ax_conf.hist(scores, bins=40, color='#fbbc05',
                 alpha=0.85, edgecolor='white', linewidth=0.5)
    ax_conf.axvline(CONF_THRESH, color=WARN_COLOR, lw=1.8,
                    linestyle='--',
                    label=f'Threshold = {CONF_THRESH}')
    ax_conf.legend(fontsize=FONT_LABEL, framealpha=0.7)
    _style_ax(ax_conf, 'Detection Confidence Distribution',
              'Confidence Score', 'Count')

    # ── 5. Summary table ─────────────────────────────────────────────────
    best_p = float(p_arr[best_idx])
    best_r = float(r_arr[best_idx])

    rows = [
        ['Metric',                     'Value'],
        ['mAP @ IoU = 0.50',           f'{ap*100:.2f}%'],
        ['Best F1-Score',              f'{best_f1:.4f}'],
        ['Precision @ best F1',        f'{best_p:.4f}'],
        ['Recall @ best F1',           f'{best_r:.4f}'],
        ['Best confidence threshold',  f'{best_thr:.2f}'],
        ['Mean FPS',                   f'{mean_fps:.2f}'],
        ['Std FPS',                    f'{std_fps:.2f}'],
        ['Total test images',          str(len(fps_list))],
        ['Total GT objects',           '(see console)'],
        ['Total detections (≥ thr)',   str(len(all_detections))],
        ['Device',                     device_name],
        ['IoU threshold',              f'{IOU_THRESH:.2f}'],
        ['Multi-scale',                str(USE_MULTI_SCALE)],
    ]

    col_w = [0.52, 0.48]
    row_h = 1.0 / len(rows)
    x_start = 0.03

    for r_idx, row in enumerate(rows):
        y = 1.0 - (r_idx + 0.5) * row_h
        bg = '#e8f0fe' if r_idx == 0 else ('#ffffff' if r_idx % 2 else '#f1f3f4')
        rect = FancyBboxPatch((x_start, y - row_h * 0.5),
                              col_w[0] + col_w[1], row_h,
                              boxstyle='round,pad=0.002',
                              facecolor=bg, edgecolor='#dadce0',
                              linewidth=0.6,
                              transform=ax_tab.transAxes, clip_on=False)
        ax_tab.add_patch(rect)

        fw0 = 'bold' if r_idx == 0 else 'normal'
        fw1 = 'bold' if r_idx in (0, 1) else 'normal'
        c1  = BRAND_COLOR if r_idx == 1 else '#202124'

        ax_tab.text(x_start + 0.01, y, row[0],
                    transform=ax_tab.transAxes,
                    va='center', ha='left',
                    fontsize=FONT_LABEL, fontweight=fw0, color='#202124')
        ax_tab.text(x_start + col_w[0] + 0.01, y, row[1],
                    transform=ax_tab.transAxes,
                    va='center', ha='left',
                    fontsize=FONT_LABEL, fontweight=fw1, color=c1)

    ax_tab.set_title('Baseline Summary — DAI-Net Zero-Shot',
                     fontsize=FONT_TITLE, fontweight='bold', pad=10)

    # ── Main title ───────────────────────────────────────────────────────
    fig.suptitle(
        'DAI-Net Baseline Evaluation  |  Low-Light Human Detection  '
        '(Zero-Shot)',
        fontsize=16, fontweight='bold', color='#202124', y=0.96
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Text report
# ═══════════════════════════════════════════════════════════════════════════

def save_text_report(ap, best_f1, best_p, best_r, best_thr,
                     fps_list, all_detections, total_gt, out_path):
    fps_arr = np.array(fps_list)
    lines = [
        '=' * 60,
        'DAI-Net Baseline Evaluation — Low-Light Human Detection',
        '=' * 60,
        f'  mAP @ IoU = 0.50        : {ap*100:.2f}%',
        f'  Best F1-Score            : {best_f1:.4f}',
        f'  Precision  @ best F1     : {best_p:.4f}',
        f'  Recall     @ best F1     : {best_r:.4f}',
        f'  Confidence @ best F1     : {best_thr:.2f}',
        '',
        f'  Mean FPS                 : {fps_arr.mean():.2f}',
        f'  Median FPS               : {np.median(fps_arr):.2f}',
        f'  Std  FPS                 : {fps_arr.std():.2f}',
        f'  Min  FPS                 : {fps_arr.min():.2f}',
        f'  Max  FPS                 : {fps_arr.max():.2f}',
        '',
        f'  Total test images        : {len(fps_list)}',
        f'  Total GT objects         : {total_gt}',
        f'  Total detections (raw)   : {len(all_detections)}',
        '',
        f'  Device                   : {device_name}',
        f'  Multi-scale              : {USE_MULTI_SCALE}',
        f'  IoU threshold            : {IOU_THRESH}',
        f'  Confidence threshold     : {CONF_THRESH}',
        '=' * 60,
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
    """
    Create and return a unique run directory:
        <results_root>/<tag>_<NNN>_<YYYYMMDD_HHMMSS>/
            figures/
            reports/

    The three-digit counter NNN auto-increments so existing runs are
    never overwritten even if two runs start within the same second.
    """
    os.makedirs(results_root, exist_ok=True)

    # find the next free counter
    existing = [
        d for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d))
        and d.startswith(tag + '_')
    ]
    numbers = []
    for name in existing:
        parts = name.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            numbers.append(int(parts[1]))
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
    # ── CLI arguments ─────────────────────────────────────────────────────
    _WEIGHT_DEFAULTS = {
        'dark':      './weights/dsfd.pth',
        'yolo_dark': './weights/yolo_dark/dsfd.pth',
    }
    parser = argparse.ArgumentParser(
        description='DAI-Net evaluation — pass --model to switch architectures')
    parser.add_argument('--model',   default='yolo_dark',
                        choices=['dark', 'yolo_dark'],
                        help='Model architecture to evaluate (default: yolo_dark)')
    parser.add_argument('--weights', default=None,
                        help='Path to .pth weights file (auto-detected if omitted)')
    parser.add_argument('--tag',     default=None,
                        help='Experiment tag for result subfolder (default: model name)')
    cli = parser.parse_args()

    # Override module-level globals based on CLI
    WEIGHTS_PATH    = cli.weights or _WEIGHT_DEFAULTS[cli.model]
    EXPERIMENT_TAG  = cli.tag    or cli.model

    # 0. Create run subfolder
    run_dir, figures_dir, reports_dir = make_run_dir(RESULTS_ROOT, EXPERIMENT_TAG)
    OUT_FIGURE = os.path.join(figures_dir, 'metrics.png')
    OUT_TXT    = os.path.join(reports_dir, 'summary.txt')

    # 1. Load model
    net = load_model(cli.model)

    # 2. Run inference
    all_dets, all_gts, total_gt, fps_list = run_inference(net)

    # 3. Evaluate — PR curve & AP
    rec, prec, ap, conf, tp_raw, fp_raw = evaluate(all_dets, all_gts, total_gt)

    # 4. F1 / P / R sweep over confidence thresholds
    print('[INFO] Computing F1 curve across confidence thresholds…')
    thresholds, f1_arr, p_arr, r_arr = compute_f1_curve(
        rec, prec, conf, all_dets, all_gts, total_gt)

    best_idx = int(np.argmax(f1_arr))
    best_thr = float(thresholds[best_idx])
    best_f1  = float(f1_arr[best_idx])
    best_p   = float(p_arr[best_idx])
    best_r   = float(r_arr[best_idx])

    # 5. Save text report
    save_text_report(ap, best_f1, best_p, best_r, best_thr,
                     fps_list, all_dets, total_gt, OUT_TXT)

    # 6. Build & save figure
    print('[INFO] Generating figure…')
    fig = build_figure(rec, prec, ap, conf,
                       thresholds, f1_arr, p_arr, r_arr,
                       fps_list, all_dets)
    fig.savefig(OUT_FIGURE, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[INFO] Figure saved  → {OUT_FIGURE}')
    print('[DONE]')
