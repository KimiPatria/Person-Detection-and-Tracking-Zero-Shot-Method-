# -*- coding: utf-8 -*-
"""
Ablation Study Evaluation — DAI-Net Component Contribution
============================================================
Evaluates all four ablation variants and produces a side-by-side
comparison figure + report for thesis documentation.

Variants
--------
  1. YOLOv8n Baseline          — detection only
  2. + Reflectance Decoder     — adds R branch + enhance losses
  3. + Mutual Alignment Loss   — adds KL feature alignment
  4. Full DAI-Net              — all components (+ full Retinex coherence)

Usage
-----
    python evaluate_ablation.py
    python evaluate_ablation.py --images ./dataset/roboflow/test/images/

Output
------
    result/ablation_<NNN>_<timestamp>/figures/ablation_comparison.png
    result/ablation_<NNN>_<timestamp>/reports/ablation_summary.txt
"""

from __future__ import division, absolute_import, print_function

import os
import glob
import time
import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import OrderedDict

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

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
RESULTS_ROOT    = './result/'

# ─── Inference settings ──────────────────────────────────────────────────────
USE_MULTI_SCALE = True
MY_SHRINK       = 1.0
CONF_THRESH     = 0.01
IOU_THRESH      = 0.50

# ─── Ablation variants ──────────────────────────────────────────────────────
VARIANTS = OrderedDict([
    ('baseline',    {
        'label':   'YOLOv8n Baseline',
        'short':   'Baseline',
        'weights': './weights/ablation_baseline/dsfd.pth',
        'color':   '#9e9e9e',
    }),
    ('reflectance', {
        'label':   '+ Reflectance Decoder',
        'short':   '+RefDec',
        'weights': './weights/ablation_reflectance/dsfd.pth',
        'color':   '#4285f4',
    }),
    ('ref_kl',      {
        'label':   '+ Mutual Alignment',
        'short':   '+RefDec+KL',
        'weights': './weights/ablation_ref_kl/dsfd.pth',
        'color':   '#fbbc05',
    }),
    ('full',        {
        'label':   'Full DAI-Net',
        'short':   'Full',
        'weights': './weights/yolo_dark/dsfd.pth',
        'color':   '#34a853',
    }),
])


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions (same as evaluate_baseline.py)
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
    px_scale = np.array([target_size] * 4)
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
    return np.column_stack((np.array(boxes), np.array(scores)))


def flip_test(net, image, shrink):
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
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(net, image, st)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s, detect_face(net, image, 0.75)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1,
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
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1,
                                    det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1,
                                    det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]
    return det_s, det_b


def multi_scale_test_pyramid(net, image, max_shrink):
    det_b = detect_face(net, image, 0.25)
    index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1,
                                det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
    det_b = det_b[index, :]
    for s in [1.25, 1.75, 2.25]:
        if s <= max_shrink:
            det_temp = detect_face(net, image, s)
            if s > 1:
                index = np.where(np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                                            det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
            else:
                index = np.where(np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                                            det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
            det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b


def bbox_vote(det_):
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
            'bbox': [float(b.find('xmin').text), float(b.find('ymin').text),
                     float(b.find('xmax').text), float(b.find('ymax').text)],
            'matched': False,
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
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        p = np.max(prec[rec >= t]) if np.sum(rec >= t) > 0 else 0.0
        ap += p / 11.0
    return ap


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(weights_path):
    """Load a YOLOv8n DAI-Net model from weights."""
    net = build_net('test', num_classes=1, model='yolo_dark')
    net.eval()
    ckpt = torch.load(weights_path,
                      map_location='cuda' if use_cuda else 'cpu')
    if isinstance(ckpt, dict) and 'weight' in ckpt:
        net.load_state_dict(ckpt['weight'])
    else:
        net.load_state_dict(ckpt)
    if use_cuda:
        net = net.cuda()
    return net


# ═══════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(net, img_paths):
    """Run inference on all test images, return detections and GT."""
    all_detections = []
    all_gts = {}
    total_gt = 0
    fps_list = []

    for idx, img_path in enumerate(img_paths, 1):
        img_id = Path(img_path).stem
        xml_path = os.path.join(ANNOTATIONS_DIR, img_id + '.xml')

        if os.path.exists(xml_path):
            gts = parse_voc_xml(xml_path)
            all_gts[img_id] = gts
            total_gt += len(gts)
        else:
            all_gts[img_id] = []

        img = np.array(Image.open(img_path).convert('RGB'))

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
        fps_list.append(1.0 / (time.time() - t0))

        for i in range(dets.shape[0]):
            score = float(dets[i, 4])
            if score > CONF_THRESH:
                all_detections.append(
                    [img_id, score,
                     float(dets[i, 0]), float(dets[i, 1]),
                     float(dets[i, 2]), float(dets[i, 3])])

        print(f'\r  Processed {idx}/{len(img_paths)} | FPS {fps_list[-1]:.1f}', end='')

    print()
    return all_detections, all_gts, total_gt, fps_list


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(all_detections, all_gts, total_gt):
    """Compute PR curve, AP, and best-F1 metrics."""
    all_detections.sort(key=lambda x: x[1], reverse=True)
    nd = len(all_detections)

    tp_raw = np.zeros(nd)
    fp_raw = np.zeros(nd)

    for d_idx, det in enumerate(all_detections):
        img_id = det[0]
        bb = det[2:]
        gts = all_gts.get(img_id, [])

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
    rec = cum_tp / float(total_gt) if total_gt > 0 else np.zeros(nd)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    # Best F1 sweep
    best_f1, best_p, best_r, best_thr = 0.0, 0.0, 0.0, 0.0
    for thr in np.linspace(0.01, 0.95, 95):
        for gts_list in all_gts.values():
            for gt in gts_list:
                gt['matched'] = False
        tp = fp = 0
        for det in all_detections:
            if det[1] < thr:
                continue
            img_id = det[0]
            bb = det[2:]
            gts = all_gts.get(img_id, [])
            best_iou_t, best_k_t = -np.inf, -1
            for k, gt in enumerate(gts):
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
        r = tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_p, best_r, best_thr = f1, p, r, thr

    # Reset matched flags
    for gts_list in all_gts.values():
        for gt in gts_list:
            gt['matched'] = False

    return {
        'rec': rec, 'prec': prec, 'ap': ap,
        'best_f1': best_f1, 'best_p': best_p, 'best_r': best_r,
        'best_thr': best_thr,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════

FONT_TITLE = 14
FONT_LABEL = 11
FONT_TICK  = 9


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.35)


def build_figure(results, fps_data):
    """Build ablation comparison dashboard."""
    fig = plt.figure(figsize=(20, 12), facecolor='#f8f9fa')
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.06, right=0.97,
                           top=0.88, bottom=0.08,
                           hspace=0.38, wspace=0.30)

    ax_pr   = fig.add_subplot(gs[0, 0])    # PR curves overlay
    ax_ap   = fig.add_subplot(gs[0, 1])    # AP bar chart
    ax_f1   = fig.add_subplot(gs[0, 2])    # F1 bar chart
    ax_pr2  = fig.add_subplot(gs[1, 0])    # Precision & Recall bars
    ax_fps  = fig.add_subplot(gs[1, 1])    # FPS comparison
    ax_tab  = fig.add_subplot(gs[1, 2])    # Summary table
    ax_tab.axis('off')

    variant_keys = list(results.keys())
    labels = [VARIANTS[k]['label'] for k in variant_keys]
    colors = [VARIANTS[k]['color'] for k in variant_keys]
    shorts = [VARIANTS[k]['short'] for k in variant_keys]

    # ── 1. PR curves (overlaid) ──────────────────────────────────────────
    for k, label, color in zip(variant_keys, labels, colors):
        r = results[k]
        ax_pr.plot(r['rec'], r['prec'], color=color, lw=2,
                   label=f'{label} (AP={r["ap"]*100:.1f}%)')
    ax_pr.set_xlim([0, 1]); ax_pr.set_ylim([0, 1.05])
    ax_pr.legend(fontsize=8, framealpha=0.7, loc='lower left')
    _style_ax(ax_pr, 'Precision-Recall Curves (IoU >= 0.50)',
              'Recall', 'Precision')

    # ── 2. AP bar chart ──────────────────────────────────────────────────
    aps = [results[k]['ap'] * 100 for k in variant_keys]
    bars = ax_ap.bar(range(len(aps)), aps, color=colors,
                     edgecolor='white', linewidth=0.5, width=0.6)
    ax_ap.set_xticks(range(len(aps)))
    ax_ap.set_xticklabels(shorts, fontsize=FONT_TICK)
    ax_ap.set_ylim([0, max(aps) * 1.25 if aps else 100])
    for i, v in enumerate(aps):
        ax_ap.text(i, v + 0.5, f'{v:.1f}%', ha='center',
                   fontsize=FONT_LABEL, fontweight='bold', color=colors[i])
    _style_ax(ax_ap, 'mAP @ IoU=0.50', 'Variant', 'AP (%)')

    # ── 3. F1 bar chart ─────────────────────────────────────────────────
    f1s = [results[k]['best_f1'] * 100 for k in variant_keys]
    ax_f1.bar(range(len(f1s)), f1s, color=colors,
              edgecolor='white', linewidth=0.5, width=0.6)
    ax_f1.set_xticks(range(len(f1s)))
    ax_f1.set_xticklabels(shorts, fontsize=FONT_TICK)
    ax_f1.set_ylim([0, max(f1s) * 1.25 if f1s else 100])
    for i, v in enumerate(f1s):
        ax_f1.text(i, v + 0.5, f'{v:.1f}%', ha='center',
                   fontsize=FONT_LABEL, fontweight='bold', color=colors[i])
    _style_ax(ax_f1, 'Best F1-Score', 'Variant', 'F1 (%)')

    # ── 4. Precision & Recall grouped bars ───────────────────────────────
    precs = [results[k]['best_p'] * 100 for k in variant_keys]
    recs  = [results[k]['best_r'] * 100 for k in variant_keys]
    x = np.arange(len(variant_keys))
    w = 0.3
    ax_pr2.bar(x - w/2, precs, w, label='Precision', color='#1a73e8', alpha=0.85)
    ax_pr2.bar(x + w/2, recs,  w, label='Recall',    color='#ea4335', alpha=0.85)
    ax_pr2.set_xticks(x)
    ax_pr2.set_xticklabels(shorts, fontsize=FONT_TICK)
    ax_pr2.set_ylim([0, max(max(precs), max(recs)) * 1.25 if precs else 100])
    ax_pr2.legend(fontsize=FONT_TICK)
    _style_ax(ax_pr2, 'Precision & Recall @ Best F1', 'Variant', 'Score (%)')

    # ── 5. FPS comparison ────────────────────────────────────────────────
    mean_fps = [np.mean(fps_data[k]) for k in variant_keys]
    ax_fps.bar(range(len(mean_fps)), mean_fps, color=colors,
               edgecolor='white', linewidth=0.5, width=0.6)
    ax_fps.set_xticks(range(len(mean_fps)))
    ax_fps.set_xticklabels(shorts, fontsize=FONT_TICK)
    for i, v in enumerate(mean_fps):
        ax_fps.text(i, v + 0.02, f'{v:.2f}', ha='center',
                    fontsize=FONT_LABEL, fontweight='bold', color=colors[i])
    _style_ax(ax_fps, f'Mean FPS ({device_name})', 'Variant', 'FPS')

    # ── 6. Summary table ─────────────────────────────────────────────────
    headers = ['Variant', 'AP(%)', 'F1(%)', 'Prec(%)', 'Rec(%)', 'FPS']
    rows = [headers]
    for k in variant_keys:
        r = results[k]
        rows.append([
            VARIANTS[k]['short'],
            f'{r["ap"]*100:.1f}',
            f'{r["best_f1"]*100:.1f}',
            f'{r["best_p"]*100:.1f}',
            f'{r["best_r"]*100:.1f}',
            f'{np.mean(fps_data[k]):.2f}',
        ])

    # Delta row (full vs baseline improvement)
    if 'baseline' in results and 'full' in results:
        rb = results['baseline']
        rf = results['full']
        rows.append([
            'Delta',
            f'+{(rf["ap"] - rb["ap"])*100:.1f}',
            f'+{(rf["best_f1"] - rb["best_f1"])*100:.1f}',
            f'+{(rf["best_p"] - rb["best_p"])*100:.1f}',
            f'+{(rf["best_r"] - rb["best_r"])*100:.1f}',
            '',
        ])

    row_h = 1.0 / len(rows)
    col_w = [0.22, 0.14, 0.14, 0.14, 0.14, 0.14]
    x_start = 0.02

    for r_idx, row in enumerate(rows):
        y = 1.0 - (r_idx + 0.5) * row_h
        if r_idx == 0:
            bg = '#e8f0fe'
        elif r_idx == len(rows) - 1 and len(rows) > len(variant_keys) + 1:
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
            if r_idx > 0 and r_idx < len(rows) - 1 and c_idx == 1:
                color = VARIANTS[variant_keys[r_idx - 1]]['color']
            ax_tab.text(cx + 0.01, y, cell,
                        transform=ax_tab.transAxes,
                        va='center', ha='left',
                        fontsize=FONT_TICK, fontweight=fw, color=color)
            cx += col_w[c_idx]

    ax_tab.set_title('Ablation Summary',
                     fontsize=FONT_TITLE, fontweight='bold', pad=10)

    fig.suptitle(
        'DAI-Net Ablation Study  |  Component Contribution Analysis',
        fontsize=16, fontweight='bold', color='#202124', y=0.96
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Text report
# ═══════════════════════════════════════════════════════════════════════════

def save_text_report(results, fps_data, total_gt, n_images, out_path):
    lines = [
        '=' * 72,
        'DAI-Net Ablation Study — Component Contribution Analysis',
        '=' * 72,
        '',
        f'  {"Variant":<30s} {"AP(%)":>8s} {"F1(%)":>8s} '
        f'{"Prec(%)":>8s} {"Rec(%)":>8s} {"FPS":>8s}',
        '-' * 72,
    ]

    for k in results:
        r = results[k]
        lines.append(
            f'  {VARIANTS[k]["label"]:<30s} {r["ap"]*100:>8.2f} '
            f'{r["best_f1"]*100:>8.2f} {r["best_p"]*100:>8.2f} '
            f'{r["best_r"]*100:>8.2f} {np.mean(fps_data[k]):>8.2f}'
        )

    lines.append('-' * 72)

    # Incremental deltas
    variant_keys = list(results.keys())
    if len(variant_keys) > 1:
        lines.append('')
        lines.append('  Incremental Improvement (delta AP):')
        prev_ap = results[variant_keys[0]]['ap']
        for k in variant_keys[1:]:
            delta = results[k]['ap'] - prev_ap
            lines.append(
                f'    {VARIANTS[variant_keys[0]]["short"]} -> '
                f'{VARIANTS[k]["short"]}: '
                f'{"+":s}{delta*100:.2f}% AP'
            )
            prev_ap = results[k]['ap']

    # Pairwise deltas from baseline
    if 'baseline' in results:
        lines.append('')
        lines.append('  Improvement over Baseline:')
        rb = results['baseline']
        for k in variant_keys[1:]:
            r = results[k]
            lines.append(
                f'    {VARIANTS[k]["label"]:<30s}: '
                f'+{(r["ap"] - rb["ap"])*100:.2f}% AP, '
                f'+{(r["best_f1"] - rb["best_f1"])*100:.2f}% F1'
            )

    lines += [
        '',
        f'  Total test images        : {n_images}',
        f'  Total GT objects         : {total_gt}',
        f'  Device                   : {device_name}',
        f'  Multi-scale              : {USE_MULTI_SCALE}',
        f'  IoU threshold            : {IOU_THRESH}',
        '=' * 72,
    ]

    report = '\n'.join(lines)
    print('\n' + report)
    with open(out_path, 'w') as f:
        f.write(report + '\n')
    print(f'[INFO] Report saved -> {out_path}')


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
        if len(parts) >= 2 and parts[1].isdigit():
            numbers.append(int(parts[1]))
    nxt = (max(numbers) + 1) if numbers else 1

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'{tag}_{nxt:03d}_{timestamp}'
    run_dir = os.path.join(results_root, run_name)

    figures_dir = os.path.join(run_dir, 'figures')
    reports_dir = os.path.join(run_dir, 'reports')
    os.makedirs(figures_dir)
    os.makedirs(reports_dir)
    print(f'[INFO] Run directory -> {run_dir}')
    return run_dir, figures_dir, reports_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DAI-Net ablation study evaluation')
    parser.add_argument('--images', default=IMAGES_DIR,
                        help='Test images directory')
    parser.add_argument('--annotations', default=ANNOTATIONS_DIR,
                        help='Test annotations directory')
    parser.add_argument('--variants', nargs='+',
                        default=['baseline', 'reflectance', 'ref_kl', 'full'],
                        choices=['baseline', 'reflectance', 'ref_kl', 'full'],
                        help='Variants to evaluate (default: all four)')
    cli = parser.parse_args()

    IMAGES_DIR = cli.images
    ANNOTATIONS_DIR = cli.annotations

    # Discover test images
    img_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, '*.jpg')))
    print(f'[INFO] Found {len(img_paths)} test images.')

    # Create output directory
    run_dir, figures_dir, reports_dir = make_run_dir(RESULTS_ROOT, 'ablation')

    # Check which variants have weights available
    available = []
    for key in cli.variants:
        wp = VARIANTS[key]['weights']
        if os.path.exists(wp):
            available.append(key)
            print(f'[OK]   {VARIANTS[key]["label"]:<30s} -> {wp}')
        else:
            print(f'[SKIP] {VARIANTS[key]["label"]:<30s} -> {wp} (not found)')

    if not available:
        print('[ERROR] No variant weights found. Train the ablation variants first:')
        print('        python train_yolo.py --ablation baseline')
        print('        python train_yolo.py --ablation reflectance')
        print('        python train_yolo.py --ablation ref_kl')
        print('        python train_yolo.py --ablation full')
        exit(1)

    # Evaluate each available variant
    results = OrderedDict()
    fps_data = OrderedDict()
    total_gt = 0

    for key in available:
        info = VARIANTS[key]
        print(f'\n{"="*60}')
        print(f'Evaluating: {info["label"]}')
        print(f'{"="*60}')

        net = load_model(info['weights'])
        all_dets, all_gts, total_gt, fps_list = run_inference(net, img_paths)
        metrics = evaluate(all_dets, all_gts, total_gt)
        results[key] = metrics
        fps_data[key] = fps_list

        print(f'  AP={metrics["ap"]*100:.2f}%  '
              f'F1={metrics["best_f1"]*100:.2f}%  '
              f'P={metrics["best_p"]*100:.2f}%  '
              f'R={metrics["best_r"]*100:.2f}%  '
              f'FPS={np.mean(fps_list):.2f}')

        # Free GPU memory
        del net
        if use_cuda:
            torch.cuda.empty_cache()

    # Save report
    OUT_TXT = os.path.join(reports_dir, 'ablation_summary.txt')
    save_text_report(results, fps_data, total_gt, len(img_paths), OUT_TXT)

    # Build and save figure
    print('\n[INFO] Generating comparison figure...')
    OUT_FIG = os.path.join(figures_dir, 'ablation_comparison.png')
    fig = build_figure(results, fps_data)
    fig.savefig(OUT_FIG, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[INFO] Figure saved -> {OUT_FIG}')
    print('[DONE]')
