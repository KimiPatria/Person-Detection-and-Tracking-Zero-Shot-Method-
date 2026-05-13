# -*- coding: utf-8 -*-
"""
Thesis Metrics — DAI-Net
========================
Five supplementary analyses required for thesis defence:

  1. Per-illumination breakdown   — AP per brightness tier (dark/dim/bright)
  2. Visual qualitative examples  — prediction vs. GT figure grid
  3. Failure case analysis        — worst-performing images (FN-heavy / FP-heavy)
  4. Model size report            — parameter count + GPU/CPU memory footprint
  5. FPS coefficient of variation — stability metric (CV = std/mean FPS)

Usage
-----
    python evaluate_thesis_metrics.py
    python evaluate_thesis_metrics.py --model yolo_dark --weights ./weights/dsfd.pth
    python evaluate_thesis_metrics.py --n_qual 8 --n_fail 6

Output
------
    result/thesis_metrics_<NNN>_<timestamp>/
        figures/illumination_breakdown.png
        figures/qualitative_examples.png
        figures/failure_cases.png
        figures/model_size.png
        figures/fps_cv.png
        reports/thesis_summary.txt
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
import matplotlib.patches as mpatches
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

# ─── Defaults ────────────────────────────────────────────────────────────────
IMAGES_DIR      = './dataset/roboflow/test/images/'
ANNOTATIONS_DIR = './dataset/roboflow/test/annotations/'
WEIGHTS_PATH    = './weights/yolo_dark/dsfd.pth'
RESULTS_ROOT    = './result/'
MODEL_TYPE      = 'yolo_dark'

CONF_THRESH = 0.25   # display/eval threshold
IOU_THRESH  = 0.50

# Illumination buckets (mean pixel value in [0,255] after converting to grey)
ILLUM_BINS  = {'Dark': (0, 60), 'Dim': (60, 120), 'Bright': (120, 256)}
ILLUM_ORDER = ['Dark', 'Dim', 'Bright']
ILLUM_COLORS = {'Dark': '#2d2d2d', 'Dim': '#f4a442', 'Bright': '#4caf50'}

# ─── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='DAI-Net Thesis Metrics')
parser.add_argument('--model',       default=MODEL_TYPE,   type=str)
parser.add_argument('--weights',     default=WEIGHTS_PATH, type=str)
parser.add_argument('--images',      default=IMAGES_DIR,   type=str)
parser.add_argument('--annotations', default=ANNOTATIONS_DIR, type=str)
parser.add_argument('--n_qual',      default=6,  type=int,
                    help='Number of images in qualitative grid.')
parser.add_argument('--n_fail',      default=6,  type=int,
                    help='Number of failure cases to visualise.')
parser.add_argument('--seed',        default=42, type=int)
args = parser.parse_args()

np.random.seed(args.seed)

IMAGES_DIR      = args.images
ANNOTATIONS_DIR = args.annotations
WEIGHTS_PATH    = args.weights
MODEL_TYPE      = args.model


# ═══════════════════════════════════════════════════════════════════════════
# Output directory
# ═══════════════════════════════════════════════════════════════════════════

def make_run_dir(tag='thesis_metrics'):
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    existing = glob.glob(os.path.join(RESULTS_ROOT, f'{tag}_*'))
    run_id   = len(existing) + 1
    stamp    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir  = os.path.join(RESULTS_ROOT, f'{tag}_{run_id:03d}_{stamp}')
    os.makedirs(os.path.join(run_dir, 'figures'),  exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'reports'),  exist_ok=True)
    return run_dir

RUN_DIR = make_run_dir()
print(f'[INFO] Saving outputs to: {RUN_DIR}')


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers  (adapted from evaluate_baseline.py)
# ═══════════════════════════════════════════════════════════════════════════

def to_chw_bgr(image):
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    image = image[[2, 1, 0], :, :]
    return image


def letterbox(img, target_size=640):
    h, w   = img.shape[:2]
    scale  = min(target_size / h, target_size / w)
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.zeros((target_size, target_size, 3), dtype=img.dtype)
    pad_top  = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, pad_left, pad_top


def detect_single(net, img, shrink=1.0):
    """Single-scale inference; returns (N,5) array [x1,y1,x2,y2,score]."""
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
    px_scale   = np.array([target_size] * 4)

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
        return np.zeros((0, 5))
    return np.column_stack((np.array(boxes), np.array(scores)))


def parse_voc_xml(xml_path):
    tree   = ET.parse(xml_path)
    bboxes = []
    for obj in tree.getroot().findall('object'):
        b = obj.find('bndbox')
        bboxes.append([
            float(b.find('xmin').text),
            float(b.find('ymin').text),
            float(b.find('xmax').text),
            float(b.find('ymax').text),
        ])
    return bboxes


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(aA + aB - inter)


def voc_ap(rec, prec):
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        p = np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0
        ap += p / 11.0
    return ap


def mean_luma(img_rgb):
    """Return mean luminance of an RGB image (0–255)."""
    grey = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(grey.mean())


def illum_tier(luma):
    for tier, (lo, hi) in ILLUM_BINS.items():
        if lo <= luma < hi:
            return tier
    return 'Bright'


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def _infer_model_type_from_ckpt(ckpt_keys):
    """Detect architecture from checkpoint key names."""
    if any(k.startswith('backbone.stem') for k in ckpt_keys):
        return 'yolo_dark'
    if any(k.startswith('vgg.') for k in ckpt_keys):
        # original DAINet uses vgg backbone with dark-ISP components
        return 'dark'
    if any(k.startswith('layer') for k in ckpt_keys):
        return 'resnet50'
    return None


def load_model():
    global MODEL_TYPE

    ckpt = torch.load(WEIGHTS_PATH,
                      map_location='cuda' if use_cuda else 'cpu')
    state = ckpt['weight'] if isinstance(ckpt, dict) and 'weight' in ckpt else ckpt

    # Auto-detect architecture when checkpoint doesn't match the requested model
    detected = _infer_model_type_from_ckpt(state.keys())
    if detected and detected != MODEL_TYPE:
        print(f'[WARN] Checkpoint keys suggest "{detected}" architecture '
              f'but --model={MODEL_TYPE} was requested.')
        print(f'[INFO] Switching model type to "{detected}" to match weights.')
        MODEL_TYPE = detected

    print(f'[INFO] Building model: {MODEL_TYPE}')
    num_classes = 1 if MODEL_TYPE == 'yolo_dark' else 2
    net = build_net('test', num_classes=num_classes, model=MODEL_TYPE)
    net.eval()
    net.load_state_dict(state)
    if use_cuda:
        net = net.cuda()
    print(f'[INFO] Weights loaded from {WEIGHTS_PATH}')
    return net


# ═══════════════════════════════════════════════════════════════════════════
# Full inference pass — collects everything needed for all five analyses
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(net):
    img_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, '*.jpg')))
    if not img_paths:
        img_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, '*.png')))
    n = len(img_paths)
    print(f'[INFO] Found {n} test images.')

    records = []   # one dict per image

    for idx, img_path in enumerate(img_paths, 1):
        img_id   = Path(img_path).stem
        xml_path = os.path.join(ANNOTATIONS_DIR, img_id + '.xml')

        img_rgb = np.array(Image.open(img_path).convert('RGB'))
        luma    = mean_luma(img_rgb)
        tier    = illum_tier(luma)

        gts = parse_voc_xml(xml_path) if os.path.exists(xml_path) else []

        t0 = time.time()
        with torch.no_grad():
            dets = detect_single(net, img_rgb, shrink=1.0)
        elapsed = time.time() - t0
        fps     = 1.0 / elapsed if elapsed > 0 else 0.0

        # filter by confidence
        if dets.shape[0] > 0:
            keep = dets[:, 4] >= CONF_THRESH
            dets = dets[keep]

        records.append({
            'img_id':   img_id,
            'img_path': img_path,
            'img_rgb':  img_rgb,
            'luma':     luma,
            'tier':     tier,
            'gts':      gts,           # list of [x1,y1,x2,y2]
            'dets':     dets,          # (K,5) float32 — filtered
            'fps':      fps,
        })
        print(f'\r[INFO] {idx}/{n} | tier={tier} luma={luma:.0f} FPS={fps:.1f}', end='')

    print()
    return records


# ═══════════════════════════════════════════════════════════════════════════
# Per-image TP / FN / FP counts (used by analyses 1 & 3)
# ═══════════════════════════════════════════════════════════════════════════

def count_tp_fp_fn(gts, dets, iou_thresh=IOU_THRESH):
    """Returns (tp, fp, fn) counts for one image."""
    matched_gt = [False] * len(gts)
    tp = fp = 0

    # sort detections by descending confidence
    if dets.shape[0] > 0:
        order = np.argsort(-dets[:, 4])
        dets  = dets[order]

    for det in dets:
        bb = det[:4]
        best_iou, best_k = -1.0, -1
        for k, gt in enumerate(gts):
            v = iou(bb, gt)
            if v > best_iou:
                best_iou, best_k = v, k

        if best_iou >= iou_thresh and best_k >= 0 and not matched_gt[best_k]:
            tp += 1
            matched_gt[best_k] = True
        else:
            fp += 1

    fn = matched_gt.count(False)
    return tp, fp, fn


def annotate_records(records):
    for r in records:
        r['tp'], r['fp'], r['fn'] = count_tp_fp_fn(r['gts'], r['dets'])
        n_gt = len(r['gts'])
        r['recall']    = r['tp'] / n_gt if n_gt > 0 else 1.0
        r['precision'] = r['tp'] / (r['tp'] + r['fp']) if (r['tp'] + r['fp']) > 0 else 1.0
        f1_denom = r['precision'] + r['recall']
        r['f1'] = 2 * r['precision'] * r['recall'] / f1_denom if f1_denom > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 1 — Per-illumination AP breakdown
# ═══════════════════════════════════════════════════════════════════════════

def compute_tier_ap(records, tier):
    """VOC AP for images belonging to `tier`."""
    subset = [r for r in records if r['tier'] == tier]
    if not subset:
        return 0.0, 0, 0

    # build detection list sorted by score
    all_dets = []
    all_gts  = {}
    total_gt = 0
    for r in subset:
        gts_reset = [{'bbox': g, 'matched': False} for g in r['gts']]
        all_gts[r['img_id']] = gts_reset
        total_gt += len(r['gts'])
        for det in r['dets']:
            all_dets.append((r['img_id'], float(det[4]), det[:4]))

    if total_gt == 0:
        return 0.0, len(subset), 0

    all_dets.sort(key=lambda x: x[1], reverse=True)
    nd    = len(all_dets)
    tp_v  = np.zeros(nd)
    fp_v  = np.zeros(nd)

    for i, (img_id, score, bb) in enumerate(all_dets):
        gts = all_gts.get(img_id, [])
        best_iou, best_k = -1.0, -1
        for k, g in enumerate(gts):
            v = iou(bb, g['bbox'])
            if v > best_iou:
                best_iou, best_k = v, k

        if best_iou >= IOU_THRESH and best_k >= 0 and not gts[best_k]['matched']:
            tp_v[i] = 1.
            gts[best_k]['matched'] = True
        else:
            fp_v[i] = 1.

    cum_tp = np.cumsum(tp_v)
    cum_fp = np.cumsum(fp_v)
    rec    = cum_tp / float(total_gt)
    prec   = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(float).eps)
    ap     = voc_ap(rec, prec)
    return ap, len(subset), total_gt


def plot_illumination_breakdown(records, save_path):
    results = {}
    for tier in ILLUM_ORDER:
        ap, n_imgs, n_gt = compute_tier_ap(records, tier)
        results[tier] = {'ap': ap, 'n_imgs': n_imgs, 'n_gt': n_gt}

    # also collect mean FPS & F1 per tier
    tier_fps = defaultdict(list)
    tier_f1  = defaultdict(list)
    for r in records:
        tier_fps[r['tier']].append(r['fps'])
        tier_f1[r['tier']].append(r['f1'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#f8f9fa')
    fig.suptitle('Detection Performance by Illumination Level',
                 fontsize=14, fontweight='bold', y=1.02)

    # ── AP bar chart ──
    ax = axes[0]
    aps    = [results[t]['ap']     for t in ILLUM_ORDER]
    colors = [ILLUM_COLORS[t]      for t in ILLUM_ORDER]
    bars   = ax.bar(ILLUM_ORDER, aps, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, aps):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('AP @ IoU=0.50', fontsize=11)
    ax.set_title('Average Precision', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # label with image count
    for bar, t in zip(bars, ILLUM_ORDER):
        ax.text(bar.get_x() + bar.get_width() / 2., -0.07,
                f'n={results[t]["n_imgs"]}', ha='center', va='top',
                fontsize=9, color='#555')

    # ── Mean F1 bar chart ──
    ax = axes[1]
    f1s = [np.mean(tier_f1[t]) if tier_f1[t] else 0.0 for t in ILLUM_ORDER]
    bars = ax.bar(ILLUM_ORDER, f1s, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Mean F1-Score', fontsize=11)
    ax.set_title('F1-Score by Illumination', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # ── FPS box plot ──
    ax = axes[2]
    fps_data = [tier_fps[t] if tier_fps[t] else [0.0] for t in ILLUM_ORDER]
    bp = ax.boxplot(fps_data, patch_artist=True, widths=0.4,
                    medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_xticklabels(ILLUM_ORDER)
    ax.set_ylabel('FPS', fontsize=11)
    ax.set_title('FPS Distribution by Illumination', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK]  Illumination breakdown → {save_path}')
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 2 — Qualitative examples  (prediction vs. GT)
# ═══════════════════════════════════════════════════════════════════════════

def _draw_boxes(ax, img_rgb, boxes, color, label=None):
    ax.imshow(img_rgb)
    ax.axis('off')
    if label:
        ax.set_title(label, fontsize=8, pad=3)
    h, w = img_rgb.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        rect = mpatches.FancyArrowPatch  # unused — use Rectangle below
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=1.5, edgecolor=color,
                              facecolor='none', clip_on=True)
        ax.add_patch(rect)
        if len(box) == 5:
            ax.text(x1, max(y1 - 3, 0), f'{box[4]:.2f}',
                    color='white', fontsize=6,
                    bbox=dict(facecolor=color, alpha=0.7, pad=1, linewidth=0))


def plot_qualitative_examples(records, save_path, n=6):
    # sample a spread across illumination tiers; fall back to random
    chosen = []
    per_tier = max(1, n // 3)
    for tier in ILLUM_ORDER:
        pool = [r for r in records if r['tier'] == tier and len(r['gts']) > 0]
        np.random.shuffle(pool)
        chosen.extend(pool[:per_tier])
    if len(chosen) < n:
        remaining = [r for r in records if r not in chosen and len(r['gts']) > 0]
        np.random.shuffle(remaining)
        chosen.extend(remaining[:n - len(chosen)])
    chosen = chosen[:n]

    cols = 2   # (predicted | GT) per image — we stack rows
    rows = len(chosen)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2),
                             facecolor='#111111')
    fig.suptitle('Qualitative Predictions  —  Green: Predicted   |   Red: Ground Truth',
                 fontsize=11, color='white', fontweight='bold', y=1.005)

    if rows == 1:
        axes = [axes]

    for row, r in enumerate(chosen):
        ax_pred, ax_gt = axes[row][0], axes[row][1]

        # predictions
        _draw_boxes(ax_pred, r['img_rgb'], r['dets'],
                    color='#2ecc71',
                    label=f"Predicted  [{r['tier']} | luma={r['luma']:.0f}]")

        # ground truth
        _draw_boxes(ax_gt, r['img_rgb'], r['gts'],
                    color='#e74c3c',
                    label=f"GT  (TP={r['tp']} FP={r['fp']} FN={r['fn']})")

        # dark background for the image axes
        for ax in (ax_pred, ax_gt):
            ax.set_facecolor('#111111')

    plt.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#111111')
    plt.close(fig)
    print(f'[OK]  Qualitative examples  → {save_path}')


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 3 — Failure case analysis
# ═══════════════════════════════════════════════════════════════════════════

def plot_failure_cases(records, save_path, n=6):
    # Rank by total errors: fn weighted more than fp (missed detections are worse)
    def failure_score(r):
        return r['fn'] * 2 + r['fp']

    ranked = sorted(
        [r for r in records if len(r['gts']) > 0],
        key=failure_score, reverse=True
    )
    chosen = ranked[:n]

    cols = 3
    rows = max(1, (len(chosen) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5),
                             facecolor='#111111')
    fig.suptitle('Failure Cases  —  Green: Predicted   Red: Missed GT   Orange: False Positive',
                 fontsize=11, color='white', fontweight='bold', y=1.005)

    axes_flat = np.array(axes).flatten() if rows > 1 else [axes] if cols == 1 else list(axes)

    for i, r in enumerate(chosen):
        ax = axes_flat[i]
        ax.imshow(r['img_rgb'])
        ax.axis('off')
        ax.set_facecolor('#111111')

        matched_gt = [False] * len(r['gts'])
        tp_dets, fp_dets = [], []

        if r['dets'].shape[0] > 0:
            order = np.argsort(-r['dets'][:, 4])
            dets_sorted = r['dets'][order]
            for det in dets_sorted:
                bb = det[:4]
                best_iou, best_k = -1.0, -1
                for k, gt in enumerate(r['gts']):
                    v = iou(bb, gt)
                    if v > best_iou:
                        best_iou, best_k = v, k
                if best_iou >= IOU_THRESH and best_k >= 0 and not matched_gt[best_k]:
                    matched_gt[best_k] = True
                    tp_dets.append(det)
                else:
                    fp_dets.append(det)

        # draw TP predictions in green
        for det in tp_dets:
            x1, y1, x2, y2 = det[:4]
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=1.5, edgecolor='#2ecc71',
                                       facecolor='none'))

        # draw FP predictions in orange
        for det in fp_dets:
            x1, y1, x2, y2 = det[:4]
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=1.5, edgecolor='#f39c12',
                                       facecolor='none'))

        # draw unmatched GT in red (missed)
        for k, gt in enumerate(r['gts']):
            x1, y1, x2, y2 = gt
            color = '#e74c3c' if not matched_gt[k] else '#aaaaaa'
            lw    = 2.0       if not matched_gt[k] else 1.0
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=lw, edgecolor=color,
                                       facecolor='none', linestyle='--'))

        title  = f"FN={r['fn']} FP={r['fp']}  [{r['tier']} | F1={r['f1']:.2f}]"
        ax.set_title(title, fontsize=8, color='white', pad=3)

    # hide spare axes
    for j in range(len(chosen), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # legend
    legend_patches = [
        mpatches.Patch(edgecolor='#2ecc71', facecolor='none', label='TP (correct)'),
        mpatches.Patch(edgecolor='#f39c12', facecolor='none', label='FP (false alarm)'),
        mpatches.Patch(edgecolor='#e74c3c', facecolor='none', linestyle='--',
                       label='FN (missed GT)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               framealpha=0.2, labelcolor='white', fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#111111')
    plt.close(fig)
    print(f'[OK]  Failure cases         → {save_path}')


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 4 — Model parameter count + memory footprint
# ═══════════════════════════════════════════════════════════════════════════

def compute_model_size(net):
    total_params     = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # Memory in MB: each float32 param = 4 bytes
    param_mem_mb   = total_params * 4 / (1024 ** 2)

    # Activation memory estimate for a single 640×640 forward pass
    # (rough: ~3× param memory for typical CNNs)
    activation_mem_mb = param_mem_mb * 3.0

    # Try to get actual GPU memory allocated after a dummy forward
    gpu_alloc_mb = None
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        dummy = torch.zeros(1, 3, 640, 640).cuda()
        with torch.no_grad():
            try:
                net.test_forward(dummy)
                gpu_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            except Exception:
                pass

    return {
        'total_params':       total_params,
        'trainable_params':   trainable_params,
        'param_mem_mb':       param_mem_mb,
        'activation_mem_mb':  activation_mem_mb,
        'gpu_peak_mb':        gpu_alloc_mb,
    }


def plot_model_size(size_info, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#f8f9fa')
    fig.suptitle('Model Size & Memory Footprint', fontsize=14, fontweight='bold')

    # ── Parameter breakdown pie ──
    ax = axes[0]
    frozen = size_info['total_params'] - size_info['trainable_params']
    labels = ['Trainable', 'Frozen']
    sizes  = [size_info['trainable_params'], frozen]
    colors = ['#1a73e8', '#dadce0']
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90,
        wedgeprops=dict(linewidth=1.5, edgecolor='white'))
    for t in autotexts:
        t.set_fontsize(11)
    ax.set_title(
        f'Parameters\nTotal: {size_info["total_params"]:,}  '
        f'({size_info["total_params"] / 1e6:.2f} M)',
        fontsize=11, fontweight='bold')

    # ── Memory bar chart ──
    ax = axes[1]
    mem_labels = ['Param\n(weights)', 'Est. Activation\n(single pass)']
    mem_values = [size_info['param_mem_mb'], size_info['activation_mem_mb']]
    bar_colors = ['#1a73e8', '#fbbc04']

    if size_info['gpu_peak_mb'] is not None:
        mem_labels.append('GPU Peak\n(measured)')
        mem_values.append(size_info['gpu_peak_mb'])
        bar_colors.append('#34a853')

    bars = ax.bar(mem_labels, mem_values, color=bar_colors,
                  width=0.45, edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, mem_values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                f'{v:.1f} MB', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.set_ylabel('Memory (MB)', fontsize=11)
    ax.set_title('Memory Footprint', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_ylim(0, max(mem_values) * 1.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK]  Model size            → {save_path}')
    return size_info


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 5 — FPS Coefficient of Variation (CV = std / mean)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fps_cv(records, save_path):
    fps_arr  = np.array([r['fps'] for r in records])
    mean_fps = float(np.mean(fps_arr))
    std_fps  = float(np.std(fps_arr))
    cv_fps   = std_fps / mean_fps if mean_fps > 0 else 0.0
    min_fps  = float(np.min(fps_arr))
    max_fps  = float(np.max(fps_arr))
    p5       = float(np.percentile(fps_arr, 5))   # worst-5 % tail

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#f8f9fa')
    fig.suptitle('FPS Distribution & Coefficient of Variation (CV)',
                 fontsize=14, fontweight='bold')

    # ── Histogram + KDE ──
    ax = axes[0]
    ax.hist(fps_arr, bins=30, color='#1a73e8', alpha=0.75,
            edgecolor='white', linewidth=0.8)
    ax.axvline(mean_fps, color='#ea4335', linewidth=2,
               linestyle='--', label=f'Mean={mean_fps:.1f}')
    ax.axvline(p5, color='#fbbc04', linewidth=2,
               linestyle=':', label=f'P5={p5:.1f}')
    ax.set_xlabel('FPS', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Per-Image FPS Distribution', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(fontsize=10)

    # ── CV summary table ──
    ax = axes[1]
    ax.axis('off')
    table_data = [
        ['Metric',              'Value'],
        ['Mean FPS',            f'{mean_fps:.2f}'],
        ['Std FPS',             f'{std_fps:.2f}'],
        ['CV (std/mean)',       f'{cv_fps:.4f}   ({cv_fps*100:.2f} %)'],
        ['Min FPS',             f'{min_fps:.2f}'],
        ['Max FPS',             f'{max_fps:.2f}'],
        ['P5 (worst 5%)',       f'{p5:.2f}'],
        ['N images',            f'{len(fps_arr)}'],
    ]
    tbl = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center', cellLoc='left',
        colWidths=[0.52, 0.44]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)

    # header row styling
    for col in range(2):
        tbl[0, col].set_facecolor('#1a73e8')
        tbl[0, col].set_text_props(color='white', fontweight='bold')

    # highlight CV row
    for col in range(2):
        tbl[3, col].set_facecolor('#fce8e6')

    ax.set_title(
        f'CV = {cv_fps:.4f}  →  {"Stable ✓" if cv_fps < 0.25 else "Variable ✗"}',
        fontsize=12, fontweight='bold', pad=12,
        color='#34a853' if cv_fps < 0.25 else '#ea4335')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK]  FPS CV analysis       → {save_path}')

    return {
        'mean_fps': mean_fps,
        'std_fps':  std_fps,
        'cv_fps':   cv_fps,
        'min_fps':  min_fps,
        'max_fps':  max_fps,
        'p5_fps':   p5,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Text summary report
# ═══════════════════════════════════════════════════════════════════════════

def write_report(illum_results, size_info, fps_stats, records, save_path):
    total = len(records)
    tier_counts = defaultdict(int)
    for r in records:
        tier_counts[r['tier']] += 1

    overall_f1  = np.mean([r['f1']  for r in records])
    overall_rec = np.mean([r['recall']    for r in records])
    overall_pre = np.mean([r['precision'] for r in records])

    lines = [
        '=' * 70,
        '  DAI-Net — Thesis Metrics Summary',
        f'  Model   : {MODEL_TYPE}',
        f'  Weights : {WEIGHTS_PATH}',
        f'  Date    : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}',
        '=' * 70,
        '',
        '─── 1. Per-Illumination Breakdown ─────────────────────────────────',
        f'{"Tier":<10} {"N images":>10} {"AP@0.5":>10} {"Mean F1":>10}',
        '-' * 44,
    ]
    tier_f1 = defaultdict(list)
    for r in records:
        tier_f1[r['tier']].append(r['f1'])

    for tier in ILLUM_ORDER:
        res    = illum_results.get(tier, {})
        ap     = res.get('ap', 0.0)
        n_imgs = res.get('n_imgs', 0)
        mf1    = np.mean(tier_f1[tier]) if tier_f1[tier] else 0.0
        lines.append(f'{tier:<10} {n_imgs:>10} {ap:>10.4f} {mf1:>10.4f}')

    lines += [
        '',
        '─── Overall ────────────────────────────────────────────────────────',
        f'  Images evaluated : {total}',
        f'  Mean Precision   : {overall_pre:.4f}',
        f'  Mean Recall      : {overall_rec:.4f}',
        f'  Mean F1          : {overall_f1:.4f}',
        '',
        '─── 4. Model Size ──────────────────────────────────────────────────',
        f'  Total parameters  : {size_info["total_params"]:,}',
        f'  Trainable params  : {size_info["trainable_params"]:,}',
        f'  Weight memory     : {size_info["param_mem_mb"]:.2f} MB',
        f'  Est. activation   : {size_info["activation_mem_mb"]:.2f} MB',
    ]
    if size_info['gpu_peak_mb'] is not None:
        lines.append(f'  GPU peak (640px)  : {size_info["gpu_peak_mb"]:.2f} MB')

    lines += [
        '',
        '─── 5. FPS Coefficient of Variation ────────────────────────────────',
        f'  Mean FPS  : {fps_stats["mean_fps"]:.2f}',
        f'  Std  FPS  : {fps_stats["std_fps"]:.2f}',
        f'  CV        : {fps_stats["cv_fps"]:.4f}  ({fps_stats["cv_fps"]*100:.2f} %)',
        f'  Min FPS   : {fps_stats["min_fps"]:.2f}',
        f'  Max FPS   : {fps_stats["max_fps"]:.2f}',
        f'  P5        : {fps_stats["p5_fps"]:.2f}',
        f'  Verdict   : {"Stable (CV < 0.25)" if fps_stats["cv_fps"] < 0.25 else "Variable (CV >= 0.25)"}',
        '',
        '─── Failure Cases ──────────────────────────────────────────────────',
    ]
    ranked = sorted(
        [r for r in records if len(r['gts']) > 0],
        key=lambda r: r['fn'] * 2 + r['fp'], reverse=True
    )
    lines.append(f'  {"Image":<30} {"FN":>4} {"FP":>4} {"F1":>6} {"Tier":>7}')
    lines.append('  ' + '-' * 56)
    for r in ranked[:10]:
        lines.append(
            f'  {r["img_id"]:<30} {r["fn"]:>4} {r["fp"]:>4} '
            f'{r["f1"]:>6.3f} {r["tier"]:>7}')

    lines += ['', '=' * 70]

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'[OK]  Text report           → {save_path}')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f'[INFO] Device: {device_name}')

    # ── Load model ──────────────────────────────────────────────────────────
    net = load_model()

    # ── Analysis 4: model size (no inference needed) ─────────────────────
    print('\n[STEP 4] Model size...')
    size_info = compute_model_size(net)
    print(f'         Params: {size_info["total_params"]:,}  '
          f'({size_info["total_params"]/1e6:.2f} M)')
    print(f'         Weight mem: {size_info["param_mem_mb"]:.1f} MB')
    size_path = os.path.join(RUN_DIR, 'figures', 'model_size.png')
    plot_model_size(size_info, size_path)

    # ── Inference ───────────────────────────────────────────────────────────
    print('\n[STEP] Running inference...')
    records = run_inference(net)

    # ── Per-image TP/FP/FN ──────────────────────────────────────────────────
    annotate_records(records)

    # ── Analysis 1: Illumination breakdown ──────────────────────────────────
    print('\n[STEP 1] Illumination breakdown...')
    illum_path    = os.path.join(RUN_DIR, 'figures', 'illumination_breakdown.png')
    illum_results = plot_illumination_breakdown(records, illum_path)

    # ── Analysis 2: Qualitative examples ────────────────────────────────────
    print('\n[STEP 2] Qualitative examples...')
    qual_path = os.path.join(RUN_DIR, 'figures', 'qualitative_examples.png')
    plot_qualitative_examples(records, qual_path, n=args.n_qual)

    # ── Analysis 3: Failure cases ────────────────────────────────────────────
    print('\n[STEP 3] Failure case analysis...')
    fail_path = os.path.join(RUN_DIR, 'figures', 'failure_cases.png')
    plot_failure_cases(records, fail_path, n=args.n_fail)

    # ── Analysis 5: FPS CV ───────────────────────────────────────────────────
    print('\n[STEP 5] FPS CV analysis...')
    fps_path  = os.path.join(RUN_DIR, 'figures', 'fps_cv.png')
    fps_stats = plot_fps_cv(records, fps_path)

    # ── Text report ─────────────────────────────────────────────────────────
    report_path = os.path.join(RUN_DIR, 'reports', 'thesis_summary.txt')
    write_report(illum_results, size_info, fps_stats, records, report_path)

    print(f'\n[DONE] All outputs saved to: {RUN_DIR}')


if __name__ == '__main__':
    main()
