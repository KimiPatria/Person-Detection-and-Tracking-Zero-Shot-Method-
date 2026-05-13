# -*- coding: utf-8 -*-
"""
Qualitative visualisation of DAI-Net predictions on test images.

Produces a figure per image showing:
  Left  : Dark input (DarkISP-degraded) with PREDICTED bounding boxes  (green)
  Centre: Dark input with GROUND TRUTH bounding boxes                   (red)
  Right : Original (normal-light) image for reference

Usage
-----
  python visualize_predictions.py
  python visualize_predictions.py --weights weights/yolo_dark/dsfd.pth
  python visualize_predictions.py --ablation baseline --n_images 6
  python visualize_predictions.py --conf_thresh 0.3 --save_dir vis_output
"""

import os
import argparse
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data.config import cfg
from models.factory import build_net
from utils.DarkISP import Low_Illumination_Degrading

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE   = 640
PRED_COLOR = (0.18, 0.80, 0.44)   # green  (normalised 0-1 for matplotlib)
GT_COLOR   = (0.95, 0.26, 0.21)   # red
FONT_SIZE  = 7

# ── Ablation weight paths ──────────────────────────────────────────────────────
_WEIGHT_MAP = {
    'baseline':    'weights/ablation_baseline/dsfd.pth',
    'reflectance': 'weights/ablation_reflectance/dsfd.pth',
    'ref_kl':      'weights/ablation_ref_kl/dsfd.pth',
    'full':        'weights/yolo_dark/dsfd.pth',
}

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='DAI-Net Qualitative Visualisation')
parser.add_argument('--weights',      default=None,  type=str,
                    help='Path to model weights (.pth). '
                         'If omitted, inferred from --ablation.')
parser.add_argument('--ablation',     default='full', type=str,
                    choices=['baseline', 'reflectance', 'ref_kl', 'full'],
                    help='Which ablation variant to visualise.')
parser.add_argument('--data_dir',     default=None,  type=str,
                    help='Path to image directory (default: cfg.params.img_val_path).')
parser.add_argument('--n_images',     default=6,     type=int,
                    help='Number of images to visualise.')
parser.add_argument('--conf_thresh',  default=0.25,  type=float,
                    help='Confidence threshold for displayed predictions.')
parser.add_argument('--seed',         default=42,    type=int,
                    help='Random seed for image selection.')
parser.add_argument('--save_dir',     default='vis_output', type=str,
                    help='Directory to save output figures.')
parser.add_argument('--no_show',      action='store_true',
                    help='Do not open interactive plot windows.')
args = parser.parse_args()

random.seed(args.seed)
os.makedirs(args.save_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Using device: {device}')

# ── Resolve weight path ────────────────────────────────────────────────────────
weight_path = args.weights or _WEIGHT_MAP[args.ablation]
if not os.path.exists(weight_path):
    raise FileNotFoundError(
        f'Weight file not found: {weight_path}\n'
        f'Train the model first or pass --weights <path>.')
print(f'[INFO] Loading weights from: {weight_path}')

# ── Build and load model ───────────────────────────────────────────────────────
net = build_net('test', num_classes=1, model='yolo_dark')
ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)

# Handles both EMA state_dict save and {'epoch', 'weight'} checkpoint save
if isinstance(ckpt, dict) and 'weight' in ckpt:
    state = ckpt['weight']
elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
    state = ckpt
else:
    raise ValueError(f'Unrecognised checkpoint format in {weight_path}')

net.load_state_dict(state, strict=True)
net.to(device)
net.eval()
print('[INFO] Model loaded successfully.')

# ── Locate images ──────────────────────────────────────────────────────────────
data_dir = args.data_dir or cfg.params.img_val_path

if os.path.isdir(os.path.join(data_dir, 'images')):
    img_dir = os.path.join(data_dir, 'images')
    ann_dir = os.path.join(data_dir, 'annotations')
else:
    img_dir = data_dir
    ann_dir = data_dir

img_files = [f for f in os.listdir(img_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not img_files:
    raise RuntimeError(f'No images found in {img_dir}')

# Prefer images that have annotation files with at least one person
def _has_person(img_file):
    stem = os.path.splitext(img_file)[0]
    xml_path = os.path.join(ann_dir, stem + '.xml')
    if not os.path.exists(xml_path):
        xml_path = os.path.join(img_dir, stem + '.xml')
    if not os.path.exists(xml_path):
        return False
    try:
        root = ET.parse(xml_path).getroot()
        return any(
            obj.find('name').text.lower().strip() == 'person'
            for obj in root.iter('object')
        )
    except Exception:
        return False

annotated = [f for f in img_files if _has_person(f)]
pool = annotated if len(annotated) >= args.n_images else img_files
random.shuffle(pool)
selected = pool[:args.n_images]
print(f'[INFO] Visualising {len(selected)} images from {img_dir}')


# ── Helper: load GT boxes ──────────────────────────────────────────────────────
def load_gt_boxes(img_file, img_w, img_h):
    """Return list of (x1,y1,x2,y2) in pixel coordinates."""
    stem = os.path.splitext(img_file)[0]
    xml_path = os.path.join(ann_dir, stem + '.xml')
    if not os.path.exists(xml_path):
        xml_path = os.path.join(img_dir, stem + '.xml')
    if not os.path.exists(xml_path):
        return []
    boxes = []
    try:
        root = ET.parse(xml_path).getroot()
        for obj in root.iter('object'):
            if obj.find('name').text.lower().strip() != 'person':
                continue
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append((x1, y1, x2, y2))
    except Exception:
        pass
    return boxes


# ── Helper: run inference ──────────────────────────────────────────────────────
def predict(img_bgr):
    """
    img_bgr : (H, W, 3) uint8  BGR image (original resolution).
    Returns : list of (score, x1, y1, x2, y2) in *original* pixel coords.
    """
    h_orig, w_orig = img_bgr.shape[:2]

    # Resize to 640×640 for the model
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # To tensor [0,1]
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    # Apply DarkISP degradation
    tensor_norm = tensor  # already [0,1]
    with torch.no_grad():
        dark_tensor, _ = Low_Illumination_Degrading(tensor_norm[0])
    dark_tensor = dark_tensor.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output, _ = net.test_forward(dark_tensor)  # (1, 2, TOP_K, 5)

    dets = output[0, 1]   # (TOP_K, 5): [score, x1_n, y1_n, x2_n, y2_n]
    results = []
    for j in range(dets.shape[0]):
        score = dets[j, 0].item()
        if score < args.conf_thresh:
            continue
        x1_n, y1_n, x2_n, y2_n = dets[j, 1:5].tolist()
        # Denormalise to original image size
        x1 = x1_n * w_orig;  y1 = y1_n * h_orig
        x2 = x2_n * w_orig;  y2 = y2_n * h_orig
        results.append((score, x1, y1, x2, y2))
    return results


# ── Helper: draw boxes on a copy of the image ─────────────────────────────────
def draw_boxes(ax, boxes_with_score, color, label_prefix=''):
    """Draw bounding boxes on a matplotlib Axes."""
    for item in boxes_with_score:
        if len(item) == 5:
            score, x1, y1, x2, y2 = item
            label = f'{label_prefix}{score:.2f}'
        else:
            x1, y1, x2, y2 = item
            label = label_prefix.rstrip()

        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=1.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(y1 - 3, 0), label,
                fontsize=FONT_SIZE, color='white',
                bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor='none'))


# ── Main visualisation loop ────────────────────────────────────────────────────
for idx, img_file in enumerate(selected):
    img_path = os.path.join(img_dir, img_file)
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        print(f'[WARN] Cannot read {img_path}, skipping.')
        continue

    h_orig, w_orig = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Ground truth boxes (original resolution)
    gt_boxes = load_gt_boxes(img_file, w_orig, h_orig)

    # Predicted boxes (original resolution)
    preds = predict(img_bgr)

    # Dark version for display (apply DarkISP at display resolution)
    tensor_small = torch.from_numpy(
        cv2.cvtColor(cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE)),
                     cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    ).float() / 255.0
    with torch.no_grad():
        dark_display, _ = Low_Illumination_Degrading(tensor_small)
    dark_np = (dark_display.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    # Resize dark image back to original resolution for display
    dark_np = cv2.resize(dark_np, (w_orig, h_orig))

    # ── Figure layout: 3 panels per image ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f'Image: {img_file}  |  Ablation: {args.ablation}  |  '
        f'Conf ≥ {args.conf_thresh}  |  '
        f'Pred: {len(preds)} boxes  |  GT: {len(gt_boxes)} boxes',
        fontsize=9, y=1.01)

    # Panel 1 — Dark input + PREDICTIONS
    axes[0].imshow(dark_np)
    axes[0].set_title('Predictions (green)', fontsize=9, pad=3)
    axes[0].axis('off')
    pred_items = [(s, x1, y1, x2, y2) for s, x1, y1, x2, y2 in preds]
    draw_boxes(axes[0], pred_items, color=PRED_COLOR, label_prefix='')

    # Panel 2 — Dark input + GROUND TRUTH
    axes[1].imshow(dark_np)
    axes[1].set_title('Ground Truth (red)', fontsize=9, pad=3)
    axes[1].axis('off')
    gt_items = [(x1, y1, x2, y2) for x1, y1, x2, y2 in gt_boxes]
    draw_boxes(axes[1], gt_items, color=GT_COLOR, label_prefix='person ')

    # Panel 3 — Original (normal-light) reference
    axes[2].imshow(img_rgb)
    axes[2].set_title('Original (reference)', fontsize=9, pad=3)
    axes[2].axis('off')

    plt.tight_layout()

    out_path = os.path.join(args.save_dir, f'vis_{idx:02d}_{os.path.splitext(img_file)[0]}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'[INFO] Saved → {out_path}')

    if not args.no_show:
        plt.show()
    plt.close(fig)

print(f'\n[DONE] All figures saved to: {args.save_dir}/')
