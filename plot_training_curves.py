# -*- coding: utf-8 -*-
"""
Plot Training Loss Curves — DAI-Net
====================================
Reads the loss_history.csv produced by train_yolo.py and generates
a publication-ready figure showing all loss components over epochs.

Usage
-----
    python plot_training_curves.py
    python plot_training_curves.py --csv weights/yolo_dark/loss_history.csv
    python plot_training_curves.py --out result/training_curves.png
"""

from __future__ import division, print_function

import os
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_CSV = './weights/yolo_dark/loss_history.csv'
DEFAULT_OUT = './result/training_curves.png'

# ─── Style ───────────────────────────────────────────────────────────────────
COLORS = {
    'total':  '#1a73e8',
    'box':    '#ea4335',
    'cls':    '#fbbc05',
    'enh':    '#34a853',
    'kl':     '#ff6d01',
    'val':    '#9c27b0',
    'lr':     '#607d8b',
}
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
    ax.legend(fontsize=FONT_TICK, framealpha=0.7)


def load_csv(csv_path):
    """Load loss_history.csv into a dict of numpy arrays."""
    data = {
        'epoch': [], 'train_loss': [], 'train_box': [], 'train_cls': [],
        'train_enh': [], 'train_kl': [], 'val_loss': [], 'lr': [],
    }
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['epoch'].append(int(row['epoch']))
            data['train_loss'].append(float(row['train_loss']))
            data['train_box'].append(float(row['train_box']))
            data['train_cls'].append(float(row['train_cls']))
            data['train_enh'].append(float(row['train_enh']))
            data['train_kl'].append(float(row['train_kl']))
            data['val_loss'].append(float(row['val_loss']) if row['val_loss'] else np.nan)
            data['lr'].append(float(row['lr']))
    for k in data:
        data[k] = np.array(data[k])
    return data


def build_figure(data):
    """Build a 2x3 dashboard of training curves."""
    epochs = data['epoch']

    fig = plt.figure(figsize=(20, 12), facecolor='#f8f9fa')
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.06, right=0.97,
                           top=0.88, bottom=0.08,
                           hspace=0.38, wspace=0.30)

    ax_total = fig.add_subplot(gs[0, 0])
    ax_comp  = fig.add_subplot(gs[0, 1])
    ax_val   = fig.add_subplot(gs[0, 2])
    ax_box   = fig.add_subplot(gs[1, 0])
    ax_enh   = fig.add_subplot(gs[1, 1])
    ax_lr    = fig.add_subplot(gs[1, 2])

    # ── 1. Total training loss ───────────────────────────────────────────
    ax_total.plot(epochs, data['train_loss'], color=COLORS['total'],
                  lw=2, label='Train Loss')
    # overlay val loss where available
    val_mask = ~np.isnan(data['val_loss'])
    if val_mask.any():
        ax_total.plot(epochs[val_mask], data['val_loss'][val_mask],
                      color=COLORS['val'], lw=2, linestyle='--',
                      marker='o', markersize=4, label='Val Loss')
    min_idx = np.argmin(data['train_loss'])
    ax_total.annotate(f"min={data['train_loss'][min_idx]:.4f}\n(ep {int(epochs[min_idx])})",
                      xy=(epochs[min_idx], data['train_loss'][min_idx]),
                      xytext=(epochs[min_idx] + 3, data['train_loss'][min_idx] + 0.1),
                      fontsize=8, color=COLORS['total'],
                      arrowprops=dict(arrowstyle='->', color=COLORS['total'], lw=0.8))
    _style_ax(ax_total, 'Total Loss (Train vs Val)', 'Epoch', 'Loss')

    # ── 2. All components stacked ────────────────────────────────────────
    ax_comp.plot(epochs, data['train_box'], color=COLORS['box'],
                 lw=1.5, label='Box (CIoU)')
    ax_comp.plot(epochs, data['train_cls'], color=COLORS['cls'],
                 lw=1.5, label='Cls (BCE)')
    ax_comp.plot(epochs, data['train_enh'], color=COLORS['enh'],
                 lw=1.5, label='Enhance (L1+SSIM)')
    ax_comp.plot(epochs, data['train_kl'],  color=COLORS['kl'],
                 lw=1.5, label='KL (mutual)')
    _style_ax(ax_comp, 'Loss Components (All)', 'Epoch', 'Loss')

    # ── 3. Validation loss ───────────────────────────────────────────────
    if val_mask.any():
        ax_val.plot(epochs[val_mask], data['val_loss'][val_mask],
                    color=COLORS['val'], lw=2, marker='o', markersize=5,
                    label='Val Loss')
        val_min_idx = np.nanargmin(data['val_loss'])
        ax_val.annotate(f"best={data['val_loss'][val_min_idx]:.4f}\n(ep {int(epochs[val_min_idx])})",
                        xy=(epochs[val_min_idx], data['val_loss'][val_min_idx]),
                        xytext=(epochs[val_min_idx] + 3, data['val_loss'][val_min_idx] + 0.05),
                        fontsize=8, color=COLORS['val'],
                        arrowprops=dict(arrowstyle='->', color=COLORS['val'], lw=0.8))
    else:
        ax_val.text(0.5, 0.5, 'No validation data',
                    transform=ax_val.transAxes, ha='center', va='center',
                    fontsize=12, color='#999')
    _style_ax(ax_val, 'Validation Loss', 'Epoch', 'Loss')

    # ── 4. Box loss (detection regression) ───────────────────────────────
    ax_box.plot(epochs, data['train_box'], color=COLORS['box'], lw=2,
                label='Box Loss')
    ax_box.fill_between(epochs, data['train_box'], alpha=0.1, color=COLORS['box'])
    _style_ax(ax_box, 'Box Loss (CIoU)', 'Epoch', 'Loss')

    # ── 5. Enhancement loss ──────────────────────────────────────────────
    ax_enh.plot(epochs, data['train_enh'], color=COLORS['enh'], lw=2,
                label='Enhance Loss')
    ax_enh.plot(epochs, data['train_kl'], color=COLORS['kl'], lw=2,
                linestyle='--', label='KL Loss')
    ax_enh.fill_between(epochs, data['train_enh'], alpha=0.1, color=COLORS['enh'])
    _style_ax(ax_enh, 'Enhancement & KL Loss', 'Epoch', 'Loss')

    # ── 6. Learning rate schedule ────────────────────────────────────────
    ax_lr.plot(epochs, data['lr'], color=COLORS['lr'], lw=2, label='LR')
    ax_lr.set_yscale('log')
    _style_ax(ax_lr, 'Learning Rate Schedule', 'Epoch', 'LR')

    # ── Main title ───────────────────────────────────────────────────────
    n_epochs = int(epochs[-1]) if len(epochs) else 0
    fig.suptitle(
        f'DAI-Net Training Loss Curves  |  {n_epochs} Epochs  |  YOLOv8n + Retinex',
        fontsize=16, fontweight='bold', color='#202124', y=0.96
    )

    return fig


def save_summary(data, out_path):
    """Save a plain-text training summary alongside the figure."""
    txt_path = out_path.replace('.png', '_summary.txt')
    epochs = data['epoch']
    n = len(epochs)
    val_mask = ~np.isnan(data['val_loss'])

    lines = [
        '=' * 60,
        'DAI-Net Training Summary',
        '=' * 60,
        f'  Total epochs             : {int(epochs[-1]) if n else 0}',
        f'  Final train loss         : {data["train_loss"][-1]:.6f}' if n else '',
        f'  Min   train loss         : {data["train_loss"].min():.6f}'
        f'  (epoch {int(epochs[np.argmin(data["train_loss"])])})'  if n else '',
        '',
        f'  Final val loss           : {data["val_loss"][val_mask][-1]:.6f}'
        if val_mask.any() else '  Final val loss           : N/A',
        f'  Best  val loss           : {np.nanmin(data["val_loss"]):.6f}'
        f'  (epoch {int(epochs[np.nanargmin(data["val_loss"])])})'
        if val_mask.any() else '  Best  val loss           : N/A',
        '',
        '  Loss components (final epoch):',
        f'    Box  (CIoU)            : {data["train_box"][-1]:.6f}' if n else '',
        f'    Cls  (BCE)             : {data["train_cls"][-1]:.6f}' if n else '',
        f'    Enhance (L1+SSIM)      : {data["train_enh"][-1]:.6f}' if n else '',
        f'    KL   (mutual)          : {data["train_kl"][-1]:.6f}' if n else '',
        '',
        f'  Final LR                 : {data["lr"][-1]:.6e}' if n else '',
        '=' * 60,
    ]
    report = '\n'.join(lines)
    print(report)
    with open(txt_path, 'w') as f:
        f.write(report + '\n')
    print(f'[INFO] Summary saved → {txt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot DAI-Net training curves')
    parser.add_argument('--csv', default=DEFAULT_CSV,
                        help=f'Path to loss_history.csv (default: {DEFAULT_CSV})')
    parser.add_argument('--out', default=DEFAULT_OUT,
                        help=f'Output PNG path (default: {DEFAULT_OUT})')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f'[ERROR] CSV not found: {args.csv}')
        print(f'        Run train_yolo.py first to generate loss_history.csv.')
        exit(1)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    print(f'[INFO] Loading {args.csv}')
    data = load_csv(args.csv)
    print(f'[INFO] Found {len(data["epoch"])} epochs of training data.')

    fig = build_figure(data)
    fig.savefig(args.out, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[INFO] Figure saved → {args.out}')

    save_summary(data, args.out)
    print('[DONE]')
