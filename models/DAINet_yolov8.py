# -*- coding: utf-8 -*-
"""
DAI-Net with YOLOv8n backbone.

Zero-shot low-light human detection.
  - Backbone upgraded: VGG16-DSFD → YOLOv8n (138M → ~3.2M params)
  - Zero-shot mechanism preserved: ReflectanceBranch + DistillKL alignment
  - Detection head: anchor-free, 3 FPN scales (strides 8 / 16 / 32)
"""

from __future__ import division, absolute_import, print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yolov8_modules import Conv, C2f, SPPF


# ═══════════════════════════════════════════════════════════════════════════
# Zero-shot components  (preserved from original DAI-Net)
# ═══════════════════════════════════════════════════════════════════════════

class ReflectanceBranch(nn.Module):
    """
    Decodes an illumination-invariant reflectance map R from shallow features.

    In the original DAI-Net this tapped vgg[4] (C=64, stride 2).
    Here it taps the YOLOv8n stem output (C=16, stride 2) — equivalent depth.

    Output: (B, 3, H, W)  — same spatial size as model input (upsample ×2).
    """
    def __init__(self, in_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),   # undo stride-2 stem
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class DistillKL(nn.Module):
    """KL-divergence loss for mutual day/night feature alignment (T=4.0)."""
    def __init__(self, T: float = 4.0):
        super().__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s  = F.log_softmax(y_s / self.T, dim=1)
        p_t  = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss


# ═══════════════════════════════════════════════════════════════════════════
# YOLOv8n backbone
# ═══════════════════════════════════════════════════════════════════════════

class YOLOv8nBackbone(nn.Module):
    """
    YOLOv8n feature extractor  (width=0.25, depth=0.33 — nano config).

    For a 640×640 input returns:
        stem : (B, 16,  320, 320)   ← ReflectanceBranch tap
        P3   : (B, 64,   80,  80)   stride  8
        P4   : (B, 128,  40,  40)   stride 16
        P5   : (B, 256,  20,  20)   stride 32  (after SPPF)
    """
    def __init__(self):
        super().__init__()
        self.stem   = Conv(3,   16,  3, 2)                          # /2  → 16ch
        self.stage1 = nn.Sequential(                                # /4  → 32ch  (P2, discarded)
            Conv(16,  32,  3, 2),
            C2f(32,   32,  n=1, shortcut=True),
        )
        self.stage2 = nn.Sequential(                                # /8  → 64ch  (P3)
            Conv(32,  64,  3, 2),
            C2f(64,   64,  n=2, shortcut=True),
        )
        self.stage3 = nn.Sequential(                                # /16 → 128ch (P4)
            Conv(64,  128, 3, 2),
            C2f(128, 128,  n=2, shortcut=True),
        )
        self.stage4 = nn.Sequential(                                # /32 → 256ch (P5)
            Conv(128, 256, 3, 2),
            C2f(256, 256,  n=1, shortcut=True),
            SPPF(256, 256),
        )

    def forward(self, x):
        stem = self.stem(x)           # (B, 16,  H/2,  W/2)
        p2   = self.stage1(stem)      # (B, 32,  H/4,  W/4)  — unused in neck
        P3   = self.stage2(p2)        # (B, 64,  H/8,  W/8)
        P4   = self.stage3(P3)        # (B, 128, H/16, W/16)
        P5   = self.stage4(P4)        # (B, 256, H/32, W/32)
        return stem, P3, P4, P5


# ═══════════════════════════════════════════════════════════════════════════
# PAN-FPN neck
# ═══════════════════════════════════════════════════════════════════════════

class YOLOv8nNeck(nn.Module):
    """
    Path-Aggregation FPN for YOLOv8n.

    Input : P3(C64), P4(C128), P5(C256)
    Output: N3(C64 / small),  N4(C128 / medium),  N5(C256 / large)
    """
    def __init__(self):
        super().__init__()
        # ── top-down (P5 → P3) ─────────────────────────────────────────
        self.up       = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_td4  = C2f(256 + 128, 128, n=1, shortcut=False)   # N4-td
        self.c2f_td3  = C2f(128 +  64,  64, n=1, shortcut=False)   # N3
        # ── bottom-up (N3 → N5) ────────────────────────────────────────
        self.down1    = Conv( 64,  64, 3, 2)
        self.c2f_bu4  = C2f( 64 + 128, 128, n=1, shortcut=False)   # N4
        self.down2    = Conv(128, 128, 3, 2)
        self.c2f_bu5  = C2f(128 + 256, 256, n=1, shortcut=False)   # N5

    def forward(self, P3, P4, P5):
        # top-down
        n4_td = self.c2f_td4(torch.cat([self.up(P5), P4], 1))
        N3    = self.c2f_td3(torch.cat([self.up(n4_td), P3], 1))
        # bottom-up
        N4    = self.c2f_bu4(torch.cat([self.down1(N3), n4_td], 1))
        N5    = self.c2f_bu5(torch.cat([self.down2(N4), P5],   1))
        return N3, N4, N5


# ═══════════════════════════════════════════════════════════════════════════
# Anchor-free detection head
# ═══════════════════════════════════════════════════════════════════════════

class DetectHead(nn.Module):
    """
    Per-scale anchor-free detection head.

    Predictions per grid point:
        reg branch → (dx, dy, log_w, log_h)
        cls branch → class logit(s)

    Decoding at stride s, grid cell (gi, gj):
        cx = (gi + sigmoid(dx)) * s        [pixel]
        cy = (gj + sigmoid(dy)) * s
        w  = exp(log_w) * s
        h  = exp(log_h) * s
        → (x1, y1, x2, y2) = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
    """
    def __init__(self, in_ch: int, num_classes: int = 1):
        super().__init__()
        self.reg = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1), nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1), nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, 4, 1),
        )
        self.cls = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1), nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1), nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, num_classes, 1),
        )
        # initialise cls bias for π≈0.01 (prevents loss explosion at epoch 0)
        nn.init.constant_(self.cls[-1].bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        return self.reg(x), self.cls(x)     # (B,4,H,W), (B,nc,H,W)


# ═══════════════════════════════════════════════════════════════════════════
# Loss helpers
# ═══════════════════════════════════════════════════════════════════════════

def ciou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Complete IoU loss between predicted and target boxes.
    Both inputs: (N, 4) in pixel (x1, y1, x2, y2).
    Returns: (N,) scalar loss values in [0, 2+].
    """
    pw = (pred[:, 2]   - pred[:, 0]).clamp(min=1e-6)
    ph = (pred[:, 3]   - pred[:, 1]).clamp(min=1e-6)
    gw = (target[:, 2] - target[:, 0]).clamp(min=1e-6)
    gh = (target[:, 3] - target[:, 1]).clamp(min=1e-6)

    # IoU
    ix1 = torch.max(pred[:, 0], target[:, 0])
    iy1 = torch.max(pred[:, 1], target[:, 1])
    ix2 = torch.min(pred[:, 2], target[:, 2])
    iy2 = torch.min(pred[:, 3], target[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    iou   = inter / (pw * ph + gw * gh - inter + 1e-7)

    # enclosing box diagonal²
    ex1 = torch.min(pred[:, 0], target[:, 0])
    ey1 = torch.min(pred[:, 1], target[:, 1])
    ex2 = torch.max(pred[:, 2], target[:, 2])
    ey2 = torch.max(pred[:, 3], target[:, 3])
    c2  = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-7

    # centre distance²
    d2 = ((pred[:, 0] + pred[:, 2]) / 2 - (target[:, 0] + target[:, 2]) / 2) ** 2 + \
         ((pred[:, 1] + pred[:, 3]) / 2 - (target[:, 1] + target[:, 3]) / 2) ** 2

    # aspect-ratio penalty
    v = (4 / math.pi ** 2) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    return 1.0 - iou + d2 / c2 + alpha * v


# Size ranges (max_side in pixels) assigned to each FPN stride level
_SCALE_RANGES = {8: (0, 96), 16: (48, 192), 32: (96, float('inf'))}


def build_targets(gt_list, feat_N3, feat_N4, feat_N5, strides, device):
    """
    FCOS-style target assignment for anchor-free detection (vectorised).

    Each GT box is assigned to the FPN level whose size range fits
    max(w, h), then to a 5×5 grid region around the GT centre.

    Args
    ----
    gt_list  : list[B] of (K, 5) float tensors  [x1,y1,x2,y2,cls]  normalised [0,1]
    feat_N3/N4/N5 : feature-map tensors (used only for shape info)
    strides  : [8, 16, 32]
    device   : torch.device

    Returns
    -------
    list[3] of (pos_mask, cls_tgt, box_tgt_xyxy)  — one per FPN level
        pos_mask     : (B, fH, fW) bool
        cls_tgt      : (B, fH, fW) long     (1 = person, 0 = background)
        box_tgt_xyxy : (B, fH, fW, 4) float  [x1,y1,x2,y2] in pixels
    """
    # Pre-compute 5×5 offset grid (RADIUS=2)
    RADIUS = 2
    r = torch.arange(-RADIUS, RADIUS + 1, device=device)
    off_j, off_i = torch.meshgrid(r, r, indexing='ij')
    off_i = off_i.reshape(-1)   # (25,)
    off_j = off_j.reshape(-1)   # (25,)

    results = []
    for feat, stride in zip([feat_N3, feat_N4, feat_N5], strides):
        B, _, fH, fW = feat.shape
        pos_mask     = torch.zeros(B, fH, fW,    dtype=torch.bool,  device=device)
        cls_tgt      = torch.zeros(B, fH, fW,    dtype=torch.long,  device=device)
        box_tgt_xyxy = torch.zeros(B, fH, fW, 4, dtype=torch.float, device=device)

        img_w = fW * stride
        img_h = fH * stride
        lo, hi = _SCALE_RANGES[stride]

        for b in range(B):
            gts = gt_list[b]          # (K, 5)  or empty
            if gts.shape[0] == 0:
                continue

            x1 = gts[:, 0] * img_w;  y1 = gts[:, 1] * img_h
            x2 = gts[:, 2] * img_w;  y2 = gts[:, 3] * img_h
            w  = x2 - x1;  h = y2 - y1

            # filter GTs by scale range (vectorised)
            max_side = torch.max(w, h)
            valid = (max_side >= lo) & (max_side < hi)
            if not valid.any():
                continue

            cx = ((x1[valid] + x2[valid]) * 0.5)
            cy = ((y1[valid] + y2[valid]) * 0.5)
            x1v, y1v, x2v, y2v = x1[valid], y1[valid], x2[valid], y2[valid]

            # centre grid cells + 5×5 offsets → (K', 25)
            gi_c = (cx / stride).long()
            gj_c = (cy / stride).long()
            gi_all = gi_c.unsqueeze(1) + off_i.unsqueeze(0)
            gj_all = gj_c.unsqueeze(1) + off_j.unsqueeze(0)

            # bounds check
            in_bounds = (gi_all >= 0) & (gi_all < fW) & (gj_all >= 0) & (gj_all < fH)
            gt_idx  = torch.arange(gi_c.shape[0], device=device).unsqueeze(1).expand_as(gi_all)[in_bounds]
            gi_flat = gi_all[in_bounds]
            gj_flat = gj_all[in_bounds]

            pos_mask[b, gj_flat, gi_flat] = True
            cls_tgt[b, gj_flat, gi_flat]  = 1
            box_tgt_xyxy[b, gj_flat, gi_flat, 0] = x1v[gt_idx]
            box_tgt_xyxy[b, gj_flat, gi_flat, 1] = y1v[gt_idx]
            box_tgt_xyxy[b, gj_flat, gi_flat, 2] = x2v[gt_idx]
            box_tgt_xyxy[b, gj_flat, gi_flat, 3] = y2v[gt_idx]

        results.append((pos_mask, cls_tgt, box_tgt_xyxy))
    return results


def compute_detection_loss(preds, gt_list, device):
    """
    Compute anchor-free detection loss across all FPN scales.

    preds   : list[3] of (reg, cls)  —  (B,4,H,W), (B,nc,H,W)
    gt_list : list[B] of (K,5) normalised GT boxes
    device  : torch.device

    Returns
    -------
    loss_box : CIoU box regression loss (mean over positive cells)
    loss_cls : Binary cross-entropy classification loss (mean over all cells × 3 scales)
    """
    N3_reg = preds[0][0];  N4_reg = preds[1][0];  N5_reg = preds[2][0]
    targets = build_targets(gt_list, N3_reg, N4_reg, N5_reg, [8, 16, 32], device)

    loss_box = torch.zeros(1, device=device)
    loss_cls = torch.zeros(1, device=device)
    n_scales = len(preds)

    for (reg, cls), stride, (pos_mask, cls_tgt, box_tgt) in \
            zip(preds, [8, 16, 32], targets):

        B, _, fH, fW = reg.shape

        # ── classification loss with Focal Loss (every cell) ──────────
        # alpha=0.25, gamma=2.0 (RetinaNet defaults)
        cls_logit = cls[:, 0]                       # (B, fH, fW)
        cls_label = pos_mask.float()
        bce = F.binary_cross_entropy_with_logits(
            cls_logit, cls_label, reduction='none')
        p_t = torch.exp(-bce)
        focal_weight = (1.0 - p_t) ** 2.0
        alpha_t = torch.where(cls_label > 0,
                              torch.tensor(0.25, device=device),
                              torch.tensor(0.75, device=device))
        # Normalize by number of positives (FCOS/RetinaNet standard)
        num_pos = pos_mask.sum().clamp(min=1).float()
        loss_cls += (alpha_t * focal_weight * bce).sum() / num_pos

        # ── box loss (positive cells only) ─────────────────────────────
        n_pos = int(pos_mask.sum().item())
        if n_pos == 0:
            continue

        pos_b, pos_j, pos_i = pos_mask.nonzero(as_tuple=True)

        # raw regression output at positive cells: (N_pos, 4)
        reg_pos = reg.permute(0, 2, 3, 1)[pos_b, pos_j, pos_i]

        # decode to pixel boxes
        pred_cx = (pos_i.float() + torch.sigmoid(reg_pos[:, 0])) * stride
        pred_cy = (pos_j.float() + torch.sigmoid(reg_pos[:, 1])) * stride
        pred_w  = torch.exp(reg_pos[:, 2].clamp(-4, 4)) * stride
        pred_h  = torch.exp(reg_pos[:, 3].clamp(-4, 4)) * stride
        pred_boxes = torch.stack([
            pred_cx - pred_w / 2, pred_cy - pred_h / 2,
            pred_cx + pred_w / 2, pred_cy + pred_h / 2,
        ], dim=1)

        # GT boxes at positive cells (already in pixels)
        gt_boxes = box_tgt[pos_b, pos_j, pos_i]    # (N_pos, 4)

        loss_box += ciou_loss(pred_boxes, gt_boxes).mean()

    loss_cls = loss_cls / n_scales      # average across scales
    return loss_box, loss_cls


# ═══════════════════════════════════════════════════════════════════════════
# Full model
# ═══════════════════════════════════════════════════════════════════════════

class DAINetYOLO(nn.Module):
    """
    DAI-Net zero-shot low-light human detector — YOLOv8n edition.

    Architecture
    ------------
    YOLOv8n backbone (3.2 M params)
        ↓ stem features (C=16, stride 2)
        ├── ReflectanceBranch  → R  (illumination-invariant)
        ↓ P3/P4/P5
    PAN-FPN neck
        ↓ N3 / N4 / N5
    Anchor-free detection heads (×3 scales)

    Zero-shot mechanism
    -------------------
    Training:  DAINetYOLO.forward(x_dark, x_light, I_dark, I_light)
                 computes mutual KL loss + reflectance maps
    Testing:   DAINetYOLO.test_forward(x_dark)
                 standard inference, output compatible with existing test.py
    """

    STRIDES = [8, 16, 32]

    def __init__(self, phase: str, num_classes: int = 1):
        super().__init__()
        self.phase       = phase
        self.num_classes = num_classes

        # ─ Detection backbone ─────────────────────────────────────────
        self.backbone = YOLOv8nBackbone()
        self.neck     = YOLOv8nNeck()
        self.heads    = nn.ModuleList([
            DetectHead( 64, num_classes),   # stride  8  — small objects
            DetectHead(128, num_classes),   # stride 16  — medium
            DetectHead(256, num_classes),   # stride 32  — large
        ])

        # ─ Zero-shot components (DAI-Net, unchanged) ──────────────────
        self.ref = ReflectanceBranch(in_channels=16)
        self.KL  = DistillKL(T=4.0)

        self.apply(self._init_weights)
        # re-initialise cls head bias AFTER general init
        for head in self.heads:
            nn.init.constant_(head.cls[-1].bias, -math.log((1 - 0.01) / 0.01))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ── Training forward ─────────────────────────────────────────────
    def forward(self, x_dark, x_light, I_dark, I_light):
        """
        x_dark   : (B, 3, H, W)  synthetically darkened images  [0, 1]
        x_light  : (B, 3, H, W)  original daytime images        [0, 1]
        I_dark   : (B, 1, H, W)  illumination map (RetinexNet, dark)
        I_light  : (B, 1, H, W)  illumination map (RetinexNet, light)

        Returns
        -------
        preds       : list[3] of (reg (B,4,H,W), cls (B,nc,H,W))
        R_maps      : [R_dark, R_light, R_dark_2, R_light_2]
        loss_mutual : scalar tensor
        """
        # ── dark image ──────────────────────────────────────────────
        stem_dark, P3_d, P4_d, P5_d = self.backbone(x_dark)
        R_dark  = self.ref(stem_dark)

        # ── light image (stem-only — P3/P4/P5 are unused) ──────────
        stem_light = self.backbone.stem(x_light)
        R_light = self.ref(stem_light)

        # ── illumination interchange (Retinex cross-recomposition) ──
        x_dark_2  = (I_light * R_dark).detach()
        x_light_2 = (I_dark  * R_light).detach()

        # stem-only for auxiliary passes — only stem features are
        # needed for ReflectanceBranch and KL alignment
        stem_d2 = self.backbone.stem(x_dark_2)
        stem_l2 = self.backbone.stem(x_light_2)
        R_dark_2  = self.ref(stem_l2)
        R_light_2 = self.ref(stem_d2)

        # ── mutual KL alignment  (identical formula to original) ────
        fd  = stem_dark.flatten(2).mean(-1)
        fl  = stem_light.flatten(2).mean(-1)
        fd2 = stem_d2.flatten(2).mean(-1)
        fl2 = stem_l2.flatten(2).mean(-1)

        loss_mutual = 0.1 * (
            self.KL(fl, fd)  + self.KL(fd, fl) +
            self.KL(fl2, fd2) + self.KL(fd2, fl2)
        )

        # ── detection (on dark image) ────────────────────────────────
        N3, N4, N5 = self.neck(P3_d, P4_d, P5_d)
        preds = [head(feat) for head, feat in zip(self.heads, [N3, N4, N5])]

        return preds, [R_dark, R_light, R_dark_2, R_light_2], loss_mutual

    # ── Test forward ─────────────────────────────────────────────────
    def test_forward(self, x):
        """
        Inference-only forward, compatible with existing test.py / evaluate_baseline.py.

        x : (1, 3, H, W)  low-light image  [0, 1]

        Returns
        -------
        output : (1, 2, TOP_K, 5)  — mimics DSFD Detect layer output
                   output[0, 1, j] = [score, x1_n, y1_n, x2_n, y2_n]  (normalised)
                   class 0 = background zeros, class 1 = person
        R      : (1, 3, H, W)  reflectance map
        """
        from torchvision.ops import nms as torchvision_nms

        TOP_K    = 750
        NMS_THR  = 0.30
        CONF_THR = 0.01

        B, _, H, W = x.shape

        # Pad to the next multiple of 32 so neck upsample/concat sizes align.
        # Boxes are always normalised by original H, W so detections are correct.
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        x_pad = F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else x

        stem, P3, P4, P5 = self.backbone(x_pad)
        R = self.ref(stem)
        N3, N4, N5 = self.neck(P3, P4, P5)

        all_boxes  = []
        all_scores = []

        for head, feat, stride in zip(self.heads, [N3, N4, N5], self.STRIDES):
            reg, cls = head(feat)
            _, _, fH, fW = reg.shape

            scores = torch.sigmoid(cls[:, 0])   # (B, fH, fW)

            # build grid
            gy, gx = torch.meshgrid(
                torch.arange(fH, dtype=reg.dtype, device=reg.device),
                torch.arange(fW, dtype=reg.dtype, device=reg.device),
                indexing='ij',
            )
            cx = (gx + torch.sigmoid(reg[0, 0])) * stride
            cy = (gy + torch.sigmoid(reg[0, 1])) * stride
            bw = torch.exp(reg[0, 2].clamp(-4, 4)) * stride
            bh = torch.exp(reg[0, 3].clamp(-4, 4)) * stride

            boxes = torch.stack([
                (cx - bw / 2) / W,
                (cy - bh / 2) / H,
                (cx + bw / 2) / W,
                (cy + bh / 2) / H,
            ], dim=-1).reshape(-1, 4).clamp(0, 1)

            all_boxes.append(boxes)
            all_scores.append(scores.reshape(-1))

        all_scores = torch.cat(all_scores)     # (total_anchors,)
        all_boxes  = torch.cat(all_boxes,  0)  # (total_anchors, 4)

        # confidence filter
        keep = all_scores > CONF_THR
        all_scores = all_scores[keep]
        all_boxes  = all_boxes[keep]

        # pack dummy result if nothing passes threshold
        output = torch.zeros(1, 2, TOP_K, 5)
        if all_scores.numel() > 0:
            keep_nms = torchvision_nms(all_boxes, all_scores, NMS_THR)
            keep_nms = keep_nms[:TOP_K]
            s = all_scores[keep_nms]
            b = all_boxes[keep_nms]
            order = s.argsort(descending=True)
            s = s[order];  b = b[order]
            n = len(s)
            output[0, 1, :n, 0]   = s.cpu()
            output[0, 1, :n, 1:5] = b.cpu()

        return output, R

    # ── Pretrained backbone loader ────────────────────────────────────
    def load_pretrained_backbone(self, pt_path: str) -> None:
        """
        Load YOLOv8n COCO backbone weights from an Ultralytics .pt file.

        Only backbone weights are mapped; neck, heads, and zero-shot
        components keep their random initialisation.

        Ultralytics YOLOv8n layer index → this model's key prefix:
            model.0.  → backbone.stem.
            model.1.  → backbone.stage1.0.
            model.2.  → backbone.stage1.1.
            model.3.  → backbone.stage2.0.
            model.4.  → backbone.stage2.1.
            model.5.  → backbone.stage3.0.
            model.6.  → backbone.stage3.1.
            model.7.  → backbone.stage4.0.
            model.8.  → backbone.stage4.1.
            model.9.  → backbone.stage4.2.
        """
        import torch as _torch
        ckpt = _torch.load(pt_path, map_location='cpu', weights_only=False)
        if hasattr(ckpt, 'state_dict'):
            src = ckpt.state_dict()
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            src = ckpt['model'].float().state_dict()
        else:
            src = ckpt

        key_map = {
            'model.0.': 'backbone.stem.',
            'model.1.': 'backbone.stage1.0.',
            'model.2.': 'backbone.stage1.1.',
            'model.3.': 'backbone.stage2.0.',
            'model.4.': 'backbone.stage2.1.',
            'model.5.': 'backbone.stage3.0.',
            'model.6.': 'backbone.stage3.1.',
            'model.7.': 'backbone.stage4.0.',
            'model.8.': 'backbone.stage4.1.',
            'model.9.': 'backbone.stage4.2.',
        }

        mapped = {}
        for k, v in src.items():
            for prefix, new_prefix in key_map.items():
                if k.startswith(prefix):
                    mapped[k.replace(prefix, new_prefix, 1)] = v
                    break

        missing, unexpected = self.load_state_dict(mapped, strict=False)
        backbone_keys = [k for k in mapped if k.startswith('backbone.')]
        print(f'[INFO] Pretrained backbone loaded from {pt_path}: '
              f'{len(backbone_keys)} backbone keys mapped, '
              f'{len(missing)} missing, {len(unexpected)} unexpected')

    # ── Checkpoint helpers ────────────────────────────────────────────
    def load_weights(self, path: str) -> int:
        ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict) and 'weight' in ckpt:
            self.load_state_dict(ckpt['weight'])
            return int(ckpt.get('epoch', 0))
        self.load_state_dict(ckpt)
        return 0

    def weights_init(self, m):
        """Called by existing train.py-style loops via net.ref.apply(net.weights_init)."""
        self._init_weights(m)


def build_net_yolo(phase: str, num_classes: int = 1) -> DAINetYOLO:
    return DAINetYOLO(phase, num_classes)
