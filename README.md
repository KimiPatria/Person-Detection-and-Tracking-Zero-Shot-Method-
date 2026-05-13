# DAI-Net: Zero-Shot Day-Night Domain Adaptation for Person Detection

A lightweight person detector that generalizes to nighttime images **without any nighttime training data**, built on top of [Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation](https://arxiv.org/abs/2312.01220) (CVPR 2024).

This project replaces the original heavy VGG16-DSFD backbone (138M params) with a **YOLOv8n backbone** (~3.7M params total), making it practical for real-world deployment while preserving the zero-shot domain adaptation capability.

> **Personal project / undergraduate thesis** — adapted from Du et al. (CVPR 2024).

---

## How It Works

The model solves a fundamental problem: object detectors trained on daytime images fail at night because the visual distributions differ dramatically. Rather than collecting paired day/night data, DAI-Net uses **Retinex theory** to decompose images into:

- **Reflectance (R)** — the illumination-invariant surface properties (what we care about)
- **Illumination (I)** — the lighting conditions (what changes between day and night)

By training the detector to operate on reflectance features instead of raw pixel features, and by aligning the day/night feature distributions via **KL divergence**, the model learns representations that transfer to unseen night images at inference time.

**Training-time pipeline:**

1. **DarkISP synthesis** — Realistic low-light degradation simulates nighttime images from daytime data (RAW noise model, sensor quantization, ISP pipeline)
2. **Retinex decomposition** — A frozen pretrained RetinexNet decomposes both day and synthetic night images into R and I maps
3. **Reflectance decoding** — A lightweight branch on top of the backbone predicts reflectance from features; supervised by Retinex outputs
4. **KL alignment** — The stem features from day and night inputs are aligned in distribution, encouraging illumination-invariant representations
5. **Coherence losses** — Enforce Retinex consistency: smooth illumination, reconstructible images, equal reflectance across lighting conditions

At **inference**, only the backbone + neck + detection heads run — no Retinex decomposition, no DarkISP. The model simply handles dark images natively.

---

## Architecture

```
Input (640×640)
    │
    ▼
YOLOv8n Backbone (~3.2M params)
├── stem  →  (16, 320×320)
├── stage1 → (32, 160×160)
├── stage2 → P3 (64, 80×80)
├── stage3 → P4 (128, 40×40)
└── stage4 → P5 (256, 20×20)  [SPPF]
    │
    ├──[training only]──► ReflectanceBranch → R map (3, 640×640)
    │
    ▼
Path-Aggregation FPN Neck
    N3 (64, 80×80), N4 (128, 40×40), N5 (256, 20×20)
    │
    ▼
Anchor-Free Detection Heads (×3 scales)
    stride 8/16/32 → reg (4) + cls (1) per grid cell
    │
    ▼
Post-processing: sigmoid decode → NMS → Top-K
```

**Zero-shot components (training only):**

```
frozen RetinexNet
    ├── img_dark → R_dark_gt, I_dark
    └── img_light → R_light_gt, I_light

ReflectanceBranch (stem_dark) → R_dark_pred
ReflectanceBranch (stem_light) → R_light_pred

Illumination interchange:
    x_dark_2 = I_light × R_dark   (swap illumination)
    x_light_2 = I_dark × R_light

KL alignment:
    L_mutual = KL(day_stem ∥ night_stem) + KL(night_stem ∥ day_stem)  [×4 combinations]
```

### Available Model Variants

| Key | Backbone | Params | Notes |
|-----|----------|--------|-------|
| `yolo_dark` | YOLOv8n + Retinex + KL | ~3.7M | This project's main contribution |
| `dark` | VGG16 DSFD + Retinex | ~138M | Original DAI-Net |
| `vgg` | VGG16 DSFD | ~138M | Baseline (no adaptation) |
| `resnet50/101/152` | ResNet DSFD | ~60–100M | Alternative baselines |

Models are instantiated via `models/factory.py:build_net()`.

---

## Loss Functions

| Component | Formula | Weight |
|-----------|---------|--------|
| CIoU box regression | 1 - IoU + d²/c² + α·v | λ_box = 5.0 |
| Focal classification | BCE with α=0.25, γ=2.0 | λ_cls = 1.0 |
| Reflectance L1 + SSIM | ‖R_pred − R_gt‖₁ + (1 − SSIM) | 0.3 |
| Retinex coherence | Smooth illum + recon + equal-R | 0.1 |
| KL distribution alignment | KL(day ‖ night) + KL(night ‖ day) | 0.5 |

---

## Project Structure

```
DAI-Net/
├── models/
│   ├── factory.py            # build_net() entry point
│   ├── DAINet_yolov8.py      # YOLOv8n + Retinex + KL (main model)
│   ├── yolov8_modules.py     # Conv, C2f, SPPF, Bottleneck
│   ├── enhancer.py           # RetinexNet (frozen during training)
│   ├── DAINet.py             # Original DSFD + Retinex (VGG)
│   ├── DSFD_vgg.py           # DSFD VGG backbone
│   └── DSFD_resnet.py        # DSFD ResNet backbone
├── layers/
│   ├── bbox_utils.py         # IoU, NMS, coordinate transforms
│   ├── modules/
│   │   ├── enhance_loss.py   # Retinex coherence losses
│   │   └── multibox_loss.py  # SSD-style loss (DSFD variants)
│   └── functions/
│       ├── detection.py      # Raw predictions → detections
│       └── prior_box.py      # Anchor generation
├── data/
│   ├── config.py             # Dataset paths, thresholds, loss weights
│   ├── people_dataset.py     # Roboflow VOC-format loader
│   └── widerface.py          # WIDER Face loader (legacy)
├── utils/
│   ├── DarkISP.py            # Low-light image degradation simulator
│   └── augmentations.py      # Training augmentation pipeline
├── train_yolo.py             # Main training script (YOLOv8n variant)
├── train.py                  # Training script (DSFD variants)
├── test.py                   # Image inference
├── test_video.py             # Video inference
├── evaluate_baseline.py      # Comprehensive evaluation + plots
├── evaluate_ablation.py      # Ablation comparison
├── evaluate_exdark.py        # ExDARK benchmark
├── evaluate_per_category.py  # Per-category breakdown
├── evaluate_thesis_metrics.py
├── visualize_predictions.py  # Qualitative visualization
├── plot_training_curves.py   # Training loss curves
├── export_onnx.py            # ONNX export
├── setup.sh                  # vast.ai environment setup
├── requirements.txt
└── weights/
    ├── decomp.pth            # Pretrained RetinexNet
    ├── vgg16_reducedfc.pth   # VGG16 base weights
    └── yolo_dark/            # YOLOv8n checkpoints
```

---

## Installation

### Local (conda)

```bash
git clone https://github.com/kimipatria220904/DAI-Net
cd DAI-Net

conda create -y -n dainet python=3.10
conda activate dainet

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### vast.ai (RTX 5090 / Blackwell)

```bash
chmod +x setup.sh && ./setup.sh
```

The script installs system packages, PyTorch with CUDA 12.8, Python dependencies, and verifies Blackwell bf16 support.

---

## Dataset

The project uses a **Roboflow VOC-format** person detection dataset. Expected structure:

```
dataset/
└── roboflow/
    ├── train/
    │   ├── images/        # .jpg / .png
    │   └── annotations/   # Pascal VOC .xml
    ├── valid/
    │   ├── images/
    │   └── annotations/
    └── test/
        ├── images/
        └── annotations/
```

Update paths in `data/config.py` if your layout differs.

---

## Pretrained Weights

| File | Description | Download |
|------|-------------|----------|
| `weights/decomp.pth` | Pretrained RetinexNet (required for training) | [Google Drive](https://drive.google.com/file/d/1MaRK-VZmjBvkm79E1G77vFccb_9GWrfG/view?usp=drive_link) |
| `weights/vgg16_reducedfc.pth` | VGG16 base (required for DSFD variants) | [Google Drive](https://drive.google.com/file/d/1whV71K42YYduOPjTTljBL8CB-Qs4Np6U/view?usp=drive_link) |
| `weights/yolo_dark/yolo_checkpoint.pth` | YOLOv8n training checkpoint | — |

---

## Training

### YOLOv8n (recommended)

```bash
python train_yolo.py --batch_size 32 --num_workers 8 --amp true
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 32 | 32 for 32GB VRAM, 16 for 24GB |
| `--num_workers` | 8 | DataLoader workers |
| `--lr` | 1e-3 | Initial learning rate |
| `--weight_decay` | 5e-4 | AdamW weight decay |
| `--lambda_box` | 5.0 | CIoU loss weight |
| `--lambda_cls` | 1.0 | Focal loss weight |
| `--amp` | true | Mixed precision (bf16 on Blackwell, fp16 fallback) |
| `--warmup_epochs` | 3 | Linear LR warmup |
| `--val_interval` | 3 | Validate every N epochs |
| `--resume` | — | Resume from checkpoint path |
| `--ablation` | full | `baseline` / `reflectance` / `ref_kl` / `full` |

Training runs for **60 epochs** with AdamW + cosine annealing LR + EMA (decay=0.9999). Checkpoints and `loss_history.csv` are saved to `weights/yolo_dark/`.

**Ablation modes:**

| Mode | Components |
|------|-----------|
| `baseline` | Detector only (no Retinex, no KL) |
| `reflectance` | Detector + reflectance decoding |
| `ref_kl` | Detector + reflectance + KL alignment |
| `full` | All components |

### Original DSFD (VGG/ResNet variants)

```bash
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS train.py
```

---

## Evaluation

### Comprehensive metrics

```bash
python evaluate_baseline.py
```

Outputs `result/baseline_metrics.png` (precision-recall curve, F1 curve, FPS distribution, confidence distribution) and `result/baseline_results.txt`.

Configure paths and thresholds at the top of the script:

| Variable | Description |
|----------|-------------|
| `IMAGES_DIR` / `ANNOTATIONS_DIR` | Test set location |
| `CONF_THRESH` | Detection confidence threshold |
| `IOU_THRESH` | IoU threshold for TP/FP |
| `USE_MULTI_SCALE` | Multi-scale testing toggle |

### Ablation comparison

```bash
python evaluate_ablation.py
```

### ExDARK benchmark

```bash
python evaluate_exdark.py
```

### Qualitative visualization

```bash
python visualize_predictions.py
```

Produces side-by-side panels (predicted / ground truth / reference) in `vis_output/`.

### Training curves

```bash
python plot_training_curves.py
```

Reads `loss_history.csv` and plots component losses over epochs.

---

## Inference

### Image set

```bash
python test.py
```

Runs inference with letterbox preprocessing (640×640), optional multi-scale testing, and horizontal flip TTA.

### Video

```bash
python test_video.py
```

Processes a video file with bounding box overlay and FPS counter. Configure `VIDEO_INPUT`, `VIDEO_OUTPUT`, and `CONFIDENCE_THRESHOLD` inside the script.

---

## ONNX Export

```bash
python export_onnx.py
```

Exports the DSFD model to `weights/dsfd_optimized.onnx` (opset 11, dynamic axes for variable input resolution).

---

## Citation

If you use this work, please cite the original paper:

```bibtex
@inproceedings{du2024boosting,
  title     = {Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation},
  author    = {Du, Zhipeng and Shi, Miaojing and Deng, Jiankang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {12666--12676},
  year      = {2024}
}
```

---

## Acknowledgements

Built upon:
- [DSFD.pytorch](https://github.com/yxlijun/DSFD.pytorch)
- [RetinexNet_PyTorch](https://github.com/aasharma90/RetinexNet_PyTorch)
- [MAET](https://github.com/cuiziteng/ICCV_MAET)
- [HLA-Face](https://github.com/daooshee/HLA-Face-Code)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
