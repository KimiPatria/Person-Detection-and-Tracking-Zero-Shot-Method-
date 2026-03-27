#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# DAI-Net YOLOv8n — vast.ai instance setup script
# Target: RTX 5090 (Blackwell / sm_120), CUDA 12.8, Ubuntu
#
# Usage (run once after SSH-ing into the instance):
#   chmod +x setup.sh && ./setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e   # exit on any error

echo "========================================================"
echo " DAI-Net YOLOv8n — environment setup"
echo "========================================================"

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/5] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    git wget curl unzip \
    libgl1 libglib2.0-0 \
    python3-pip

# ── 2. PyTorch (CUDA 12.8 wheels — required for RTX 5090 / Blackwell) ────────
echo "[2/5] Installing PyTorch with CUDA 12.8 support..."

# Try stable release first (PyTorch 2.6+ supports sm_120 / Blackwell)
pip install --quiet \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Verify that the GPU is detected and sm_120 (Blackwell) is supported
python3 - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available!"
name = torch.cuda.get_device_name(0)
cap  = torch.cuda.get_device_capability(0)
print(f"  GPU     : {name}")
print(f"  Compute : {cap[0]}.{cap[1]}")
print(f"  PyTorch : {torch.__version__}")
print(f"  CUDA    : {torch.version.cuda}")
print(f"  bf16    : {torch.cuda.is_bf16_supported()}")
assert cap[0] >= 9, f"Expected Blackwell (sm_12x) or Hopper (sm_9x), got sm_{cap[0]}{cap[1]}"
print("  [OK] GPU check passed")
EOF

# ── 3. Python dependencies ────────────────────────────────────────────────────
echo "[3/5] Installing Python dependencies..."
pip install --quiet -r requirements.txt

# ── 4. Verify full import chain ───────────────────────────────────────────────
echo "[4/5] Verifying imports..."
python3 - <<'EOF'
import torch, torchvision, torchmetrics, cv2, PIL, easydict, scipy, matplotlib
print(f"  torch        {torch.__version__}")
print(f"  torchvision  {torchvision.__version__}")
print(f"  torchmetrics {torchmetrics.__version__}")
print(f"  opencv       {cv2.__version__}")
print("  [OK] All imports passed")
EOF

# ── 5. Dataset reminder ───────────────────────────────────────────────────────
echo "[5/5] Dataset setup..."
echo ""
echo "  Upload your Roboflow dataset to:  dataset/roboflow/"
echo "  Expected layout:"
echo "    dataset/roboflow/train/   (images + XML annotations)"
echo "    dataset/roboflow/valid/   (images + XML annotations)"
echo ""
echo "  Fastest upload options:"
echo "    rsync -avz --progress  ./dataset/  user@<vast-ip>:/workspace/DAI-Net/dataset/"
echo "    wget '<signed-url>'  -O dataset.zip && unzip dataset.zip -d dataset/"
echo ""

# ── Optional: upload RetinexNet pretrained weights ────────────────────────────
mkdir -p weights
echo "  If you have decomp.pth (RetinexNet pretrained), copy it to:  weights/decomp.pth"
echo ""

echo "========================================================"
echo " Setup complete. Start training with:"
echo "   python train_yolo.py"
echo ""
echo " Recommended flags for RTX 5090:"
echo "   python train_yolo.py --batch_size 16 --amp true --num_workers 4"
echo ""
echo " Resume from checkpoint:"
echo "   python train_yolo.py --resume weights/yolo_dark/yolo_checkpoint.pth"
echo "========================================================"