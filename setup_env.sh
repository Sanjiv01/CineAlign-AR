#!/bin/bash
# CineAlign-AR Environment Setup for BU SCC
# Usage: source setup_env.sh [--skip-models] [--skip-flash-attn]
#
# This script:
#   1. Loads SCC modules (python3, cuda, ffmpeg)
#   2. Creates/activates a Python venv
#   3. Installs all pip dependencies including PyTorch + Flash Attention
#   4. Downloads pre-trained model weights (Wan2.1, EchoShot, InsightFace, YOLO)
#   5. Downloads yt-dlp for Condensed Movies fetching
#   6. Creates required data directories
#   7. Verifies the environment

set -e

# --- Parse flags ---
SKIP_MODELS=false
SKIP_FLASH_ATTN=false
for arg in "$@"; do
    case $arg in
        --skip-models) SKIP_MODELS=true ;;
        --skip-flash-attn) SKIP_FLASH_ATTN=true ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ============================================================
# 1. SCC MODULES
# ============================================================
echo "============================================================"
echo "[PHASE 2.1] Loading SCC modules..."
echo "============================================================"
module load python3/3.10.12
module load cuda/12.2
module load ffmpeg/6.0     # needed for video processing & yt-dlp

# ============================================================
# 2. PYTHON VENV
# ============================================================
echo ""
echo "============================================================"
echo "[PHASE 2.2] Setting up Python virtual environment..."
echo "============================================================"

if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "[INFO] Using Python: $(which python)"

echo "[INFO] Upgrading pip + build tools..."
pip install --upgrade pip setuptools wheel packaging 2>&1 | tail -1

# ============================================================
# 3. PIP DEPENDENCIES
# ============================================================
echo ""
echo "============================================================"
echo "[PHASE 2.3] Installing Python dependencies..."
echo "============================================================"

echo "[INFO] Installing PyTorch with CUDA 12.1..."
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3

if [ "$SKIP_FLASH_ATTN" = false ]; then
    echo "[INFO] Installing Flash Attention (this may take a few minutes)..."
    pip install flash-attn==2.7.4.post1 --no-build-isolation 2>&1 | tail -3
else
    echo "[INFO] Skipping Flash Attention (--skip-flash-attn)"
fi

echo "[INFO] Installing remaining dependencies from requirements.txt..."
pip install -r requirements.txt 2>&1 | tail -5

echo "[INFO] Installing yt-dlp for video downloading..."
pip install yt-dlp 2>&1 | tail -1

# ============================================================
# 4. MODEL WEIGHTS & PRE-TRAINED ASSETS
# ============================================================
if [ "$SKIP_MODELS" = false ]; then
    echo ""
    echo "============================================================"
    echo "[PHASE 2.4] Downloading model weights and assets..."
    echo "============================================================"

    mkdir -p models

    # --- Wan2.1-T2V-1.3B (base video generation backbone) ---
    if [ ! -d "models/Wan2.1-T2V-1.3B" ]; then
        echo "[INFO] Downloading Wan2.1-T2V-1.3B (~2.6GB)..."
        huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./models/Wan2.1-T2V-1.3B
    else
        echo "[OK] Wan2.1-T2V-1.3B already exists."
    fi

    # --- EchoShot fine-tuned weights ---
    if [ ! -d "models/EchoShot" ]; then
        echo "[INFO] Downloading EchoShot weights..."
        huggingface-cli download JonneyWang/EchoShot --local-dir ./models/EchoShot
    else
        echo "[OK] EchoShot weights already exist."
    fi

    # --- YOLOv8m (person detection for dataset pipeline) ---
    if [ ! -f "models/yolov8m.pt" ]; then
        echo "[INFO] Downloading YOLOv8m..."
        python -c "from ultralytics import YOLO; m = YOLO('yolov8m.pt')" 2>/dev/null
        # ultralytics auto-downloads to current dir; move to models/
        [ -f "yolov8m.pt" ] && mv yolov8m.pt models/yolov8m.pt
    else
        echo "[OK] YOLOv8m already exists."
    fi

    # --- InsightFace buffalo_l (face recognition) ---
    INSIGHTFACE_DIR="$HOME/.insightface/models/buffalo_l"
    if [ ! -d "$INSIGHTFACE_DIR" ]; then
        echo "[INFO] Downloading InsightFace buffalo_l model..."
        python -c "
from insightface.app import FaceAnalysis
import numpy as np
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(256, 256))
print('[OK] InsightFace buffalo_l ready.')
"
    else
        echo "[OK] InsightFace buffalo_l already exists."
    fi

else
    echo ""
    echo "[INFO] Skipping model downloads (--skip-models)"
fi

# ============================================================
# 5. DATA DIRECTORIES
# ============================================================
echo ""
echo "============================================================"
echo "[PHASE 2.5] Creating data directory structure..."
echo "============================================================"

mkdir -p dataset_in                    # source videos go here
mkdir -p dataset_out/clips             # processed clips output
mkdir -p data/condensed_movies         # Condensed Movies downloads
mkdir -p data/metadata                 # Condensed Movies metadata CSVs
mkdir -p data/video                    # training data for model
mkdir -p data/actor_clips              # actor-centric pipeline output
mkdir -p logs                          # job logs
mkdir -p outputs                       # inference outputs

echo "[OK] Data directories created."

# ============================================================
# 6. VERIFY ENVIRONMENT
# ============================================================
echo ""
echo "============================================================"
echo "[PHASE 2.6] Verifying environment..."
echo "============================================================"

python test_env.py

echo ""
echo "============================================================"
echo "[DONE] CineAlign-AR environment setup complete!"
echo "============================================================"
echo "  Python  : $(python --version 2>&1)"
echo "  Torch   : $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA    : $(python -c 'import torch; print(torch.version.cuda)')"
echo "  GPUs    : $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "  FlashAttn: $(python -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null || echo 'not installed')"
echo "  yt-dlp  : $(yt-dlp --version 2>/dev/null || echo 'not installed')"
echo ""
echo "Next steps:"
echo "  1. Place source videos in dataset_in/ OR download Condensed Movies:"
echo "     bash scripts/download_condensed_movies.sh"
echo "  2. Process videos:  qsub scripts/process_dataset.sh"
echo "  3. Train model:     qsub scripts/train_lora.sh"
