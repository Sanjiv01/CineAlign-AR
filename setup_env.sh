#!/bin/bash
# CineAlign-AR Environment Setup for BU SCC
# Usage: source setup_env.sh

set -e

echo "[INFO] Loading SCC modules..."
module load python3/3.10.12
module load cuda/12.2

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv .venv
fi

echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

echo "[INFO] Upgrading pip..."
pip install --upgrade pip setuptools wheel packaging

echo "[INFO] Installing PyTorch with CUDA 12.1..."
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo "[INFO] Installing Flash Attention..."
pip install flash-attn==2.7.4.post1 --no-build-isolation

echo "[INFO] Installing remaining dependencies..."
pip install -r requirements.txt

echo "[INFO] Downloading model weights..."
mkdir -p models

if [ ! -d "models/Wan2.1-T2V-1.3B" ]; then
    echo "[INFO] Downloading Wan2.1-T2V-1.3B..."
    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./models/Wan2.1-T2V-1.3B
fi

if [ ! -d "models/EchoShot" ]; then
    echo "[INFO] Downloading EchoShot weights..."
    huggingface-cli download JonneyWang/EchoShot --local-dir ./models/EchoShot
fi

echo "[INFO] Verifying GPU access..."
python test_env.py

echo "[DONE] Environment setup complete!"
echo "  - Python: $(python --version)"
echo "  - Torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "  - CUDA:   $(python -c 'import torch; print(torch.cuda.is_available())')"
