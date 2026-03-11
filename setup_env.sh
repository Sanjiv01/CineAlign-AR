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

module load miniconda
module load academic-ml/spring-2026
conda activate spring-2026-pyt
