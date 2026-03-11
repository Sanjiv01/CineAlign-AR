#!/bin/bash
# ============================================================
# SCC Job: AR Inference for CineAlign-AR
# ============================================================
# Generates multi-shot videos using the trained autoregressive
# model with sliding-window inference.
#
# Usage:
#   qsub scripts/inference.sh
# ============================================================

#$ -P cs523aw
#$ -l h_rt=02:00:00
#$ -N cinealign_inference
#$ -j y
#$ -o logs/inference_$JOB_ID.log

#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l mem_per_core=16G

PROJECT_ROOT="/projectnb/cs523aw/students/$USER/Identity-centric-Autoregressive-Multi-Shot-Video-Generation"
cd "$PROJECT_ROOT"

module load python3/3.10.12
module load cuda/12.2

source .venv/bin/activate

echo "============================================================"
echo "CineAlign-AR Inference"
echo "  Date:    $(date)"
echo "  Host:    $(hostname)"
echo "  GPU:     $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "============================================================"

# Verify environment
python test_env.py

# Create output directory
mkdir -p outputs

# Run generation
cd cinealign_ar
bash ./generate.sh

echo ""
echo "[DONE] Inference complete at $(date)"
echo "Outputs saved to: $PROJECT_ROOT/outputs/"
