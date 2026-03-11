#!/bin/bash
# ============================================================
# SCC Job: Process videos into training clips
# ============================================================
# Runs the full processing pipeline:
#   1. Scene detection (PySceneDetect)
#   2. Person detection (YOLOv8m)
#   3. Face embedding + clustering (InsightFace buffalo_l)
#   4. Multi-shot clip export (832x480, 125 frames, 2-4 shots)
#
# Prerequisites: Videos must be in dataset_in/
# Output: dataset_out/clips/*.mp4 + dataset_out/train.json
#
# Usage: qsub scripts/process_dataset.sh
# ============================================================

#$ -P cs523aw
#$ -l h_rt=48:00:00
#$ -N dataset_processing
#$ -j y
#$ -o logs/process_dataset_$JOB_ID.log

#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l mem_per_core=8G

PROJECT_ROOT="/projectnb/cs523aw/students/$USER/Identity-centric-Autoregressive-Multi-Shot-Video-Generation"
cd "$PROJECT_ROOT"

module load python3/3.10.12
module load cuda/12.2
module load ffmpeg/6.0

source .venv/bin/activate

echo "============================================================"
echo "Dataset Processing Pipeline"
echo "  Date:    $(date)"
echo "  Host:    $(hostname)"
echo "  GPU:     $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "  Videos:  $(find dataset_in -name '*.mp4' 2>/dev/null | wc -l)"
echo "============================================================"

# Create output directories
mkdir -p dataset_out/clips
mkdir -p logs

# Run video processing
echo ""
echo "[STEP 1/2] Processing videos -> clips..."
echo "============================================================"
cd dataset_generation/processing
python data_process_v2.py
cd "$PROJECT_ROOT"

# Report results
CLIP_COUNT=$(find dataset_out/clips -name '*.mp4' 2>/dev/null | wc -l)
ENTRY_COUNT=$(python -c "import json; d=json.load(open('dataset_out/train.json')); print(len(d))" 2>/dev/null || echo 0)

echo ""
echo "============================================================"
echo "Processing Results:"
echo "  Clips exported:       $CLIP_COUNT"
echo "  Training entries:     $ENTRY_COUNT"
echo "============================================================"

echo ""
echo "[DONE] Processing complete at $(date)"
echo "Next: qsub scripts/caption_dataset.sh"
