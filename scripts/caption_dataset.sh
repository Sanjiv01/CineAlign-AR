#!/bin/bash
# ============================================================
# SCC Job: VLM Captioning for processed clips
# ============================================================
# Generates detailed captions using LLaVA-Next-7B for each
# video clip in train.json. Requires GPU for VLM inference.
#
# Prerequisites: dataset_out/train.json and clips must exist
# Output: Updated train.json with VLM captions (replaces heuristic ones)
#
# Usage:
#   qsub scripts/caption_dataset.sh
#   qsub -v BATCH_START=0,BATCH_END=500 scripts/caption_dataset.sh  # partial
# ============================================================

#$ -P cs523aw
#$ -l h_rt=24:00:00
#$ -N vlm_captioning
#$ -j y
#$ -o logs/caption_dataset_$JOB_ID.log

#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l mem_per_core=16G

PROJECT_ROOT="/projectnb/cs523aw/students/$USER/CineAlign-AR"
cd "$PROJECT_ROOT"

module load python3/3.10.12
module load cuda/12.2

source .venv/bin/activate

echo "============================================================"
echo "VLM Captioning Pipeline"
echo "  Date:    $(date)"
echo "  Host:    $(hostname)"
echo "  GPU:     $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "============================================================"

# Build command with optional batch range
CMD="python dataset_generation/processing/vlm_captioner.py"
CMD="$CMD --train_json dataset_out/train.json"
CMD="$CMD --data_root dataset_out"

if [ -n "$BATCH_START" ]; then
    CMD="$CMD --batch_start $BATCH_START"
fi
if [ -n "$BATCH_END" ]; then
    CMD="$CMD --batch_end $BATCH_END"
fi

echo "[CMD] $CMD"
echo ""

$CMD

# Stage to training data directory
echo ""
echo "[INFO] Staging captioned data to training directory..."
mkdir -p data/video/clips

if [ -d "dataset_out/clips" ]; then
    for clip in dataset_out/clips/*.mp4; do
        basename=$(basename "$clip")
        dst="data/video/clips/$basename"
        if [ ! -e "$dst" ]; then
            ln -s "$(realpath "$clip")" "$dst" 2>/dev/null || cp "$clip" "$dst"
        fi
    done
fi

cp dataset_out/train.json data/video/train.json 2>/dev/null || true

echo ""
echo "[DONE] Captioning complete at $(date)"
echo "Next: qsub scripts/train_lora.sh"
