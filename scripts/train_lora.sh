#!/bin/bash
# ============================================================
# SCC Job: LoRA Fine-Tuning for CineAlign-AR
# ============================================================
# Trains LoRA adapters on the Wan2.1-T2V backbone with EchoShot
# multi-shot architecture. Uses FSDP with 2 H200 GPUs.
#
# LoRA config: rank=8, alpha=16, targets=q,k,v,o
# Training data: data/video/train.json + data/video/clips/
#
# Usage:
#   qsub scripts/train_lora.sh
#   qsub -v RESUME_CKPT=runs/step_5000.pt scripts/train_lora.sh
# ============================================================

#$ -P cs585
#$ -l h_rt=96:00:00
#$ -N cinealign_lora_train
#$ -j y
#$ -o logs/train_lora_$JOB_ID.log

#$ -pe omp 8
#$ -l gpus=2
#$ -l gpu_type=H200
#$ -l mem_per_core=16G

PROJECT_ROOT="/projectnb/cs585/students/sanjiv/CineAlign-AR"
cd "$PROJECT_ROOT"

module load miniconda
module load academic-ml/spring-2026
conda activate spring-2026-pyt

echo "============================================================"
echo "CineAlign-AR LoRA Training"
echo "  Date:      $(date)"
echo "  Host:      $(hostname)"
echo "  Job ID:    $JOB_ID"
echo "  GPUs:      $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "  GPU names: $(python -c 'import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])')"
echo "============================================================"

# Verify environment
python test_env.py

# Verify training data exists
TRAIN_JSON="data/video/train.json"
if [ ! -f "$TRAIN_JSON" ]; then
    echo "[ERROR] Training data not found: $TRAIN_JSON"
    echo "Run the dataset pipeline first: qsub scripts/process_dataset.sh"
    exit 1
fi
ENTRY_COUNT=$(python -c "import json; d=json.load(open('$TRAIN_JSON')); print(len(d))")
echo "[INFO] Training entries: $ENTRY_COUNT"
echo ""

# Create checkpoint directory
mkdir -p runs/cinealign_ar
mkdir -p logs

# Run training
cd cinealign_ar

echo "[INFO] Starting LoRA training with torchrun..."
torchrun --nproc_per_node=2 \
    --master_port=29500 \
    train_lora.py

echo ""
echo "[DONE] Training complete at $(date)"
