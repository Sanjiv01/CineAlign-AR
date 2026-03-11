#!/bin/bash

#$ -P cs523aw
#$ -l h_rt=48:00:00
#$ -N dataset_processing
#$ -j y

#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.0

module load python3/3.10.12
module load cuda/12.2

cd /projectnb/cs523aw/students/$USER/Identity-centric-Autoregressive-Multi-Shot-Video-Generation
source .venv/bin/activate

echo "=== Processing Traditional Dataset ==="
cd dataset_generation/processing
python data_process_v2.py

echo "=== Running VLM Captioner ==="
python vlm_captioner.py \
    --train_json ../../dataset_out/train.json \
    --data_root ../../dataset_out/clips

echo "=== DONE ==="
