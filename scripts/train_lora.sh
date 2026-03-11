#!/bin/bash

#$ -P cs523aw
#$ -l h_rt=96:00:00
#$ -N cinealign_lora_train
#$ -j y

#$ -pe omp 8
#$ -l gpus=2
#$ -l gpu_type=H200

module load python3/3.10.12
module load cuda/12.2

cd /projectnb/cs523aw/students/$USER/Identity-centric-Autoregressive-Multi-Shot-Video-Generation
source .venv/bin/activate

python test_env.py

cd cinealign_ar
torchrun --nproc_per_node=2 train_lora.py
