#!/bin/bash

#$ -P cs523aw
#$ -l h_rt=01:00:00
#$ -N cinealign_inference
#$ -j y

#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0

module load python3/3.10.12
module load cuda/12.2

cd /projectnb/cs523aw/students/$USER/Identity-centric-Autoregressive-Multi-Shot-Video-Generation
source .venv/bin/activate

python test_env.py

cd cinealign_ar
bash ./generate.sh
