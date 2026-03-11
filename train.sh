#!/bin/bash


#$ -P cs523aw         # Specify the SCC project name you want to use
#$ -l h_rt=00:30:00   # Specify the hard time limit for the job
#$ -N logs/test_lora  # Give job a name
#$ -j y               # Merge the error and output streams into a single file

#$ -pe omp 8
#$ -l gpus=2

# -l gpu_c=8.0
#$ -l gpu_type=H200

# load module
module load python3/3.10.12 
module load cuda/12.2

# switch directory
# cd /projectnb/cs523aw/project/multishot_video_gen
cd /projectnb/cs523aw/students/erioe/Style-Aligned-Multi-Shot-Video-Generation
# load .venv
source .venv/bin/activate

python test_env.py

cd EchoShot
bash ./train.sh
