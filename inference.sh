#!/bin/bash


#$ -P cs523aw         # Specify the SCC project name you want to use
#$ -l h_rt=00:30:00   # Specify the hard time limit for the job
#$ -N test_inference  # Give job a name
#$ -j y               # Merge the error and output streams into a single file

# Request 4 CPUs
#$ -pe omp 4
# Request 1 GPU 
#$ -l gpus=1
# Specify the minimum GPU compute capability. 
#$ -l gpu_c=8.0

# load module
module load python3/3.10.12 
module load cuda/12.2

# switch directory
cd /projectnb/cs523aw/project/multishot_video_gen

# load .venv
source .venv/bin/activate

python test_env.py

cd EchoShot
bash ./generate.sh
