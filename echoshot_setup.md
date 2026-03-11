# EchoShot Setup

## Python venv
### Option 1: setup from scratch
```sh
module load python3/3.10.12 
module load cuda/12.2
pip install packaging setuptools wheel
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r echoshot_requirements.txt
```

### Option 2: use my venv
```sh
source /projectnb/cs523aw/project/multishot_video_gen/.venv/bin/activate
```

## EchoShot Setup

```sh
git clone https://github.com/D2I-ai/EchoShot
cd EchoShot

# get model weights
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./models/Wan2.1-T2V-1.3B
huggingface-cli download JonneyWang/EchoShot --local-dir ./models/EchoShot
```
