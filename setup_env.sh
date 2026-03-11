#!/bin/bash
# ============================================================
# CineAlign-AR: SCC Environment Setup
# ============================================================
# Sets up the full environment on BU SCC.
#
# Usage:
#   source setup_env.sh                    # Full setup (deps + models)
#   source setup_env.sh --skip-models      # Skip model weight downloads
#   source setup_env.sh --deps-only        # Only install pip deps
#   source setup_env.sh --activate-only    # Just activate the env
# ============================================================

set -e

# --- Parse args ---
SKIP_MODELS=false
DEPS_ONLY=false
ACTIVATE_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-models)   SKIP_MODELS=true ;;
        --deps-only)     DEPS_ONLY=true ;;
        --activate-only) ACTIVATE_ONLY=true ;;
    esac
done

# --- Load SCC modules ---
echo "[ENV] Loading SCC modules..."
module load miniconda
module load academic-ml/spring-2026
conda activate spring-2026-pyt

echo "[ENV] Python: $(which python)"
echo "[ENV] Conda env: $CONDA_DEFAULT_ENV"

if [ "$ACTIVATE_ONLY" = true ]; then
    echo "[ENV] Environment activated. Done."
    return 0 2>/dev/null || exit 0
fi

# --- Project root ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"
echo "[ENV] Project root: $PROJECT_ROOT"

# --- Install pip dependencies ---
echo ""
echo "[ENV] Installing pip dependencies..."
pip install -r requirements.txt 2>&1 | tail -5

# Check for flash-attn (may need special install)
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "[ENV] Installing flash-attn (this takes a few minutes)..."
    pip install flash-attn --no-build-isolation 2>&1 | tail -3
fi

if [ "$DEPS_ONLY" = true ]; then
    echo "[ENV] Dependencies installed. Done."
    return 0 2>/dev/null || exit 0
fi

# --- Create data directories ---
echo ""
echo "[ENV] Creating data directories..."
mkdir -p data/condensed_movies/videos
mkdir -p data/metadata
mkdir -p data/video/clips
mkdir -p dataset_in
mkdir -p dataset_out/clips
mkdir -p models
mkdir -p runs/cinealign_ar
mkdir -p outputs
mkdir -p logs

# --- Download model weights ---
if [ "$SKIP_MODELS" = true ]; then
    echo "[ENV] Skipping model downloads (--skip-models)"
else
    echo ""
    echo "[ENV] Downloading model weights..."
    echo "  This may take a while on first run (~10GB total)"
    echo ""

    # Wan2.1-T2V-1.3B backbone
    if [ ! -d "models/Wan2.1-T2V-1.3B" ]; then
        echo "[ENV] Downloading Wan2.1-T2V-1.3B..."
        huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
            --local-dir models/Wan2.1-T2V-1.3B
    else
        echo "[ENV] Wan2.1-T2V-1.3B already exists, skipping."
    fi

    # EchoShot pretrained weights
    if [ ! -d "models/EchoShot" ]; then
        echo "[ENV] Downloading EchoShot weights..."
        huggingface-cli download JonneyWang/EchoShot \
            --local-dir models/EchoShot
    else
        echo "[ENV] EchoShot weights already exist, skipping."
    fi

    # InsightFace buffalo_l (face recognition)
    INSIGHTFACE_DIR="$HOME/.insightface/models/buffalo_l"
    if [ ! -d "$INSIGHTFACE_DIR" ]; then
        echo "[ENV] Downloading InsightFace buffalo_l..."
        python -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(160, 160))
print('InsightFace buffalo_l downloaded successfully')
"
    else
        echo "[ENV] InsightFace buffalo_l already exists, skipping."
    fi

    # YOLOv8m (person detection)
    if [ ! -f "models/yolov8m.pt" ]; then
        echo "[ENV] Downloading YOLOv8m..."
        python -c "
from ultralytics import YOLO
model = YOLO('yolov8m.pt')
print('YOLOv8m downloaded successfully')
" && mv yolov8m.pt models/yolov8m.pt 2>/dev/null || true
    else
        echo "[ENV] YOLOv8m already exists, skipping."
    fi
fi

# --- Verify environment ---
echo ""
echo "[ENV] Running environment verification..."
python test_env.py

echo ""
echo "============================================================"
echo "[ENV] Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Download videos locally:  python dataset_generation/data_prep/download.py --batch"
echo "  2. Transfer to SCC:          scp -r data/condensed_movies/videos sanjiv@scc1.bu.edu:$(pwd)/data/condensed_movies/"
echo "  3. Process dataset:          qsub scripts/process_dataset.sh"
echo "  4. Caption clips:            qsub scripts/caption_dataset.sh"
echo "  5. Train LoRA:               qsub scripts/train_lora.sh"
