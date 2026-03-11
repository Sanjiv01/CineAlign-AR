#!/bin/bash
# ============================================================
# CineAlign-AR: Full Traditional Dataset Pipeline
# ============================================================
#
# End-to-end pipeline:
#   1. Download Condensed Movies from YouTube (or use existing videos)
#   2. Prepare videos for processing (symlink into dataset_in/)
#   3. Process videos: scene detect -> YOLO -> face cluster -> clip export
#   4. VLM captioning on exported clips
#   5. Copy final train.json + clips to training data directory
#
# Usage:
#   bash scripts/run_full_pipeline.sh                    # Full pipeline
#   bash scripts/run_full_pipeline.sh --skip-download    # Skip YouTube download
#   bash scripts/run_full_pipeline.sh --step process     # Run only processing
#   bash scripts/run_full_pipeline.sh --step caption     # Run only captioning
#
# For SCC batch submission, use the individual job scripts instead:
#   qsub scripts/download_condensed_movies.sh
#   qsub scripts/process_dataset.sh
#   qsub scripts/caption_dataset.sh

set -e

# --- Parse arguments ---
SKIP_DOWNLOAD=false
STEP="all"  # all | download | prepare | process | caption | stage
MAX_VIDEOS=-1

for arg in "$@"; do
    case $arg in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --step) shift; STEP="$1" ;;
        --max-videos=*) MAX_VIDEOS="${arg#*=}" ;;
    esac
    shift 2>/dev/null || true
done

# --- Paths ---
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

CM_DATA_DIR="$PROJECT_ROOT/data/condensed_movies"
METADATA_DIR="$PROJECT_ROOT/data/metadata"
DATASET_IN="$PROJECT_ROOT/dataset_in"
DATASET_OUT="$PROJECT_ROOT/dataset_out"
TRAIN_DATA_DIR="$PROJECT_ROOT/data/video"

echo "============================================================"
echo "CineAlign-AR: Full Traditional Dataset Pipeline"
echo "============================================================"
echo "  Project root:  $PROJECT_ROOT"
echo "  Step:          $STEP"
echo "  Skip download: $SKIP_DOWNLOAD"
echo "  Max videos:    $MAX_VIDEOS"
echo "============================================================"
echo ""

# Activate environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# ============================================================
# STEP 1: Download Condensed Movies
# ============================================================
if [ "$STEP" = "all" ] || [ "$STEP" = "download" ]; then
    if [ "$SKIP_DOWNLOAD" = true ]; then
        echo "[STEP 1] SKIPPED: YouTube download (--skip-download)"
    else
        echo "============================================================"
        echo "[STEP 1] Downloading Condensed Movies..."
        echo "============================================================"

        python dataset_generation/data_prep/download.py \
            --data_dir "$CM_DATA_DIR" \
            --metadata_dir "$METADATA_DIR" \
            --batch
    fi
    echo ""
fi

# ============================================================
# STEP 2: Prepare videos for processing
# ============================================================
if [ "$STEP" = "all" ] || [ "$STEP" = "prepare" ]; then
    echo "============================================================"
    echo "[STEP 2] Preparing videos for processing pipeline..."
    echo "============================================================"

    PREPARE_ARGS="--src_dir $CM_DATA_DIR/videos --dst_dir $DATASET_IN --min_duration 30"
    if [ "$MAX_VIDEOS" -gt 0 ] 2>/dev/null; then
        PREPARE_ARGS="$PREPARE_ARGS --max_videos $MAX_VIDEOS"
    fi

    python dataset_generation/processing/prepare_input.py $PREPARE_ARGS

    echo "Videos in dataset_in/: $(find $DATASET_IN -name '*.mp4' | wc -l)"
    echo ""
fi

# ============================================================
# STEP 3: Process videos -> training clips
# ============================================================
if [ "$STEP" = "all" ] || [ "$STEP" = "process" ]; then
    echo "============================================================"
    echo "[STEP 3] Processing videos -> scene detect -> face cluster -> clip export"
    echo "============================================================"

    cd dataset_generation/processing
    python data_process_v2.py
    cd "$PROJECT_ROOT"

    echo ""
    echo "Processing complete."
    echo "  Clips: $(find $DATASET_OUT/clips -name '*.mp4' 2>/dev/null | wc -l)"
    echo "  train.json entries: $(python -c "import json; d=json.load(open('$DATASET_OUT/train.json')); print(len(d))" 2>/dev/null || echo 0)"
    echo ""
fi

# ============================================================
# STEP 4: VLM Captioning
# ============================================================
if [ "$STEP" = "all" ] || [ "$STEP" = "caption" ]; then
    echo "============================================================"
    echo "[STEP 4] Running VLM captioner on processed clips..."
    echo "============================================================"

    if [ ! -f "$DATASET_OUT/train.json" ]; then
        echo "[ERROR] No train.json found at $DATASET_OUT/train.json. Run processing step first."
        exit 1
    fi

    python dataset_generation/processing/vlm_captioner.py \
        --train_json "$DATASET_OUT/train.json" \
        --data_root "$DATASET_OUT"

    echo ""
fi

# ============================================================
# STEP 5: Stage training data
# ============================================================
if [ "$STEP" = "all" ] || [ "$STEP" = "stage" ]; then
    echo "============================================================"
    echo "[STEP 5] Staging training data to $TRAIN_DATA_DIR..."
    echo "============================================================"

    mkdir -p "$TRAIN_DATA_DIR/clips"

    # Copy/link clips
    if [ -d "$DATASET_OUT/clips" ]; then
        echo "[INFO] Linking clips..."
        for clip in "$DATASET_OUT/clips"/*.mp4; do
            basename=$(basename "$clip")
            dst="$TRAIN_DATA_DIR/clips/$basename"
            if [ ! -e "$dst" ]; then
                ln -s "$(realpath "$clip")" "$dst" 2>/dev/null || cp "$clip" "$dst"
            fi
        done
    fi

    # Copy train.json
    if [ -f "$DATASET_OUT/train.json" ]; then
        cp "$DATASET_OUT/train.json" "$TRAIN_DATA_DIR/train.json"
        echo "[OK] train.json copied to $TRAIN_DATA_DIR/"
    fi

    echo ""
    echo "Training data staged:"
    echo "  Clips: $(find $TRAIN_DATA_DIR/clips -name '*.mp4' 2>/dev/null | wc -l)"
    echo "  train.json: $(python -c "import json; d=json.load(open('$TRAIN_DATA_DIR/train.json')); print(len(d))" 2>/dev/null || echo 0) entries"
fi

echo ""
echo "============================================================"
echo "[DONE] Traditional dataset pipeline complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  - Review data:  python -c \"import json; d=json.load(open('$TRAIN_DATA_DIR/train.json')); print(len(d), 'samples')\""
echo "  - Train model:  qsub scripts/train_lora.sh"
