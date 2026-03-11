#!/bin/bash
#
# Download Condensed Movies dataset from YouTube.
# Run on SCC login node (network access required) or as a batch job.
#
# Usage:
#   bash scripts/download_condensed_movies.sh              # Full download
#   bash scripts/download_condensed_movies.sh --metadata   # Metadata only
#
#$ -P cs523aw
#$ -l h_rt=24:00:00
#$ -N cm_download
#$ -j y
#
#$ -pe omp 4
#$ -l mem_per_core=4G

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Load modules
module load python3/3.10.12
module load ffmpeg/6.0

source .venv/bin/activate

DATA_DIR="$PROJECT_ROOT/data/condensed_movies"
META_DIR="$PROJECT_ROOT/data/metadata"

echo "============================================================"
echo "Condensed Movies Download"
echo "  Data dir:     $DATA_DIR"
echo "  Metadata dir: $META_DIR"
echo "============================================================"

if [ "$1" = "--metadata" ]; then
    echo "[INFO] Downloading metadata only..."
    python dataset_generation/data_prep/download.py \
        --data_dir "$DATA_DIR" \
        --metadata_dir "$META_DIR" \
        --metadata_only
else
    echo "[INFO] Full download (metadata + videos + trim)..."
    python dataset_generation/data_prep/download.py \
        --data_dir "$DATA_DIR" \
        --metadata_dir "$META_DIR" \
        --batch
fi

echo ""
echo "============================================================"
echo "[DONE] Download complete."
echo "============================================================"

# Count downloaded videos
VIDEO_COUNT=$(find "$DATA_DIR/videos" -name "*.mp4" 2>/dev/null | wc -l)
echo "Total .mp4 files: $VIDEO_COUNT"
