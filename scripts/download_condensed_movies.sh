#!/bin/bash -l
#
# Download Condensed Movies dataset from YouTube.
#
#$ -P cs585
#$ -N cm_download
#$ -l h_rt=24:00:00
#$ -l mem_per_core=4G
#$ -pe omp 4
#$ -cwd
#$ -j y
#$ -o /projectnb/cs585/students/sanjiv/CineAlign-AR/scripts/cm_download.out

set -e

PROJECT_ROOT="/projectnb/cs585/students/sanjiv/CineAlign-AR"
cd "$PROJECT_ROOT"

module load miniconda
module load academic-ml/spring-2026
conda activate spring-2026-pyt

DATA_DIR="$PROJECT_ROOT/data/condensed_movies"
META_DIR="$PROJECT_ROOT/data/metadata"

mkdir -p "$DATA_DIR" "$META_DIR"

echo "============================================================"
echo "Condensed Movies Download"
echo "Project root:  $PROJECT_ROOT"
echo "Data dir:      $DATA_DIR"
echo "Metadata dir:  $META_DIR"
echo "Host:          $(hostname)"
echo "Python:        $(which python)"
echo "Conda env:     $CONDA_DEFAULT_ENV"
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

VIDEO_COUNT=$(find "$DATA_DIR/videos" -name "*.mp4" 2>/dev/null | wc -l)
echo "Total .mp4 files: $VIDEO_COUNT"