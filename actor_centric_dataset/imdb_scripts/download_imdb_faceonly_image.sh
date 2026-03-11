#!/bin/bash

ROOT_DIR=$(pwd)
DATA_DIR="$ROOT_DIR/imdb_images/imdb_data_faceonly"
TAR_DIR="$DATA_DIR/tar"

mkdir -p "$ROOT_DIR/imdb_images"
mkdir -p "$TAR_DIR"

echo "[INFO] Start Downloading IMDB Face Only Images"

BASE_URL="https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static"

wget -c "$BASE_URL/imdb_crop.tar" -P "$TAR_DIR"
wget -c "$BASE_URL/wiki_crop.tar" -P "$TAR_DIR"

echo "[INFO] All tar files downloaded to $TAR_DIR"

cd "$DATA_DIR"

echo "[INFO] Extracting .tar files..."
for tarfile in "$TAR_DIR"/*.tar; do
    echo "Extracting $tarfile..."
    tar -xf "$tarfile" -C "$DATA_DIR"
done

echo "[INFO] Extraction complete! Files are in $DATA_DIR"