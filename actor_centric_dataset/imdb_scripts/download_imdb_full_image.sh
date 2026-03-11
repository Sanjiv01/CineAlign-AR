#!/bin/bash

ROOT_DIR=$(pwd)
DATA_DIR="$ROOT_DIR/imdb_images/imdb_data_fullimage"
TAR_DIR="$DATA_DIR/tar"

mkdir -p "$ROOT_DIR/imdb_images"
mkdir -p "$TAR_DIR"

echo "[INFO] Start Downloading IMDB Full Images"

BASE_URL="https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static"

echo "[INFO] Downloading .tar files..."
for i in {0..9}; do
    wget -c "$BASE_URL/imdb_${i}.tar" -P "$TAR_DIR"
done

wget -c "$BASE_URL/imdb_meta.tar" -P "$TAR_DIR"
wget -c "$BASE_URL/wiki.tar.gz" -P "$TAR_DIR"

echo "[INFO] All tar files downloaded to $TAR_DIR"

cd "$DATA_DIR"

echo "[INFO] Extracting .tar files..."
for tarfile in "$TAR_DIR"/*.tar; do
    echo "Extracting $tarfile..."
    tar -xf "$tarfile" -C "$DATA_DIR"
done

echo "[INFO] Extraction complete! Files are in $DATA_DIR"