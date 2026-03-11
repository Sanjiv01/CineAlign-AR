"""
Prepare downloaded Condensed Movies videos for the dataset processing pipeline.

This script:
1. Scans the Condensed Movies download directory for .mp4 files
2. Creates symlinks (or copies) into dataset_in/ for the processing pipeline
3. Optionally filters by minimum duration or resolution

Usage:
    # Symlink all downloaded videos into dataset_in/
    python prepare_input.py --src_dir ../../data/condensed_movies/videos --dst_dir ../../dataset_in

    # Filter to only videos longer than 60 seconds
    python prepare_input.py --src_dir ../../data/condensed_movies/videos --dst_dir ../../dataset_in --min_duration 60

    # Copy instead of symlink (for cross-filesystem)
    python prepare_input.py --src_dir ../../data/condensed_movies/videos --dst_dir ../../dataset_in --copy
"""

import os
import sys
import argparse
import shutil
import subprocess


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return -1


def main():
    parser = argparse.ArgumentParser(description="Prepare videos for dataset processing")
    parser.add_argument("--src_dir", required=True, help="Source directory with downloaded videos")
    parser.add_argument("--dst_dir", default="../../dataset_in", help="Destination directory (dataset_in/)")
    parser.add_argument("--min_duration", type=float, default=30.0,
                        help="Minimum video duration in seconds (default: 30)")
    parser.add_argument("--max_videos", type=int, default=-1,
                        help="Maximum number of videos to prepare (-1 = all)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of creating symlinks")
    parser.add_argument("--dry_run", action="store_true",
                        help="Just report what would be done, don't actually link/copy")
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src_dir)
    dst_dir = os.path.abspath(args.dst_dir)

    if not os.path.isdir(src_dir):
        print(f"[ERROR] Source directory not found: {src_dir}")
        sys.exit(1)

    os.makedirs(dst_dir, exist_ok=True)

    # Collect all .mp4 files recursively
    video_files = []
    for root, _, files in os.walk(src_dir):
        for f in sorted(files):
            if f.lower().endswith('.mp4'):
                video_files.append(os.path.join(root, f))

    print(f"[INFO] Found {len(video_files)} .mp4 files in {src_dir}")

    # Filter and link/copy
    linked = 0
    skipped_duration = 0
    skipped_exists = 0

    for vpath in video_files:
        if args.max_videos > 0 and linked >= args.max_videos:
            break

        basename = os.path.basename(vpath)
        dst_path = os.path.join(dst_dir, basename)

        # Skip if already exists in destination
        if os.path.exists(dst_path):
            skipped_exists += 1
            continue

        # Check duration
        if args.min_duration > 0:
            duration = get_video_duration(vpath)
            if duration > 0 and duration < args.min_duration:
                skipped_duration += 1
                continue

        if args.dry_run:
            action = "COPY" if args.copy else "LINK"
            print(f"  [{action}] {basename}")
        else:
            if args.copy:
                shutil.copy2(vpath, dst_path)
            else:
                os.symlink(vpath, dst_path)

        linked += 1

    action = "would prepare" if args.dry_run else "prepared"
    print(f"\n[DONE] {action} {linked} videos -> {dst_dir}")
    if skipped_duration:
        print(f"  Skipped (too short): {skipped_duration}")
    if skipped_exists:
        print(f"  Skipped (already exists): {skipped_exists}")


if __name__ == "__main__":
    main()
