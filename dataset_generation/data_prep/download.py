"""
Condensed Movies Download Pipeline.

Downloads source videos from YouTube using yt-dlp, trims ad outros,
and reports download success rate.

Usage:
    # Interactive (will ask about trimming):
    python download.py

    # Non-interactive batch mode (auto-trim, for SCC jobs):
    python download.py --batch

    # Custom paths:
    python download.py --data_dir ../../data/condensed_movies --metadata_dir ../../data/metadata

    # Only download metadata CSVs:
    python download.py --metadata_only
"""

import json
import os
import argparse
import shutil
from os.path import join as osj

import pandas as pd

# Metadata lives on GitHub, NOT on the Oxford hosting server
GITHUB_RAW = 'https://raw.githubusercontent.com/m-bain/CondensedMovies/master/data/metadata'

# Default paths (relative to this file's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'condensed_movies')
DEFAULT_METADATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'metadata')


def download_metadata(metadata_dir):
    """Download Condensed Movies metadata CSVs from the GitHub repo."""
    os.makedirs(metadata_dir, exist_ok=True)

    files = ['clips.csv', 'durations.csv']

    for f in files:
        out_path = osj(metadata_dir, f)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            print(f"[OK] {f} already exists ({os.path.getsize(out_path)} bytes).")
            continue

        url = f"{GITHUB_RAW}/{f}"
        print(f"[INFO] Downloading {url} -> {out_path}")
        cmd = f'wget -q "{url}" -O "{out_path}" || curl -sL "{url}" -o "{out_path}"'
        ret = os.system(cmd)

        # Verify download succeeded
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 100:
            print(f"[ERROR] Failed to download {f}. Try manually:")
            print(f"  wget '{url}' -O '{out_path}'")
        else:
            print(f"[OK] {f}: {os.path.getsize(out_path)} bytes")

    # Download YouTube ID dump CSVs (one per upload year)
    yt_dump_dir = osj(metadata_dir, 'youtube-dl-dump')
    if not os.path.isdir(yt_dump_dir) or len(os.listdir(yt_dump_dir)) == 0:
        print(f"[INFO] Generating YouTube ID lists from clips.csv...")
        clips_csv = osj(metadata_dir, 'clips.csv')
        if os.path.exists(clips_csv):
            os.makedirs(yt_dump_dir, exist_ok=True)
            clips_df = pd.read_csv(clips_csv)
            # Group unique video IDs by upload_year
            for year, group in clips_df.groupby('upload_year'):
                year_file = osj(yt_dump_dir, f"{int(year)}.csv")
                unique_ids = group['videoid'].unique()
                with open(year_file, 'w') as fh:
                    for vid_id in unique_ids:
                        fh.write(f"{vid_id}\n")
                print(f"  [OK] {year_file}: {len(unique_ids)} video IDs")
        else:
            print(f"[ERROR] clips.csv not found at {clips_csv}. Cannot generate ID lists.")


def youtube_download(data_dir, metadata_dir, batch_mode=False):
    """
    Download source videos using yt-dlp.
    Reads YouTube ID lists from metadata_dir/youtube-dl-dump/.
    Saves videos to data_dir/videos/{upload_year}/{videoid}.mp4.
    """
    id_dir = osj(metadata_dir, 'youtube-dl-dump')
    video_dir = osj(data_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)

    if not os.path.isdir(id_dir) or len(os.listdir(id_dir)) == 0:
        print(f"[ERROR] No YouTube ID lists found at {id_dir}")
        print(f"  Run with --metadata_only first to generate them from clips.csv")
        return

    # Download videos year by year
    for file in sorted(os.listdir(id_dir)):
        if not file.endswith('.csv'):
            continue

        upload_year = file.replace('.csv', '')
        video_dir_year = osj(video_dir, upload_year)
        os.makedirs(video_dir_year, exist_ok=True)

        output_fmt = osj(video_dir_year, '%(id)s.%(ext)s')
        id_fp = osj(id_dir, file)

        cmd = (
            'yt-dlp '
            '-f "bestvideo[height<=480]+bestaudio/best[height<=480]" '
            '--merge-output-format mp4 '
            '--ignore-errors '
            '--no-overwrites '
            f'-o "{output_fmt}" '
            f'-a "{id_fp}"'
        )

        print(f"\n[CMD] Downloading {upload_year} videos...")
        print(f"  {cmd}")
        os.system(cmd)

    # Trim ad outros
    if batch_mode:
        trim = True
    else:
        response = None
        while response not in ['y', 'n']:
            response = input(
                "\nTrim advertisement outros from downloaded videos? (y/n): "
            ).strip().lower()
        trim = (response == 'y')

    if trim:
        durations_csv = osj(metadata_dir, 'durations.csv')
        if os.path.exists(durations_csv):
            print("[INFO] Trimming advertisement outros...")
            trim_video_outro(video_dir, durations_csv, video_ext='.mp4')
        else:
            print(f"[WARN] Cannot trim: {durations_csv} not found.")

    # Report download success
    clips_csv = osj(metadata_dir, 'clips.csv')
    if os.path.exists(clips_csv):
        check_missing_vids(video_dir, clips_csv, video_ext='.mp4')


def trim_video_outro(video_dir, durations_csv, video_ext='.mp4'):
    """
    Trim advertisement outro from videos using durations.csv.
    Uses ffmpeg for frame-accurate cutting.
    """
    duration_data = pd.read_csv(durations_csv).set_index('videoid')
    trimmed_count = 0

    tmp_fp = osj(video_dir, 'tmp' + video_ext)
    for root, _, files in os.walk(video_dir):
        for file in files:
            if not file.endswith(video_ext) or file == 'tmp' + video_ext:
                continue

            videoid = file.replace(video_ext, '')
            if videoid not in duration_data.index:
                continue

            video_fp = osj(root, file)
            new_duration = duration_data.loc[videoid]['duration']

            shutil.copyfile(video_fp, tmp_fp)
            cmd = f'ffmpeg -y -ss 0 -i "{tmp_fp}" -t {new_duration} -c copy "{video_fp}" -loglevel error'
            os.system(cmd)
            trimmed_count += 1

    if os.path.exists(tmp_fp):
        os.remove(tmp_fp)

    print(f"[OK] Trimmed {trimmed_count} videos.")


def check_missing_vids(video_dir, clips_csv, video_ext='.mp4'):
    """Check which videos from clips.csv were not successfully downloaded."""
    missing_ids = []
    clips_data = pd.read_csv(clips_csv)

    unique_videos = clips_data.drop_duplicates(subset='videoid')
    for _, row in unique_videos.iterrows():
        videoid = row['videoid']
        upload_year = row.get('upload_year', None)

        # Search in year subdirectory and flat directory
        found = False
        if upload_year is not None:
            year_path = osj(video_dir, str(int(upload_year)), videoid + video_ext)
            if os.path.isfile(year_path):
                found = True
        if not found:
            flat_path = osj(video_dir, videoid + video_ext)
            if os.path.isfile(flat_path):
                found = True

        if not found:
            missing_ids.append(videoid)

    total = len(unique_videos)
    downloaded = total - len(missing_ids)
    success = (downloaded / total * 100) if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Download Results: {downloaded}/{total} videos ({success:.1f}%)")
    print(f"{'='*60}")

    if missing_ids:
        print(f"{len(missing_ids)} videos failed (likely YouTube takedowns or geo-restrictions).")
        missing_file = osj(video_dir, '..', 'missing_videos.txt')
        with open(missing_file, 'w') as f:
            for mid in missing_ids:
                f.write(mid + '\n')
        print(f"Missing IDs saved to: {missing_file}")


def main():
    parser = argparse.ArgumentParser(description="Download Condensed Movies dataset")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Directory to save downloaded videos")
    parser.add_argument("--metadata_dir", default=DEFAULT_METADATA_DIR,
                        help="Directory containing/to download metadata CSVs")
    parser.add_argument("--batch", action="store_true",
                        help="Non-interactive batch mode (auto-trim, no prompts)")
    parser.add_argument("--metadata_only", action="store_true",
                        help="Only download metadata CSVs, skip video download")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip video download, only trim and check")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    metadata_dir = os.path.abspath(args.metadata_dir)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # Always download metadata first
    print(f"[INFO] Metadata directory: {metadata_dir}")
    print(f"[INFO] Data directory: {data_dir}")
    download_metadata(metadata_dir)

    if args.metadata_only:
        print("[DONE] Metadata download complete.")
        return

    if not args.skip_download:
        youtube_download(data_dir, metadata_dir, batch_mode=args.batch)
    else:
        # Just trim and check
        durations_csv = osj(metadata_dir, 'durations.csv')
        if os.path.exists(durations_csv):
            video_dir = osj(data_dir, 'videos')
            trim_video_outro(video_dir, durations_csv)
        clips_csv = osj(metadata_dir, 'clips.csv')
        if os.path.exists(clips_csv):
            check_missing_vids(osj(data_dir, 'videos'), clips_csv)

    print("\n[DONE] Condensed Movies download pipeline complete.")


if __name__ == "__main__":
    main()
