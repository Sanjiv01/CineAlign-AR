import json
import os
from os.path import join as osj
import pandas as pd
import shutil  # for portable copy

hosting_address = 'https://thor.robots.ox.ac.uk/~vgg/data/condensed-movies/data'


def download_features(data_dir):
    # Not used if config["features"] = False
    cmd = 'wget {}/features.zip -P {}; unzip {}/features.zip -d {}'.format(
        hosting_address, data_dir, data_dir, data_dir
    )
    os.system(cmd)


def download_facetracks(data_dir):
    # Not used if config["facetracks"] = False
    cmd = 'wget {}/facetracks.zip -P {}; unzip {}/facetracks.zip -d {}'.format(
        hosting_address, data_dir, data_dir, data_dir
    )
    os.system(cmd)


def youtube_download(data_dir):
    """
    Download source videos using yt-dlp instead of youtube-dl.
    - Reads CSV ID lists from ../data/metadata/youtube-dl-dump
    - Saves videos under {data_dir}/videos/{upload_year}/{videoid}.mp4
    """
    id_dir = '../data/metadata/youtube-dl-dump'
    video_dir = osj(data_dir, 'videos')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # Loop over per-year CSVs containing YouTube IDs
    for file in os.listdir(id_dir):
        if not file.endswith('.csv'):
            continue

        upload_year = file.replace('.csv', '')
        video_dir_year = osj(video_dir, upload_year)
        if not os.path.exists(video_dir_year):
            os.makedirs(video_dir_year)

        # Output pattern: {video_dir_year}/{id}.mp4
        output_fmt = osj(video_dir_year, '%(id)s.%(ext)s')
        id_fp = osj(id_dir, file)

        # yt-dlp command:
        # - bestvideo+bestaudio/best
        # - merge to mp4
        # - read IDs from CSV file (one id per line or column yt-dlp understands)
        cmd = (
            'yt-dlp '
            '-f "bestvideo+bestaudio/best" '
            '--merge-output-format mp4 '
            '-o "{}" '
            '-a "{}"'
        ).format(output_fmt, id_fp)

        print(f"[CMD] {cmd}")
        os.system(cmd)

    # Ask user if they want to trim ads/outros
    trim = None
    while trim not in ['y', 'n']:
        trim = str(input(
            "\nDo you want to trim the videos (y/n)?\n"
            "This removes the advertisements (unrelated to the film), "
            "and only needs to be done once per download."
        )).strip().lower()

        if trim not in ['y', 'n']:
            print('Please type "y" or "n"')

    if trim == "y":
        trim_video_outro(video_dir, video_ext='.mp4')

    # check for failed downloads (missing mp4 files)
    check_missing_vids(video_dir, video_ext='.mp4')


def trim_video_outro(video_dir, video_ext='.mp4'):
    """
    Trim advertisement outro from videos using durations.csv.
    Uses shutil.copyfile instead of `cp` for Windows compatibility.
    """
    duration_data = pd.read_csv('../data/metadata/durations.csv').set_index('videoid')

    tmp_fp = osj(video_dir, 'tmp' + video_ext)
    for root, subdir, files in os.walk(video_dir):
        for file in files:
            if file.endswith(video_ext) and file != 'tmp' + video_ext:
                videoid = file.split(video_ext)[0]
                if videoid not in duration_data.index:
                    raise ValueError("Videoid not found, video files should be in format {VIDEOID}" + video_ext)

                video_fp = osj(root, file)
                new_duration = duration_data.loc[videoid]['duration']

                # create tmp for untrimmed (portable copy)
                shutil.copyfile(video_fp, tmp_fp)
                cmd = f'ffmpeg -y -ss 0 -i "{tmp_fp}" -t {new_duration} -c copy "{video_fp}"'
                os.system(cmd)
                os.remove(tmp_fp)


def check_missing_vids(video_dir, video_ext='.mp4'):
    """
    Check which videos from clips.csv were not successfully downloaded.
    """
    missing_ids = []
    clips_data = pd.read_csv('../data/metadata/clips.csv').set_index('videoid')
    for idx, row in clips_data.iterrows():
        videoid = row.name
        upload_year = row['upload_year']
        video_fp = osj(video_dir, str(int(upload_year)), videoid + video_ext)
        if not os.path.isfile(video_fp):
            missing_ids.append(videoid)

    success = (1 - len(missing_ids) / len(clips_data)) * 100
    print('=======================================================')
    print('%.2f %% of clips downloaded successfully' % success)
    if success == 100:
        pass
    elif success < 100:
        print(
            '%d clips failed to download. This is likely due to geographical restrictions.\n'
            '-->Contact maxbain@robots.ox.ac.uk if this is an issue.' % len(missing_ids)
        )

    with open('missing_videos.out', 'w') as fid:
        for mid in missing_ids:
            fid.write(mid + '\n')


def main():
    config = json.load(open('config.json', 'r'))
    data_dir = config['data_dir']
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if config.get('features', False):
        download_features(data_dir)
    if config.get('facetracks', False):
        download_facetracks(data_dir)
    if config.get('src', False):
        youtube_download(data_dir)


if __name__ == "__main__":
    main()
