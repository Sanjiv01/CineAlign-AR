import os
import cv2
import fire
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def extract_keyframes(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = int(total_frames / fps)

    os.makedirs(save_dir, exist_ok=True)

    for sec in range(duration_sec):
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if success:
            cv2.imwrite(os.path.join(save_dir, f"frame_{sec:04d}.jpg"), frame)

    cap.release()

    return


def process_single_clip(video_path, save_dir, log_file):
    video_name = os.path.basename(video_path)
    save_folder = os.path.join(save_dir, os.path.splitext(video_name)[0])

    if os.path.exists(save_folder) and len(os.listdir(save_folder)) > 0:
        return 

    extract_keyframes(video_path, save_folder)

    with open(log_file, "a") as f:
        f.write(f"{video_name}\n")

    return


def main(videos_dir: str, save_dir: str, num_workers: int = 8, log_file: str="logs/imdb_features_processed.log"):
    video_clips = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
    video_paths = [os.path.join(videos_dir, f) for f in video_clips]
    os.makedirs(save_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(
            executor.map(lambda vp: process_single_clip(vp, save_dir, log_file), video_paths),
            total=len(video_paths),
            desc="Extracting frames",
        ))

    print(f"[DONE] Processed {len(video_paths)} videos. Log saved to {log_file}")

    return


if __name__ == "__main__":
    fire.Fire(main)
