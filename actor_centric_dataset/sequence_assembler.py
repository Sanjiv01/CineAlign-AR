"""
Multi-Shot Sequence Assembler for Actor-Centric Dataset.

Takes the output of actor_clip_matcher.py (actors_clips.json) and assembles
multi-shot training sequences where clips of the same actor come from different movies.

Output format matches EchoShot's train.json:
    {"shots": ["clip1.mp4", "clip2.mp4", ...], "cap_list": [...], "inner_t": [...]}

Usage:
    python sequence_assembler.py \
        --actors_json actors_clips.json \
        --video_dir /path/to/movies \
        --output_dir actor_centric_out \
        --output_json actor_centric_train.json
"""

import os
import json
import math
import random
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict


# Config
MIN_SHOTS = 2
MAX_SHOTS = 4
TARGET_TOTAL_FRAMES = 125
MIN_FRAMES_PER_SHOT = 20
MAX_SAMPLES_PER_ACTOR = 10
MAX_ATTEMPTS = 40

TARGET_W = 832
TARGET_H = 480
DOWNSAMPLE_T = 4  # temporal downsampling factor for VAE


def resize_and_pad(frame, target_w=TARGET_W, target_h=TARGET_H):
    """Resize and letterbox to target resolution."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))
    pad_w = target_w - new_w
    pad_h = target_h - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def compute_inner_t(frame_counts):
    """Convert pixel frame counts to latent time dimensions."""
    return [(f - 1) // DOWNSAMPLE_T + 1 for f in frame_counts]


def compute_weighted_trims(n_frames_list, target_total=TARGET_TOTAL_FRAMES):
    """Distribute target_total frames across shots proportionally to their available frames."""
    total_available = sum(n_frames_list)
    if total_available <= target_total:
        return n_frames_list

    trimmed = []
    remaining = target_total
    for i, nf in enumerate(n_frames_list):
        if i == len(n_frames_list) - 1:
            alloc = remaining
        else:
            alloc = max(MIN_FRAMES_PER_SHOT, int(target_total * nf / total_available))
        alloc = min(alloc, nf, remaining)
        trimmed.append(alloc)
        remaining -= alloc

    return trimmed


def export_clip(video_path, start_frame, n_frames, output_path):
    """Export a clip from a video file at target resolution."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_W, TARGET_H))

    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_and_pad(frame)
        writer.write(frame)

    writer.release()
    cap.release()


def assemble_sequences(actors_json_path, video_dir, output_dir, output_json_path):
    """Main assembly pipeline."""
    with open(actors_json_path, "r") as f:
        actors_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    clip_dir = os.path.join(output_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)

    train_entries = []
    clip_idx = 0

    # Filter to actors with clips from 2+ movies
    usable_actors = {}
    for actor, clips in actors_data.items():
        movies = set(c["movie"] for c in clips)
        if len(movies) >= 2 and len(clips) >= MIN_SHOTS:
            usable_actors[actor] = clips

    print(f"[INFO] {len(usable_actors)} actors with clips from 2+ movies")

    for actor, clips in tqdm(usable_actors.items(), desc="Assembling sequences"):
        # Group clips by movie
        by_movie = defaultdict(list)
        for c in clips:
            by_movie[c["movie"]].append(c)

        movies = list(by_movie.keys())
        if len(movies) < 2:
            continue

        samples_created = 0
        for attempt in range(MAX_ATTEMPTS):
            if samples_created >= MAX_SAMPLES_PER_ACTOR:
                break

            # Decide number of shots
            num_shots = random.randint(MIN_SHOTS, min(MAX_SHOTS, len(movies)))

            # Sample from different movies
            selected_movies = random.sample(movies, num_shots)
            selected_clips = []
            for m in selected_movies:
                selected_clips.append(random.choice(by_movie[m]))

            # Check we have enough frames
            n_frames_list = [c["n_frames"] for c in selected_clips]
            if any(nf < MIN_FRAMES_PER_SHOT for nf in n_frames_list):
                continue
            if sum(n_frames_list) < TARGET_TOTAL_FRAMES:
                continue

            # Compute frame allocation
            trimmed = compute_weighted_trims(n_frames_list)
            if any(t < MIN_FRAMES_PER_SHOT for t in trimmed):
                continue

            # Export clips
            shot_paths = []
            for i, (clip_info, n_frames) in enumerate(zip(selected_clips, trimmed)):
                clip_name = f"actor_{clip_idx:06d}_s{i:02d}.mp4"
                clip_path = os.path.join(clip_dir, clip_name)

                # Center the clip extraction
                available = clip_info["n_frames"]
                start = clip_info["start_frame"]
                if available > n_frames:
                    offset = random.randint(0, available - n_frames)
                    start += offset

                export_clip(clip_info["video_path"], start, n_frames, clip_path)
                shot_paths.append(os.path.join("clips", clip_name))

            # Create train entry (captions will be filled by VLM captioner later)
            inner_t = compute_inner_t(trimmed)
            entry = {
                "shots": shot_paths,
                "cap_list": ["" for _ in shot_paths],  # Placeholder - run vlm_captioner.py
                "inner_t": inner_t,
            }
            train_entries.append(entry)
            clip_idx += 1
            samples_created += 1

    # Save
    with open(output_json_path, "w") as f:
        json.dump(train_entries, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Created {len(train_entries)} training samples -> {output_json_path}")
    print(f"[DONE] Clips saved to {clip_dir}")


def main():
    parser = argparse.ArgumentParser(description="Actor-centric multi-shot sequence assembly")
    parser.add_argument("--actors_json", required=True, help="Output from actor_clip_matcher.py")
    parser.add_argument("--video_dir", required=True, help="Root video directory")
    parser.add_argument("--output_dir", default="actor_centric_out", help="Output directory")
    parser.add_argument("--output_json", default="actor_centric_train.json", help="Output train.json")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    assemble_sequences(args.actors_json, args.video_dir, args.output_dir, args.output_json)


if __name__ == "__main__":
    main()
