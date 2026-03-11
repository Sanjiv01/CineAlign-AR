"""
Actor-Centric Cross-Movie Clip Matching.

Matches video clips to known actors across different movies using face embeddings.
Uses InsightFace (same embedding space as the training data pipeline) for consistency.

Pipeline:
1. Load actor CSV (filtered_movies_with_cast.csv) with top-5 actors per movie
2. Load actor reference face embeddings (from IMDB images)
3. For each movie video: scene detect -> person detect -> face embed -> match to actors
4. Output: JSON mapping actor_name -> list of (movie, clip_start, clip_end, embedding)

Usage:
    python actor_clip_matcher.py \
        --actor_csv filtered_movies_with_cast.csv \
        --actor_features_dir imdb_images/actor_features \
        --video_dir /path/to/movie/clips \
        --output actors_clips.json
"""

import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity


# Config
FACE_SIM_THRESHOLD = 0.45       # Lower than within-video threshold for cross-movie matching
MIN_SCENE_FRAMES = 50           # Minimum frames per scene
KEYFRAMES_PER_SCENE = 5


def load_actor_embeddings(features_dir):
    """Load precomputed actor face embeddings from .pt files."""
    actor_embs = {}
    for fname in os.listdir(features_dir):
        if fname.endswith(".pt"):
            actor_name = fname[:-3]  # Remove .pt extension
            emb = torch.load(os.path.join(features_dir, fname), map_location="cpu")
            actor_embs[actor_name] = emb.numpy() if isinstance(emb, torch.Tensor) else emb
    print(f"[INFO] Loaded {len(actor_embs)} actor embeddings")
    return actor_embs


def load_actor_csv(csv_path):
    """Load filtered_movies_with_cast.csv and return movie -> [actor1, ..., actor5] mapping."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    movie_actors = {}
    for _, row in df.iterrows():
        title = row["title"]
        actors = []
        for i in range(1, 6):
            col = f"actor_{i}"
            if col in df.columns and pd.notna(row[col]) and str(row[col]).strip():
                actors.append(str(row[col]).strip().lower())
        if actors:
            movie_actors[title] = actors
    print(f"[INFO] Loaded {len(movie_actors)} movies with actor info")
    return movie_actors


def normalize_name(name):
    """Normalize actor name for matching."""
    return name.replace("'", "").lower().strip()


def extract_face_embedding(face_app, frame):
    """Extract face embedding from a frame using InsightFace."""
    faces = face_app.get(frame)
    if len(faces) != 1:
        return None  # Only keep single-person frames
    return faces[0].normed_embedding


def match_clips_to_actors(video_dir, movie_actors, actor_embs, face_app, output_path):
    """Main matching pipeline."""
    # Import scene detection
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    import cv2

    actors_dict = defaultdict(list)  # actor_name -> [(movie, clip_path, start, end)]

    video_files = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
    ])

    for vfile in tqdm(video_files, desc="Processing movies"):
        vpath = os.path.join(video_dir, vfile)
        movie_title = os.path.splitext(vfile)[0]

        # Find actors for this movie
        matched_actors = movie_actors.get(movie_title, [])
        if not matched_actors:
            continue

        # Get actor embeddings for this movie's top actors
        movie_actor_embs = {}
        for actor in matched_actors:
            norm = normalize_name(actor)
            if norm in actor_embs:
                movie_actor_embs[norm] = actor_embs[norm]

        if not movie_actor_embs:
            continue

        # Scene detection
        try:
            vm = VideoManager([vpath])
            sm = SceneManager()
            sm.add_detector(ContentDetector(threshold=27))
            vm.start()
            sm.detect_scenes(frame_source=vm)
            scenes = sm.get_scene_list()
            vm.release()
        except Exception as e:
            print(f"[WARN] Scene detection failed for {vfile}: {e}")
            continue

        if not scenes:
            continue

        # For each scene, extract face and match to actors
        cap = cv2.VideoCapture(vpath)
        for start_tc, end_tc in scenes:
            start_frame = int(start_tc.get_frames())
            end_frame = int(end_tc.get_frames())
            n_frames = end_frame - start_frame

            if n_frames < MIN_SCENE_FRAMES:
                continue

            # Sample keyframes
            indices = np.linspace(start_frame, end_frame - 1, KEYFRAMES_PER_SCENE, dtype=int)
            embeddings = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                emb = extract_face_embedding(face_app, frame)
                if emb is not None:
                    embeddings.append(emb)

            if len(embeddings) < 2:
                continue

            # Average embedding for this scene
            scene_emb = np.mean(embeddings, axis=0).reshape(1, -1)

            # Match against movie's actors
            for actor_name, actor_emb in movie_actor_embs.items():
                actor_emb_2d = actor_emb.reshape(1, -1)
                sim = cosine_similarity(scene_emb, actor_emb_2d)[0, 0]
                if sim > FACE_SIM_THRESHOLD:
                    actors_dict[actor_name].append({
                        "movie": movie_title,
                        "video_path": vpath,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "n_frames": n_frames,
                        "similarity": float(sim),
                    })

        cap.release()

    # Save results
    # Convert to regular dict for JSON serialization
    result = {k: v for k, v in actors_dict.items()}
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Stats
    total_clips = sum(len(v) for v in result.values())
    multi_movie = sum(1 for v in result.values()
                      if len(set(c["movie"] for c in v)) >= 2)
    print(f"[DONE] {len(result)} actors, {total_clips} total clips")
    print(f"[DONE] {multi_movie} actors with clips from 2+ movies (usable for actor-centric data)")

    return result


def main():
    parser = argparse.ArgumentParser(description="Cross-movie actor clip matching")
    parser.add_argument("--actor_csv", required=True, help="Path to filtered_movies_with_cast.csv")
    parser.add_argument("--actor_features_dir", required=True, help="Directory with actor .pt embeddings")
    parser.add_argument("--video_dir", required=True, help="Directory with movie video files")
    parser.add_argument("--output", default="actors_clips.json", help="Output JSON path")
    parser.add_argument("--face_sim_threshold", type=float, default=FACE_SIM_THRESHOLD)
    args = parser.parse_args()

    global FACE_SIM_THRESHOLD
    FACE_SIM_THRESHOLD = args.face_sim_threshold

    # Load actor data
    movie_actors = load_actor_csv(args.actor_csv)
    actor_embs = load_actor_embeddings(args.actor_features_dir)

    # Initialize face analysis
    print("[INFO] Loading InsightFace...")
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=-1, det_size=(256, 256))

    # Run matching
    match_clips_to_actors(args.video_dir, movie_actors, actor_embs, face_app, args.output)


if __name__ == "__main__":
    main()
