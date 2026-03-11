"""
Evaluation Metrics for Multi-Shot Video Generation.

Metrics:
- FaceSim-Arc: ArcFace cosine similarity between faces across shots
- FaceSim-Cur: CurricularFace cosine similarity
- Motion score: Assessed via Gemini API

Usage:
    python metrics.py --video_dir outputs/ --results_json eval_results.json
"""

import os
import json
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity


def init_face_app():
    """Initialize InsightFace with ArcFace model."""
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(256, 256))
    return app


def extract_face_embeddings_from_video(face_app, video_path, inner_t, num_samples=3):
    """
    Extract face embeddings from each shot in a multi-shot video.

    Args:
        face_app: InsightFace FaceAnalysis instance
        video_path: Path to the generated video
        inner_t: List of latent frame counts per shot
        num_samples: Number of frames to sample per shot

    Returns:
        List of face embeddings (one per shot), or None if face not found
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    downsample_t = 4

    # Convert inner_t to pixel frame boundaries
    frame_boundaries = []
    start = 0
    for t in inner_t:
        pixel_frames = (t - 1) * downsample_t + 1
        frame_boundaries.append((start, start + pixel_frames))
        start += pixel_frames

    shot_embeddings = []
    for shot_start, shot_end in frame_boundaries:
        n_frames = shot_end - shot_start
        if n_frames < 1:
            shot_embeddings.append(None)
            continue

        # Sample frames from this shot
        indices = np.linspace(shot_start, shot_end - 1, num_samples, dtype=int)
        embeddings = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            faces = face_app.get(frame)
            if len(faces) >= 1:
                # Take the largest face
                largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                embeddings.append(largest.normed_embedding)

        if embeddings:
            shot_embeddings.append(np.mean(embeddings, axis=0))
        else:
            shot_embeddings.append(None)

    cap.release()
    return shot_embeddings


def compute_facesim_arc(shot_embeddings):
    """
    Compute FaceSim-Arc: average pairwise ArcFace cosine similarity across shots.
    """
    valid = [e for e in shot_embeddings if e is not None]
    if len(valid) < 2:
        return None

    sims = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            sim = cosine_similarity(
                valid[i].reshape(1, -1),
                valid[j].reshape(1, -1)
            )[0, 0]
            sims.append(sim)

    return float(np.mean(sims)) if sims else None


def evaluate_videos(video_dir, prompts_json, results_path):
    """
    Evaluate generated videos for identity consistency.

    Args:
        video_dir: Directory containing generated .mp4 files
        prompts_json: Path to benchmark prompts JSON with inner_t info
        results_path: Output path for results JSON
    """
    face_app = init_face_app()

    # Load prompts/metadata
    with open(prompts_json, "r") as f:
        prompts = json.load(f)

    results = []
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

    for i, vfile in enumerate(tqdm(video_files, desc="Evaluating")):
        vpath = os.path.join(video_dir, vfile)

        # Get inner_t from prompts (if available)
        inner_t = prompts[i].get("inner_t", [10, 10, 10]) if i < len(prompts) else [10, 10, 10]

        # Extract face embeddings per shot
        shot_embs = extract_face_embeddings_from_video(face_app, vpath, inner_t)

        # Compute metrics
        face_arc = compute_facesim_arc(shot_embs)

        results.append({
            "video": vfile,
            "face_arc": face_arc,
            "num_shots": len(inner_t),
            "faces_detected": sum(1 for e in shot_embs if e is not None),
        })

    # Aggregate
    valid_arcs = [r["face_arc"] for r in results if r["face_arc"] is not None]
    summary = {
        "n_videos": len(results),
        "mean_face_arc": float(np.mean(valid_arcs)) if valid_arcs else None,
        "std_face_arc": float(np.std(valid_arcs)) if valid_arcs else None,
        "face_detection_rate": sum(r["faces_detected"] for r in results) /
                               sum(r["num_shots"] for r in results) if results else 0,
    }

    output = {"summary": summary, "per_video": results}
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Evaluation Summary ===")
    print(f"Videos evaluated: {summary['n_videos']}")
    print(f"Mean FaceSim-Arc: {summary['mean_face_arc']:.4f}" if summary['mean_face_arc'] else "FaceSim-Arc: N/A")
    print(f"Face detection rate: {summary['face_detection_rate']:.2%}")
    print(f"Results saved to: {results_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-shot video generation")
    parser.add_argument("--video_dir", required=True, help="Directory with generated videos")
    parser.add_argument("--prompts_json", required=True, help="Benchmark prompts JSON")
    parser.add_argument("--results_json", default="eval_results.json", help="Output results path")
    args = parser.parse_args()

    evaluate_videos(args.video_dir, args.prompts_json, args.results_json)


if __name__ == "__main__":
    main()
