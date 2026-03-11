import os
import cv2
import json
import math
import random
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG
# ============================================================

# Paths are relative to the working directory (dataset_generation/processing/)
# or can be overridden via environment variables
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

INPUT_DIR = os.environ.get("DATASET_INPUT_DIR",
                           os.path.join(_PROJECT_ROOT, "dataset_in"))
OUT_ROOT = os.environ.get("DATASET_OUTPUT_DIR",
                          os.path.join(_PROJECT_ROOT, "dataset_out"))
CLIP_DIR = os.path.join(OUT_ROOT, "clips")
TRAIN_JSON_PATH = os.path.join(OUT_ROOT, "train.json")

TARGET_W = 832                  # final video resolution
TARGET_H = 480

SCENE_THRESHOLD = 27            # PySceneDetect sensitivity
MIN_SCENE_FRAMES = 20           # ignore very short scenes

# Look for YOLO weights in project models/ first, then fall back to cwd
_YOLO_PROJECT = os.path.join(_PROJECT_ROOT, "models", "yolov8m.pt")
YOLO_MODEL = _YOLO_PROJECT if os.path.isfile(_YOLO_PROJECT) else "yolov8m.pt"
YOLO_CONF = 0.40
PERSON_CLASS = 0

NUM_WORKERS = 10                # threads for scene analysis

# IDENTITY thresholds
ID_SIM_THRESHOLD = 0.50         # shot → identity assignment (cosine sim)
MERGE_ID_THRESHOLD = 0.75       # second-pass identity merging

# Straight-face thresholds (degrees)
YAW_THRESH = 15
PITCH_THRESH = 15
ROLL_THRESH = 20

# Number of keyframes per scene used to build shot embedding
KEYFRAMES_PER_SCENE = 7         # more = more stable embeddings

# Shot/frame constraints
MIN_SHOTS_PER_ID = 2            # minimum number of raw shots for an identity
MIN_SHOTS_PER_DATA = 2          # min shots in one training sample
MAX_SHOTS_PER_DATA = 4          # max shots in one training sample
TARGET_FRAMES_PER_ID = 125      # total frames across all shots in one sample
MIN_FRAMES_PER_SHOT = 20        # minimum frames a single shot is allowed to have

MAX_DATA_PER_ID = 10            # how many different samples to generate per identity (upper bound)
MAX_DATA_ATTEMPTS_PER_ID = 40   # how many attempts we allow when sampling clip combinations


# ============================================================
# GLOBAL MODELS
# ============================================================

print("\n[INIT] Loading YOLO (GPU if available)...")
try:
    device = "cuda"
    YOLO_MODEL_OBJ = YOLO(YOLO_MODEL)
    YOLO_MODEL_OBJ.to(device)
    _ = YOLO_MODEL_OBJ.predict(np.zeros((384, 640, 3), dtype=np.uint8))
except Exception:
    print("[WARN] CUDA not available → using CPU")
    device = "cpu"
    YOLO_MODEL_OBJ = YOLO(YOLO_MODEL)
print(f"[INIT] YOLO running on: {device}")

print("[INIT] Loading InsightFace (CPU)...")
FACE_APP = FaceAnalysis(name="buffalo_l")
FACE_APP.prepare(ctx_id=-1, det_size=(256, 256))   # ctx_id=-1 → CPU
print("[INIT] InsightFace ready.\n")


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def resize_to_target(frame):
    """Resize + pad to exactly TARGET_W x TARGET_H without distortion."""
    h, w = frame.shape[:2]
    scale = min(TARGET_W / w, TARGET_H / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))
    pad_w = TARGET_W - new_w
    pad_h = TARGET_H - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def is_straight_face(face,
                     yaw_thresh=YAW_THRESH,
                     pitch_thresh=PITCH_THRESH,
                     roll_thresh=ROLL_THRESH):
    """
    Check if face is roughly frontal using InsightFace's pose:
    face.pose = (yaw, pitch, roll)
    """
    if not hasattr(face, "pose") or face.pose is None:
        return False
    yaw, pitch, roll = face.pose
    return (abs(yaw) < yaw_thresh and
            abs(pitch) < pitch_thresh and
            abs(roll) < roll_thresh)


# ============================================================
# SCENE / DETECTION HELPERS
# ============================================================

def detect_scenes(video_path):
    """Use PySceneDetect to find shot boundaries."""
    vm = VideoManager([video_path])
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=SCENE_THRESHOLD))
    vm.start()
    sm.detect_scenes(frame_source=vm)
    scenes = sm.get_scene_list()
    vm.release()

    if len(scenes) == 0:
        return None

    return [(int(a.get_frames()), int(b.get_frames())) for a, b in scenes]


def get_person_boxes(frame):
    """
    Run YOLO on full frame and return person boxes only (COCO class 0).
    """
    results = YOLO_MODEL_OBJ(frame, device=device, conf=YOLO_CONF)[0]
    boxes = []
    if results.boxes is None:
        return boxes

    for det in results.boxes:
        cls_id = int(det.cls[0])
        if cls_id != PERSON_CLASS:
            continue
        x1, y1, x2, y2 = det.xyxy[0].tolist()
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def get_face_embedding_from_frame(frame):
    """
    Use InsightFace on full frame:
      - require exactly 1 face,
      - require frontal pose (straight face),
      - return embedding and bounding box.
    """
    faces = FACE_APP.get(frame)
    if len(faces) != 1:
        return None, None, None

    face = faces[0]
    if not is_straight_face(face):
        return None, None, None

    emb = face.embedding.astype(np.float32)
    x1, y1, x2, y2 = map(int, face.bbox)
    return emb, (x1, y1, x2, y2), face


# ============================================================
# CAPTION HELPERS
# ============================================================

def classify_camera_motion(gray_frames):
    """Very simple camera motion heuristic using frame differences."""
    if len(gray_frames) < 2:
        return "still-motion camera"

    diffs = []
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
        diffs.append(np.mean(diff))
    mean_diff = float(np.mean(diffs))

    if mean_diff < 3:
        return "static camera"
    elif mean_diff < 8:
        return "slow-moving camera"
    else:
        return "hand-held camera"


def describe_person(face):
    """ATTRIBUTE string from age + gender."""
    gender = getattr(face, "gender", -1)
    age = getattr(face, "age", 30)

    if gender == 0:
        gender_str = "female"
    elif gender == 1:
        gender_str = "male"
    else:
        gender_str = "person"

    if age < 25:
        age_str = "young"
    elif age < 45:
        age_str = "middle-aged"
    else:
        age_str = "older"

    return f"{age_str} {gender_str}"


def describe_hairstyle(face_box):
    """Rough hair style based on head box aspect ratio."""
    x1, y1, x2, y2 = face_box
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    ratio = h / float(w)
    if ratio > 1.3:
        length_str = "short hair"
    else:
        length_str = "medium-length hair"
    return f"{length_str}"


COCO_LABELS = {
    0: "person",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    62: "tv",
    63: "laptop",
    64: "mouse",
    66: "keyboard",
    67: "cell phone",
    71: "toilet",
}


def describe_apparel_and_background(frame, face_box):
    """
    Use YOLO detections to infer simple apparel and background text.
    """
    results = YOLO_MODEL_OBJ(frame, device=device, conf=0.35)[0]
    apparel_items = []
    bg_items = []

    if results.boxes is None:
        return "simple clothing", "plain background"

    fx1, fy1, fx2, fy2 = face_box
    fh = fy2 - fy1

    for det in results.boxes:
        cls_id = int(det.cls[0])
        label = COCO_LABELS.get(cls_id, None)
        if label is None or cls_id == 0:
            continue

        x1, y1, x2, y2 = det.xyxy[0].tolist()
        cy = (y1 + y2) / 2.0

        if cy > fy2 + fh:  # below face → likely torso / apparel
            apparel_items.append(label)
        else:
            bg_items.append(label)

    apparel_text = "simple clothing"
    if "tie" in apparel_items:
        apparel_text = "formal outfit with a tie"
    elif any(x in apparel_items for x in ["backpack", "handbag", "suitcase"]):
        apparel_text = "casual outfit with accessories"

    if len(bg_items) == 0:
        bg_text = "simple indoor background"
    else:
        bg_text = "background with " + ", ".join(sorted(set(bg_items))[:3])

    return apparel_text, bg_text


def describe_expression_and_behavior(sample_faces, face_centers):
    """Use landmarks and motion to get expression + behavior."""
    expression_str = "neutral expression"
    if len(sample_faces) > 0 and hasattr(sample_faces[0], "landmark_2d_106"):
        lm = sample_faces[0].landmark_2d_106
        upper = lm[88]
        lower = lm[95]
        dist = np.linalg.norm(upper - lower)
        if dist > 5:
            expression_str = "slight smile"
        if dist > 9:
            expression_str = "big smile"

    behavior_str = "standing still"
    if len(face_centers) >= 2:
        start = np.array(face_centers[0], dtype=np.float32)
        end = np.array(face_centers[-1], dtype=np.float32)
        dist = np.linalg.norm(end - start)
        if dist > 20:
            behavior_str = "walking or moving"
        elif dist > 8:
            behavior_str = "subtle movement"

    return expression_str, behavior_str


def describe_lighting(gray_frames):
    """Very simple brightness estimate."""
    if len(gray_frames) == 0:
        return "even lighting"

    mean_brightness = float(np.mean(gray_frames))
    if mean_brightness < 70:
        return "dim lighting"
    elif mean_brightness > 180:
        return "bright lighting"
    else:
        return "soft, even lighting"


def build_caption_from_samples(gray_frames, rgb_frames, face_boxes, faces):
    """
    Build caption: 
    “This is a [CAMERA] of a [ATTRIBUTE].[HAIRSTYLE].[APPAREL].[EXPRESSION].[BEHAVIOR].[BACKGROUND].[LIGHTING].”
    """
    if len(rgb_frames) == 0 or len(face_boxes) == 0 or len(faces) == 0:
        camera_str = classify_camera_motion(gray_frames)
        lighting_str = describe_lighting(gray_frames)
        return (
            f"This is a {camera_str} of a person."
            f"neutral hairstyle."
            f"simple clothing."
            f"neutral expression."
            f"standing still."
            f"plain background."
            f"{lighting_str}."
        )

    mid_idx = len(rgb_frames) // 2
    mid_frame = rgb_frames[mid_idx]
    mid_box = face_boxes[mid_idx]
    mid_face = faces[mid_idx]

    camera_str = classify_camera_motion(gray_frames)
    attr_str = describe_person(mid_face)
    hair_str = describe_hairstyle(mid_box)
    apparel_str, bg_str = describe_apparel_and_background(mid_frame, mid_box)
    expr_str, behav_str = describe_expression_and_behavior(
        faces, [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in face_boxes]
    )
    lighting_str = describe_lighting(gray_frames)

    return (
        f"This is a {camera_str} of a {attr_str}."
        f"{hair_str}."
        f"{apparel_str}."
        f"{expr_str}."
        f"{behav_str}."
        f"{bg_str}."
        f"{lighting_str}."
    )


# ============================================================
# ANALYZE ONE SCENE → ONE SHOT (IF VALID)
# ============================================================

def analyze_scene(scene_idx, start_frame, end_frame, video_path):
    """
    For one scene:
      - sample multiple keyframes,
      - require exactly 1 person + 1 straight face in several frames,
      - average embeddings & face boxes,
      - return 'shot' dict or None.
    """
    cap = cv2.VideoCapture(video_path)
    length = end_frame - start_frame
    if length < MIN_SCENE_FRAMES:
        cap.release()
        return None

    if KEYFRAMES_PER_SCENE >= length:
        frame_indices = list(range(start_frame, end_frame))
    else:
        step = max(length // KEYFRAMES_PER_SCENE, 1)
        frame_indices = list(range(start_frame, end_frame, step))[:KEYFRAMES_PER_SCENE]

    embeddings = []
    face_boxes = []

    for fidx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        person_boxes = get_person_boxes(frame)
        if len(person_boxes) != 1:
            continue

        emb, fbox, _face = get_face_embedding_from_frame(frame)
        if emb is None or fbox is None:
            continue

        embeddings.append(emb)
        face_boxes.append(fbox)

    cap.release()

    if len(embeddings) < 3:
        return None

    emb_avg = np.mean(np.stack(embeddings, axis=0), axis=0)
    xs1, ys1, xs2, ys2 = zip(*face_boxes)
    avg_box = (
        int(np.mean(xs1)),
        int(np.mean(ys1)),
        int(np.mean(xs2)),
        int(np.mean(ys2)),
    )

    return {
        "scene_idx": scene_idx,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "length": length,
        "embedding": emb_avg,
        "box": avg_box,
    }


# ============================================================
# SHOT → IDENTITY CLUSTERING
# ============================================================

def assign_shots_to_identities(shots):
    """
    Greedy clustering of shots into identities using cosine similarity
    of shot-level embeddings.
    """
    identities = {}  # id -> {"shots": [], "emb_sum": ..., "count": ...}
    next_id = 0

    for shot in shots:
        emb = shot["embedding"].reshape(1, -1)

        if len(identities) == 0:
            identities[next_id] = {
                "shots": [shot],
                "emb_sum": shot["embedding"].copy(),
                "count": 1,
            }
            next_id += 1
            continue

        best_sim = -1.0
        best_id = None

        for pid, data in identities.items():
            mean_emb = data["emb_sum"] / data["count"]
            sim = cosine_similarity(emb, mean_emb.reshape(1, -1))[0, 0]
            if sim > best_sim:
                best_sim = sim
                best_id = pid

        if best_sim >= ID_SIM_THRESHOLD:
            identities[best_id]["shots"].append(shot)
            identities[best_id]["emb_sum"] += shot["embedding"]
            identities[best_id]["count"] += 1
        else:
            identities[next_id] = {
                "shots": [shot],
                "emb_sum": shot["embedding"].copy(),
                "count": 1,
            }
            next_id += 1

    return identities


def merge_identities_by_mean_embedding(identities):
    """
    Second-pass global merging:
      - compute mean embedding per identity,
      - merge IDs whose mean embeddings are very close (cos sim >= MERGE_ID_THRESHOLD).
    """
    if len(identities) <= 1:
        return identities

    ids = sorted(identities.keys())
    means = {}
    for pid in ids:
        data = identities[pid]
        means[pid] = (data["emb_sum"] / data["count"]).astype(np.float32)

    parent = {pid: pid for pid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_i, id_j = ids[i], ids[j]
            mi = means[id_i].reshape(1, -1)
            mj = means[id_j].reshape(1, -1)
            sim = cosine_similarity(mi, mj)[0, 0]
            if sim >= MERGE_ID_THRESHOLD:
                union(id_i, id_j)

    new_identities = {}
    for old_id in ids:
        root = find(old_id)
        if root not in new_identities:
            new_identities[root] = {
                "shots": [],
                "emb_sum": np.zeros_like(means[root]),
                "count": 0,
            }
        new_identities[root]["shots"].extend(identities[old_id]["shots"])
        new_identities[root]["emb_sum"] += identities[old_id]["emb_sum"]
        new_identities[root]["count"] += identities[old_id]["count"]

    final_identities = {}
    for new_idx, old_root in enumerate(sorted(new_identities.keys())):
        final_identities[new_idx] = new_identities[old_root]

    return final_identities


# ============================================================
# NEW TRIMMING & DATA-GENERATION LOGIC
# ============================================================

def _compute_weighted_trims(shots):
    """
    Given a list of shot dicts (each with 'length', 'start_frame', 'end_frame'),
    compute how many frames to drop per shot using the weighted strategy:

        frames_to_drop_i = weight_i * frames_to_drop_total

    and then randomly split that into a left and right crop per shot.

    Returns:
        list of dicts with keys:
          'shot'         -> original shot dict
          'start_frame'  -> trimmed start frame (inclusive)
          'end_frame'    -> trimmed end frame (exclusive)
    or None if not feasible.
    """
    lengths = [s["length"] for s in shots]
    total_len = sum(lengths)

    if total_len < TARGET_FRAMES_PER_ID:
        return None

    frames_to_drop = total_len - TARGET_FRAMES_PER_ID
    if frames_to_drop == 0:
        # nothing to trim, but still need to check shot lengths
        for s in shots:
            if s["length"] < MIN_FRAMES_PER_SHOT:
                return None
        return [
            {
                "shot": s,
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
            }
            for s in shots
        ]

    # Proportional allocation (floats)
    float_drops = []
    for L in lengths:
        w = L / total_len
        float_drops.append(w * frames_to_drop)

    # Round while preserving total sum
    int_drops = [int(math.floor(x)) for x in float_drops]
    remainder = frames_to_drop - sum(int_drops)
    # distribute remaining frames to the shots with largest fractional parts
    frac_parts = [x - math.floor(x) for x in float_drops]
    order = sorted(range(len(shots)), key=lambda i: frac_parts[i], reverse=True)
    for idx in order:
        if remainder <= 0:
            break
        int_drops[idx] += 1
        remainder -= 1

    # Now int_drops sum to frames_to_drop exactly
    trimmed_segments = []
    for s, L, drop in zip(shots, lengths, int_drops):
        if drop < 0:
            return None

        # Ensure we don't trim below MIN_FRAMES_PER_SHOT
        max_drop_allowed = max(0, L - MIN_FRAMES_PER_SHOT)
        if drop > max_drop_allowed:
            # Not feasible with this combination
            return None

        # Random left/right split
        left_drop = random.randint(0, drop)
        right_drop = drop - left_drop

        start_f = s["start_frame"] + left_drop
        end_f = s["end_frame"] - right_drop
        if end_f <= start_f:
            return None

        if (end_f - start_f) < MIN_FRAMES_PER_SHOT:
            return None

        trimmed_segments.append(
            {
                "shot": s,
                "start_frame": start_f,
                "end_frame": end_f,
            }
        )

    # Final sanity check on total frames
    total_trimmed = sum(seg["end_frame"] - seg["start_frame"] for seg in trimmed_segments)
    if total_trimmed != TARGET_FRAMES_PER_ID:
        # Should not really happen, but guard anyway
        return None

    return trimmed_segments


def build_identity_clips(
    global_person_idx,
    identity_data,
    video_path,
    root_clip_dir,
):
    """
    NEW VERSION:
    For a given identity, we may have many valid shots.
    We will:
      - filter out very short shots
      - repeatedly sample 2–4 distinct shots to create multiple
        *training samples* for this identity
      - for each sample:
          * compute weighted trims so that total frames == TARGET_FRAMES_PER_ID
            while trimming only from the front/back of each shot
          * write one clip per trimmed shot with filenames:
            pXXX_dYY_sZZ.mp4
          * build captions and inner_t
      - return a list of JSON entries (one per training sample)
    """
    shots = sorted(identity_data["shots"], key=lambda s: s["start_frame"])
    shots = [s for s in shots if s["length"] >= MIN_FRAMES_PER_SHOT]

    if len(shots) < MIN_SHOTS_PER_ID:
        return []

    ensure_dir(root_clip_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    samples = []
    max_shots_for_data = min(MAX_SHOTS_PER_DATA, len(shots))

    # If we don't even have enough frames in total, bail early
    total_len_all_shots = sum(s["length"] for s in shots)
    if total_len_all_shots < TARGET_FRAMES_PER_ID:
        cap.release()
        return []

    attempts = 0
    sample_idx = 0

    while sample_idx < MAX_DATA_PER_ID and attempts < MAX_DATA_ATTEMPTS_PER_ID:
        attempts += 1

        num_shots = random.randint(MIN_SHOTS_PER_DATA, max_shots_for_data)
        chosen_shots = random.sample(shots, num_shots)

        trimmed_segments = _compute_weighted_trims(chosen_shots)
        if trimmed_segments is None:
            continue

        sample_idx += 1
        pid_str = f"p{global_person_idx:03d}"
        dataset_id_str = f"d{sample_idx:02d}"

        shots_paths = []
        caps = []
        inner_t = []
        total_written_frames = 0

        for shot_order, seg in enumerate(
            sorted(trimmed_segments, key=lambda s: s["start_frame"]),
            start=1,
        ):
            start_f = seg["start_frame"]
            end_f = seg["end_frame"]
            seg_len = end_f - start_f

            if seg_len < MIN_FRAMES_PER_SHOT:
                continue

            filename = f"{pid_str}_{dataset_id_str}_s{shot_order:02d}.mp4"
            out_path = os.path.join(root_clip_dir, filename)
            writer = cv2.VideoWriter(out_path, fourcc, fps, (TARGET_W, TARGET_H))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            written = 0

            sample_gray = []
            sample_rgb = []
            sample_face_boxes = []
            sample_faces = []

            for fidx in range(start_f, end_f):
                ret, frame = cap.read()
                if not ret:
                    break

                if (fidx - start_f) % 5 == 0 or fidx == start_f or fidx == end_f - 1:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sample_gray.append(cv2.resize(gray, (96, 54)))

                    emb, fbox, face = get_face_embedding_from_frame(frame)
                    if emb is not None and fbox is not None and face is not None:
                        sample_rgb.append(frame.copy())
                        sample_face_boxes.append(fbox)
                        sample_faces.append(face)

                frame_resized = resize_to_target(frame)
                writer.write(frame_resized)
                written += 1
                total_written_frames += 1

            writer.release()

            if written < MIN_FRAMES_PER_SHOT:
                if os.path.exists(out_path):
                    os.remove(out_path)
                continue

            caption = build_caption_from_samples(
                sample_gray, sample_rgb, sample_face_boxes, sample_faces
            )

            shots_paths.append(f"clips/{filename}")
            caps.append(caption)
            inner_t_value = ((written - 1) // 4) + 1
            inner_t.append(inner_t_value)

        # Final validation for this dataset
        if len(shots_paths) < MIN_SHOTS_PER_DATA:
            # remove any clips we wrote for this faulty dataset
            for rel_path in shots_paths:
                full_path = os.path.join(root_clip_dir, os.path.basename(rel_path))
                if os.path.exists(full_path):
                    os.remove(full_path)
            # decrement index so that the next sample reuses same dXX number
            sample_idx -= 1
            continue

        # total_written_frames *should* equal TARGET_FRAMES_PER_ID,
        # but in rare decode failures it may be slightly off.
        # We still enforce equality here, to keep the dataset clean.
        if total_written_frames != TARGET_FRAMES_PER_ID:
            for rel_path in shots_paths:
                full_path = os.path.join(root_clip_dir, os.path.basename(rel_path))
                if os.path.exists(full_path):
                    os.remove(full_path)
            sample_idx -= 1
            continue

        samples.append(
            {
                "shots": shots_paths,
                "cap_list": caps,
                "inner_t": inner_t,
            }
        )

    cap.release()
    return samples


# ============================================================
# PER-VIDEO PROCESSING
# ============================================================

def process_single_video(video_path, starting_person_idx, existing_entries):
    """
    Process one video:
      - detect scenes
      - build shots
      - cluster to identities + merge
      - build multiple clips & metadata per identity
      - append entries to existing_entries
    Returns new_next_person_idx.
    """
    print(f"\n[VIDEO] Processing: {video_path}")
    ensure_dir(CLIP_DIR)

    scenes = detect_scenes(video_path)
    if scenes is None:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scenes = [(0, total)]
        cap.release()

    scenes_filtered = [
        (i, s[0], s[1])
        for i, s in enumerate(scenes)
        if s[1] - s[0] >= MIN_SCENE_FRAMES
    ]
    print(f"[SCENES] {len(scenes_filtered)} scenes to analyze.")

    if len(scenes_filtered) == 0:
        print("[VIDEO] No usable scenes.")
        return starting_person_idx

    shots = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {
            ex.submit(analyze_scene, sidx, st, en, video_path): (sidx, st, en)
            for (sidx, st, en) in scenes_filtered
        }
        for fut in as_completed(futures):
            scene_info = futures[fut]
            try:
                shot = fut.result()
                if shot is not None:
                    shots.append(shot)
            except Exception as e:
                print(f"[WARN] Scene {scene_info} failed: {e}")

    print(f"[SHOTS] Valid single-person, straight-face shots: {len(shots)}")
    if len(shots) == 0:
        print("[VIDEO] No valid shots.")
        return starting_person_idx

    print("[IDENTITIES] First-pass clustering...")
    identities = assign_shots_to_identities(shots)
    print(f"[IDENTITIES] Got {len(identities)} identities before merging.")

    print("[IDENTITIES] Second-pass merging by mean embedding...")
    identities_merged = merge_identities_by_mean_embedding(identities)
    print(f"[IDENTITIES] Final identities after merge: {len(identities_merged)}")

    kept_identities = 0
    produced_samples = 0
    next_person_idx = starting_person_idx

    for _, data in identities_merged.items():
        samples = build_identity_clips(
            next_person_idx,
            data,
            video_path,
            CLIP_DIR,
        )
        if samples:
            existing_entries.extend(samples)
            kept_identities += 1
            produced_samples += len(samples)
            next_person_idx += 1

    print(f"[VIDEO] Identities kept from this video: {kept_identities}")
    print(f"[VIDEO] Total training samples produced from this video: {produced_samples}")
    return next_person_idx


# ============================================================
# TRAIN.JSON UTILITIES
# ============================================================

def load_existing_train_json():
    if not os.path.exists(TRAIN_JSON_PATH):
        return []
    try:
        with open(TRAIN_JSON_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def save_train_json(entries):
    ensure_dir(OUT_ROOT)
    with open(TRAIN_JSON_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def compute_next_person_idx_from_existing(entries):
    """
    Old versions of the script used len(entries) as the next person index,
    assuming one JSON entry per person.

    Now that we may have multiple entries per person, we instead parse the
    filenames (clips/pXXX_...) and take max(pXXX) + 1 as the starting index.
    """
    max_pid = -1
    for entry in entries:
        for shot_path in entry.get("shots", []):
            basename = os.path.basename(shot_path)
            # expected forms: pXXX_sYY.mp4  or  pXXX_dZZ_sYY.mp4
            if not basename.startswith("p"):
                continue
            try:
                pid_str = basename[1:4]
                pid_int = int(pid_str)
                max_pid = max(max_pid, pid_int)
            except Exception:
                continue

    return max_pid + 1 if max_pid >= 0 else 0
