from utils import (
    INPUT_DIR,
    OUT_ROOT,
    CLIP_DIR,
    ensure_dir,
    load_existing_train_json,
    save_train_json,
    process_single_video,
    compute_next_person_idx_from_existing,
)
import os


def main():
    ensure_dir(OUT_ROOT)
    ensure_dir(CLIP_DIR)

    train_entries = load_existing_train_json()
    # Determine next global person index by scanning existing filenames
    global_person_idx = compute_next_person_idx_from_existing(train_entries)

    # collect all input videos
    video_files = []
    for fname in sorted(os.listdir(INPUT_DIR)):
        if fname.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
            video_files.append(os.path.join(INPUT_DIR, fname))

    if len(video_files) == 0:
        print(f"No videos found in {INPUT_DIR}")
        return

    print(f"Found {len(video_files)} videos to process.\n")

    for vpath in video_files:
        global_person_idx = process_single_video(vpath, global_person_idx, train_entries)
        # save json after each video
        save_train_json(train_entries)
        print(f"[JSON] Updated train.json with {len(train_entries)} total training samples.\n")

    print("=== DONE ===")
    print(f"Clips saved in: {CLIP_DIR}")
    print(f"train.json written to: {os.path.join(OUT_ROOT, 'train.json')}")
    print(f"Total training samples: {len(train_entries)}")


if __name__ == "__main__":
    main()
