"""
VLM Captioning Module for Multi-Shot Video Dataset.

Generates structured text descriptions for video clips using a Vision-Language Model.
Each caption follows EchoShot's format: camera motion, ethnicity, gender, age,
hairstyle, apparel, expression, action, background, lighting.

Features:
    - Resume support: skips already-captioned entries (checks for non-empty captions)
    - Periodic checkpointing: saves every N entries to avoid losing work
    - GPU memory management: explicit cache clearing between batches
    - Graceful error handling: logs failures and continues
    - Batch range support: process subsets for parallel jobs

Usage:
    # Caption all entries:
    python vlm_captioner.py --train_json dataset_out/train.json --data_root dataset_out

    # Caption entries 100-200 only (for parallel batch jobs):
    python vlm_captioner.py --train_json dataset_out/train.json --data_root dataset_out \\
        --batch_start 100 --batch_end 200

    # Use a different model:
    python vlm_captioner.py --train_json dataset_out/train.json --data_root dataset_out \\
        --model_name Qwen/Qwen2-VL-7B-Instruct

    # Force re-caption even if captions exist:
    python vlm_captioner.py --train_json dataset_out/train.json --data_root dataset_out --force
"""

import os
import gc
import json
import time
import argparse
import traceback

import torch
import cv2
import numpy as np
from tqdm import tqdm


# Caption prompt template — matches EchoShot format
CAPTION_PROMPT = """Describe this video clip in one detailed paragraph. Include ALL of the following:
1. Camera motion (static, pan, zoom, tracking, hand-held)
2. Shot type (close-up, medium, wide, over-the-shoulder)
3. Person's ethnicity and gender
4. Approximate age range
5. Hairstyle (color, length, style)
6. Clothing and accessories
7. Facial expression and emotion
8. Physical actions and movements
9. Background scene description
10. Lighting conditions (bright, dim, warm, cool, dramatic)

Write it as a single continuous paragraph, no bullet points."""

SAVE_INTERVAL = 10  # save checkpoint every N entries


def extract_keyframes(video_path, num_frames=5):
    """Extract evenly-spaced keyframes from a video clip."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    cap.release()
    return frames


class VLMCaptioner:
    """Manages VLM model lifecycle and caption generation."""

    def __init__(self, model_name="llava-hf/llava-v1.6-mistral-7b-hf", device_map="auto"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device_map = device_map

    def load(self):
        """Load the VLM model and processor."""
        if self.model is not None:
            return

        from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        print(f"[VLM] Loading {self.model_name}...")
        start = time.time()
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device_map,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        elapsed = time.time() - start
        print(f"[VLM] Loaded in {elapsed:.1f}s")

    def caption_single(self, frames, prompt=CAPTION_PROMPT, max_tokens=300):
        """Generate a caption for one clip given its keyframes."""
        from PIL import Image

        self.load()

        pil_frames = [Image.fromarray(f) for f in frames]

        conversation = [{
            "role": "user",
            "content": [{"type": "image"} for _ in pil_frames] +
                       [{"type": "text", "text": prompt}]
        }]

        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            images=pil_frames, text=text_prompt, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Extract assistant response (after prompt markers)
        for marker in ["[/INST]", "assistant\n", "ASSISTANT:"]:
            if marker in response:
                response = response.split(marker)[-1].strip()
                break

        # Clear GPU cache
        del inputs, output_ids
        torch.cuda.empty_cache()

        return response

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            torch.cuda.empty_cache()
            print("[VLM] Model unloaded.")


def entry_has_captions(entry):
    """Check if an entry already has non-empty VLM captions."""
    cap_list = entry.get("cap_list", [])
    if not cap_list:
        return False
    # Consider captioned if all captions are non-empty and longer than 50 chars
    # (heuristic captions from utils.py are typically shorter)
    return all(isinstance(c, str) and len(c) > 50 for c in cap_list)


def main():
    parser = argparse.ArgumentParser(description="VLM Captioning for multi-shot video dataset")
    parser.add_argument("--train_json", required=True, help="Path to train.json")
    parser.add_argument("--data_root", required=True, help="Root dir containing video clips")
    parser.add_argument("--output_json", default=None, help="Output path (default: overwrite train_json)")
    parser.add_argument("--model_name", default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--num_keyframes", type=int, default=5)
    parser.add_argument("--batch_start", type=int, default=0, help="Start index for batch processing")
    parser.add_argument("--batch_end", type=int, default=-1, help="End index (-1 = all)")
    parser.add_argument("--force", action="store_true", help="Re-caption even if captions exist")
    parser.add_argument("--save_interval", type=int, default=SAVE_INTERVAL,
                        help="Save checkpoint every N entries")
    args = parser.parse_args()

    # Load train.json
    with open(args.train_json, "r") as f:
        train_data = json.load(f)

    output_path = args.output_json or args.train_json
    end_idx = args.batch_end if args.batch_end > 0 else len(train_data)
    end_idx = min(end_idx, len(train_data))

    print(f"[INFO] Total entries: {len(train_data)}")
    print(f"[INFO] Processing range: [{args.batch_start}, {end_idx})")
    print(f"[INFO] Output: {output_path}")

    # Count how many need captioning
    needs_caption = 0
    for idx in range(args.batch_start, end_idx):
        if args.force or not entry_has_captions(train_data[idx]):
            needs_caption += 1
    print(f"[INFO] Entries needing captions: {needs_caption}")

    if needs_caption == 0:
        print("[DONE] All entries already have captions. Use --force to re-caption.")
        return

    # Initialize captioner
    captioner = VLMCaptioner(model_name=args.model_name)

    captioned_count = 0
    failed_count = 0
    last_save = time.time()

    for idx in tqdm(range(args.batch_start, end_idx), desc="Captioning"):
        entry = train_data[idx]

        # Skip if already captioned (unless --force)
        if not args.force and entry_has_captions(entry):
            continue

        new_captions = []
        entry_ok = True

        for shot_path in entry["shots"]:
            full_path = os.path.join(args.data_root, shot_path)
            if not os.path.exists(full_path):
                print(f"\n[WARN] Video not found: {full_path}")
                new_captions.append(entry.get("cap_list", [""])[len(new_captions)]
                                    if len(entry.get("cap_list", [])) > len(new_captions) else "")
                entry_ok = False
                continue

            try:
                frames = extract_keyframes(full_path, args.num_keyframes)
                if not frames:
                    print(f"\n[WARN] No frames extracted: {full_path}")
                    new_captions.append("")
                    entry_ok = False
                    continue

                caption = captioner.caption_single(frames)
                new_captions.append(caption)

            except torch.cuda.OutOfMemoryError:
                print(f"\n[ERROR] GPU OOM on {full_path}. Clearing cache and skipping...")
                torch.cuda.empty_cache()
                gc.collect()
                new_captions.append("")
                entry_ok = False

            except Exception as e:
                print(f"\n[ERROR] Failed to caption {full_path}: {e}")
                traceback.print_exc()
                new_captions.append("")
                entry_ok = False

        # Update captions
        entry["cap_list"] = new_captions
        captioned_count += 1
        if not entry_ok:
            failed_count += 1

        # Periodic save
        if captioned_count % args.save_interval == 0:
            with open(output_path, "w") as f:
                json.dump(train_data, f, indent=2, ensure_ascii=False)
            elapsed = time.time() - last_save
            print(f"\n[SAVE] Checkpoint at entry {idx} ({captioned_count} done, {elapsed:.0f}s since last save)")
            last_save = time.time()

    # Final save
    with open(output_path, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    # Cleanup
    captioner.unload()

    print(f"\n{'='*60}")
    print(f"[DONE] VLM Captioning Complete")
    print(f"  Captioned: {captioned_count}")
    print(f"  Failed:    {failed_count}")
    print(f"  Output:    {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
