"""
VLM Captioning Module for Multi-Shot Video Dataset.

Generates structured text descriptions for video clips using a Vision-Language Model.
Each caption follows EchoShot's format: camera motion, ethnicity, gender, age,
hairstyle, apparel, expression, action, background, lighting.

Usage:
    python vlm_captioner.py --train_json dataset_out/train.json --data_root dataset_out/clips
"""

import os
import json
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm


# Caption prompt template
CAPTION_PROMPT = """Describe this video clip in detail. Your description must include ALL of the following elements in order:
1. Camera motion (e.g., static shot, pan, zoom, tracking)
2. Shot type (e.g., close-up, medium shot, wide shot)
3. Person's ethnicity
4. Person's gender
5. Person's approximate age range
6. Hairstyle (color, length, style)
7. Clothing/apparel description
8. Facial expression and emotion
9. Physical actions and movements
10. Background/scene description
11. Lighting conditions

Format your response as a single continuous paragraph."""


def extract_keyframes(video_path, num_frames=5):
    """Extract evenly-spaced keyframes from a video clip."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    cap.release()
    return frames


def load_vlm(model_name="llava-hf/llava-v1.6-mistral-7b-hf"):
    """Load a Vision-Language Model for captioning."""
    from transformers import AutoProcessor, LlavaNextForConditionalGeneration

    print(f"[VLM] Loading {model_name}...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def caption_clip(model, processor, frames, prompt=CAPTION_PROMPT):
    """Generate a caption for a video clip given its keyframes."""
    from PIL import Image

    pil_frames = [Image.fromarray(f) for f in frames]

    conversation = [{
        "role": "user",
        "content": [{"type": "image"} for _ in pil_frames] +
                   [{"type": "text", "text": prompt}]
    }]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=pil_frames, text=text_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=300)

    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # Extract the assistant's response (after the prompt)
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()

    return response


def main():
    parser = argparse.ArgumentParser(description="VLM Captioning for multi-shot video dataset")
    parser.add_argument("--train_json", required=True, help="Path to train.json")
    parser.add_argument("--data_root", required=True, help="Root directory containing video clips")
    parser.add_argument("--output_json", default=None, help="Output path (default: overwrite train_json)")
    parser.add_argument("--model_name", default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--num_keyframes", type=int, default=5)
    parser.add_argument("--batch_start", type=int, default=0, help="Start index for batch processing")
    parser.add_argument("--batch_end", type=int, default=-1, help="End index (-1 = all)")
    args = parser.parse_args()

    with open(args.train_json, "r") as f:
        train_data = json.load(f)

    output_path = args.output_json or args.train_json

    model, processor = load_vlm(args.model_name)

    end = args.batch_end if args.batch_end > 0 else len(train_data)
    for idx in tqdm(range(args.batch_start, end), desc="Captioning"):
        entry = train_data[idx]
        new_captions = []

        for shot_path in entry["shots"]:
            full_path = os.path.join(args.data_root, shot_path)
            if not os.path.exists(full_path):
                print(f"[WARN] Video not found: {full_path}")
                new_captions.append("")
                continue

            frames = extract_keyframes(full_path, args.num_keyframes)
            if not frames:
                print(f"[WARN] No frames extracted: {full_path}")
                new_captions.append("")
                continue

            caption = caption_clip(model, processor, frames)
            new_captions.append(caption)

        entry["cap_list"] = new_captions

        # Save periodically
        if (idx + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(train_data, f, indent=2, ensure_ascii=False)

    # Final save
    with open(output_path, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Captioned {end - args.batch_start} entries -> {output_path}")


if __name__ == "__main__":
    main()
