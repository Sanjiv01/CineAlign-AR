from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from typing import List
from PIL import Image
import torch

def generate_multishot_story_and_captions(three_video_frames: List[List[Image.Image]], 
                                          model_name="llava-hf/llava-v1.6-mistral-7b-hf",
                                          max_tokens=100):
    """
    Args:
        three_video_frames: List of 3 elements, each a list of 3–5 PIL Images (frames) from one video
        model_name: HF model repo
        max_tokens: max tokens to generate for each response
    Returns:
        A dict with keys "storyline" and "captions", where:
            - "storyline": a string
            - "captions": list of 3 strings (one per clip)
    """
    assert len(three_video_frames) == 3, "Expecting exactly 3 video clips"
    all_images = sum(three_video_frames, [])  # flatten all images
    image_counts = [len(frames) for frames in three_video_frames]
    
    # Load model & processor
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # === Build Conversations ===
    conversation = []

    # Part 1: Storyline
    conversation.append({
        "role": "user",
        "content": [{"type": "image"} for _ in all_images] + 
                   [{"type": "text", "text": "Generate a storyline that connects all these video clips."}]
    })

    # Part 2: Per-clip caption
    offset = 0
    for idx, count in enumerate(image_counts):
        conversation.append({
            "role": "user",
            "content": [{"type": "image"} for _ in range(count)] +
                       [{"type": "text", "text": f"What happens in clip {idx+1}?"}]
        })
        offset += count

    # === Encode Inputs ===
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=all_images, text=prompt, return_tensors="pt").to(model.device)

    # === Generate ===
    out_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    results = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # === Parse Results ===
    storyline = results[0]
    captions = results[1:4] if len(results) >= 4 else []

    return {"storyline": storyline, "captions": captions}