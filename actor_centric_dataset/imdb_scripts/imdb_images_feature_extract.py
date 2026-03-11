import os
import torch
import fire
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm


def normalize_actor_name(name: str) -> str:
    return name.replace("'", "").lower()


def unique_save_path(save_dir: str, base_name: str, ext: str = ".pt") -> str:
    path = os.path.join(save_dir, f"{base_name}{ext}")
    if not os.path.exists(path):
        return path
    k = 1
    while True:
        cand = os.path.join(save_dir, f"{base_name}_{k}{ext}")
        if not os.path.exists(cand):
            return cand
        k += 1

    return

def extract_features_batch(model, processor, image_paths, device):
    images = []
    valid_paths = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            valid_paths.append(p)
        except Exception as e:
            print(f"[WARN] Failed to open {p}: {e}")

    if not images:
        return None

    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.pooler_output.cpu()
    

def main(
    actor_root: str = "imdb_images/actor_images",
    save_dir: str = "imdb_images/actor_features",
    model_name: str = "dinov3-vitl16-pretrain-lvd1689m",
    batch_size: int = 16,
    num_workers: int = 4,
    log_file: str = "logs/imdb_features_processed.log",
):
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processed = set()
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            processed = set(line.strip() for line in f if line.strip())
        print(f"[INFO] Loaded {len(processed)} processed actors from log.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = f"facebook/{model_name}"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    
    all_actor_dirs = [
        d for d in sorted(os.listdir(actor_root))
        if os.path.isdir(os.path.join(actor_root, d))
    ]
    
    actor_pairs = [(a, normalize_actor_name(a)) for a in all_actor_dirs]
    to_process = [(raw, norm) for (raw, norm) in actor_pairs if norm not in processed]

    print(f"[INFO] Total actors: {len(all_actor_dirs)} | To process: {len(to_process)}")

    with tqdm(total=len(to_process), desc="Actors", unit="actor") as pbar:
        for actor_name, actor_name_norm in to_process:
            actor_path = os.path.join(actor_root, actor_name)

            all_images = [
                os.path.join(actor_path, f)
                for f in os.listdir(actor_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]

            if not all_images:
                print(f"[WARN] No valid images for {actor_name}")
                with open(log_file, "a") as f:
                    f.write(actor_name_norm + "\n")
                pbar.update(1)
                continue

            feats = []
            for i in range(0, len(all_images), batch_size):
                batch_paths = all_images[i : i + batch_size]
                batch_feats = extract_features_batch(model, processor, batch_paths, device)
                if batch_feats is not None:
                    feats.append(batch_feats)

            if feats:
                actor_feat = torch.cat(feats, dim=0).mean(dim=0)
                save_path = unique_save_path(save_dir, actor_name_norm, ext=".pt")
                torch.save(actor_feat, save_path)
                # print(f"[SAVED] {actor_name} → {save_path}")
                with open(log_file, "a") as f:
                    f.write(actor_name_norm + "\n")
            else:
                print(f"[WARN] No valid features extracted for {actor_name}")
                with open(log_file, "a") as f:
                    f.write(actor_name_norm + "\n")

            pbar.update(1)

    return

            
if __name__ == "__main__":
    fire.Fire(main)
