import os
import json
import ast
import re
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download

# === Config ===
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
cache_path = os.environ.get("MODEL_CACHE", os.path.join(os.path.dirname(__file__), "..", "models", "cache"))
csv_path = "./filtered_movies_with_cast.csv"
raw_csv_path = "./movie_data/movie.csv"
save_every = 20
batch_size = 10

# === Load model ===
print("Loading model...")
local_dir = snapshot_download(model_id, cache_dir=cache_path)
tokenizer = AutoTokenizer.from_pretrained(local_dir)
tokenizer.pad_token = tokenizer.eos_token  # Add pad_token to resolve padding error
model = AutoModelForCausalLM.from_pretrained(local_dir, torch_dtype=torch.bfloat16, device_map="auto")

# === Load data ===
movies_all = pd.read_csv(raw_csv_path, encoding="big5")

fix_indices = {
    15011: ' (1970)', 16532: ' (2010)', 22412: ' (2014)', 22836: ' (2014)',
    22951: ' (2011)', 24546: ' (1990)', 24556: ' (2002)', 24606: ' (2008)',
    24753: ' (2013)', 24810: ' (2014)', 24843: ' (1991)', 25130: ' (2015)',
    25329: ' (2010)', 25340: ' (2014)', 25391: ' (2003)', 25475: ' (1993)'
}
for idx, suffix in fix_indices.items():
    if idx in movies_all.index:
        movies_all.at[idx, "title"] += suffix

# extract year
year = []
title = movies_all["title"].tolist()
for idx, i in enumerate(title):
    try:
        k = int(i[-5:-1])
    except:
        try:
            k = int(i[-6:-2])
        except:
            print(f"[WARN] Failed to extract year at index {idx}: {i}")
            k = None
    year.append(k)

movies_all['year'] = year
movies_all = movies_all[movies_all["year"] > 1975].copy()

# === Load or init checkpoint ===
if os.path.exists(csv_path):
    processed_df = pd.read_csv(csv_path)
    done_titles = set(processed_df["title"])
    print(f"[Resume] Loaded {len(done_titles)} finished records.")
else:
    processed_df = pd.DataFrame(columns=["title", "year", "film_type", "main_actors"])
    done_titles = set()

pending_movies = movies_all[~movies_all["title"].isin(done_titles)].reset_index(drop=True)

# === Prompt + Parse ===
def make_prompt(title):
    return f"""<s>[INST] Answer concisely:
1. Is the movie "{title}" an animated film or a live-action film?
2. List the top 5 main actors or voice actors of the movie in a Python list format, using real actor names only. Do not include explanations or placeholders. Only output the list of names. [/INST]"""

def parse_response(text):
    # 移除 prompt 前綴，例如包含 [INST] 的部分
    if "[/INST]" in text:
        text = text.split("[/INST]", 1)[-1].strip()

    film_type = None
    actors = []

    # 判斷影片類型
    lower = text.lower()
    if "animated" in lower:
        film_type = "animated"
    elif "live-action" in lower or "live action" in lower:
        film_type = "live-action"

    # 嘗試擷取 list 格式區塊
    match = re.search(r"\[(.*?)\]", text, re.DOTALL)
    if match:
        raw = match.group(1)

        # 分離逗號並修正缺失引號
        names = [name.strip() for name in raw.split(",")]
        quoted_names = [f'"{name}"' if not (name.startswith('"') and name.endswith('"')) else name for name in names]

        try:
            actors = ast.literal_eval(f"[{', '.join(quoted_names)}]")
        except Exception as e:
            print(f"[ParseError] Failed to parse actor list: {e}")
            actors = []

    return film_type, actors

# === Batch processing ===
batch_data = []
for i in tqdm(range(0, len(pending_movies), batch_size)):
    batch = pending_movies.iloc[i:i + batch_size]
    prompts = [make_prompt(row["title"]) for _, row in batch.iterrows()]

    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=300)
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, (_, row) in enumerate(batch.iterrows()):
            film_type, actor_list = parse_response(responses[j])
            batch_data.append({
                "title": row["title"],
                "year": row["year"],
                "film_type": film_type,
                "main_actors": ", ".join(actor_list)
            })

    except Exception as e:
        print(f"[Error] Batch {i}-{i + batch_size}: {e}")
        continue

    if len(batch_data) >= save_every or i + batch_size >= len(pending_movies):
        processed_df = pd.concat([processed_df, pd.DataFrame(batch_data)], ignore_index=True)
        split_cols = processed_df["main_actors"].str.split(",", expand=True)

        # 確保每一欄都有命名
        for i in range(5):
            processed_df[f"actor_{i+1}"] = split_cols[i].str.strip() if i in split_cols.columns else ""

        processed_df.to_csv(csv_path, index=False)
        print(f"[Saved] {len(processed_df)} records → {csv_path}")
        batch_data = []

print("✅ All done.")
