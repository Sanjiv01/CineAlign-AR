import os
import shutil
import fire
from scipy.io import loadmat
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_metadata(meta_file: str, image_dir: str):
    data = loadmat(meta_file)
    actor_list = [str(i[0]) for i in data['imdb']['celeb_names'][0][0][0]]

    paths = data['imdb']['full_path'][0][0][0]
    names = data['imdb']['name'][0][0][0]
    
    actor_image_path_dict = {}
    
    for i in range(len(paths)):
        actor = str(names[i][0])
        path = os.path.join(image_dir, str(paths[i][0]))
        
        if actor not in actor_image_path_dict.keys():
            actor_image_path_dict[actor] = []

        if path not in actor_image_path_dict[actor]:
            actor_image_path_dict[actor].append(f"{path}")
        
    return actor_list, actor_image_path_dict


def move_actor_images(actor, image_paths, save_dir):
    actor_dir = os.path.join(save_dir, actor)
    os.makedirs(actor_dir, exist_ok=True)
    for img_path in image_paths:
        img_basename = os.path.basename(img_path)
        src = img_path
        dst = os.path.join(actor_dir, img_basename)
        try:
            shutil.move(src, dst)
        except Exception as e:
            print(f"[WARN] {src} -> {dst}: {e}")
    return actor


def main(image_dir: str, meta_file: str, save_dir: str, progress_file: str = "progress.log", num_workers: int = 8):
    os.makedirs(save_dir, exist_ok=True)
    
    actor_list, actor_image_path_dict = parse_metadata(meta_file, image_dir)

    completed_actors = set()
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            completed_actors = set(line.strip() for line in f if line.strip())
    print(f"[INFO] {len(completed_actors)} actors already done")

    remaining_actors = [a for a in actor_list if a not in completed_actors]
    print(f"[INFO] {len(remaining_actors)} actors left")

    with ThreadPoolExecutor(max_workers=num_workers) as executor, open(progress_file, "a") as log_f:
        futures = {
            executor.submit(move_actor_images, actor, actor_image_path_dict[actor], save_dir): actor
            for actor in remaining_actors
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Actors Data Processed", ncols=90):
            actor = futures[future]
            try:
                result_actor = future.result()
                log_f.write(result_actor + "\n")
                log_f.flush()  
            except Exception as e:
                print(f"[ERROR] Actor: {actor}: {e}")

    print("[ALL DONE]")
    
    
if __name__ == "__main__":
    fire.Fire(main)
