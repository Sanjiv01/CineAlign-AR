# Actor-Centric Data Collection

### Movie Video Data

--- 

### Actor Image Data

#### Step1: Download IMDB actor images
This script download the actor image data from [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).

Option1: Full image (without cropping)
```sh
chmod +x ./imdb_scripts/download_imdb_full_image.sh
./imdb_scripts/download_imdb_full_image.sh
```

Option2: Face only image (with cropping)
```sh
chmod +x ./imdb_scripts/download_imdb_faceonly_image.sh
./imdb_scripts/download_imdb_faceonly_image.sh
```

#### Step2: Data Preprocess & Getting Features
This will automately generate the features of each actor.
```sh
python imdb_scripts/imdb_images_data_preprocess.py
python imdb_scripts/imdb_features_extra.py \
    --actor_root imdb_images/actor_images \
    --save_dir imdb_images/actor_features \
    --model_name dinov3-convnext-tiny-pretrain-lvd1689m
```

