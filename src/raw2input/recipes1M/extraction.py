import json
import os
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

import sys
sys.path.insert(0, os.getcwd()) # for external execution

from settings.config import RECIPES1M_PATH, RAW_RECIPES1M_PATH, VAL_SIZE, TEST_SIZE, IMG_STATS_FILENAME, METADATA_FILENAME, PROJECT_PATH
from settings.commons import tokenize_category



BASE_PATH = RECIPES1M_PATH


def trunk_str(string, max_len=15, char="_"):
    if len(string) <= max_len:
        return string
    
    trunk_id = string[:max_len].rfind(char)
    trunk_id = max_len if trunk_id == -1 else trunk_id
    
    return string[:trunk_id]

def download_img(url, save_path):
    if not url.startswith("https://"):
        url = "https://" + url    
    try:
        img_data = requests.get(url).content
        
        with open(save_path, "wb") as handler:
            handler.write(img_data)
    except Exception as e:
        # print(f"Error downloading image: {e}")
        raise FileNotFoundError

N_SAMPLES = 100000
seed = 42

print("starting")
recipes_raw = pd.read_csv(os.path.join(RAW_RECIPES1M_PATH, 'full_dataset.csv'), index_col=0)

recipes = recipes_raw.sample(frac=1, random_state=seed).reset_index().drop(columns=["index"]).reset_index()
recipes['name'] = recipes["title"].str.replace(" ", "_").str.lower().str.strip("_").fillna("UNK")
recipes = recipes.rename(columns={"link": "src_url", "index": 'id', 'NER': 'ingredients_ok'}).drop(columns=["title", "source", 'directions'])
recipes['image'] = recipes['name'].apply(lambda name: trunk_str(name))
recipes['image'] = recipes['id'].astype(str) + "_____" + recipes['image'] + ".jpg"


pbar = tqdm(total=N_SAMPLES, desc="Downloading images", )
i = 17247
n_downloads = 10004
# recipes_downloaded = []

pbar.update(n_downloads)
while n_downloads < N_SAMPLES and i < len(recipes):
    recipe = recipes.iloc[i]
    try:
        download_img(recipe['src_url'], os.path.join(BASE_PATH, "download", recipe['image']))
    except FileNotFoundError as e:
        pass
    else:
        n_downloads += 1
        # recipes_downloaded.append(recipe)
        pbar.update(1)
    i += 1

    # if n_downloads % 10000 == 0:
        # json.dump(recipes_downloaded, open(os.path.join(BASE_PATH, "download", f"{n_downloads}____" + METADATA_FILENAME), "w"), indent=4)

# json.dump(recipes_downloaded, open(os.path.join(BASE_PATH, "download", METADATA_FILENAME), "w"), indent=4)
if i == len(recipes):
    print("Not enough images found")


# recipes_downloaded['image'] =
#
# train_size = len(recipes_downloaded) - int(len(recipes_downloaded) * (VAL_SIZE + TEST_SIZE))
# val_size = int(len(recipes_downloaded) * VAL_SIZE)
# test_size = int(len(recipes_downloaded) * TEST_SIZE)
#
# recipes_train, recipes_test = train_test_split(recipes_downloaded, shuffle=True, test_size=test_size, random_state=seed)
# recipes_train, recipes_val = train_test_split(recipes_train, shuffle=True, test_size=val_size, random_state=seed)
#
# #%%
# for phase, recipes_data in zip(["train", "val", "test"], [recipes_train, recipes_val, recipes_test]):
#     os.makedirs(os.path.join(BASE_PATH, phase), exist_ok=True)
#     images_not_found = 0
#     out_recipes = []
#
#     for _, recipe in tqdm(recipes_data.iterrows(), desc=f"Downloading {phase} images"):
#         try:
#             download_img(recipe['src_url'], os.path.join(BASE_PATH, recipe['image']))
#         except FileNotFoundError as e:
#             images_not_found += 1
#         else:
#             out_recipes.append(recipe)
#     print(f"Images not found during {phase} phase: {images_not_found}")
#
#     json.dump(out_recipes, open(os.path.join(BASE_PATH, phase, METADATA_FILENAME)))
#
#
#
# img_stats = os.path.join(BASE_PATH, IMG_STATS_FILENAME)
# if os.path.exists(img_stats):
#     os.remove(img_stats)
