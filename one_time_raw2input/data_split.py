import os
import pathlib
from sklearn.model_selection import train_test_split
import json
import shutil
from tqdm import tqdm

from config import *
from settings import VAL_SIZE, TEST_SIZE
from commons import tokenize_category

seed = 42

# Create the directories for train, val, and test data
for phase in ["train", "val", "test"]:  # delete and recreate the directories
    if os.path.exists(os.path.join(IMAGES_PATH, phase)):
        shutil.rmtree(os.path.join(IMAGES_PATH, phase))
    os.mkdir(os.path.join(IMAGES_PATH, phase))

# load and split the recipes
recipes = json.load(open(os.path.join(RECIPES_PATH, "recipes_sorted.json")))

recipes_train, recipes_test = train_test_split(recipes, shuffle=False, test_size=TEST_SIZE, random_state=seed)
recipes_train, recipes_val = train_test_split(recipes_train, shuffle=False, test_size=VAL_SIZE, random_state=seed)

int_id = 1
for phase, recipes_dataset in zip(["train", "val", "test"], [recipes_train, recipes_val, recipes_test]):
    images_not_found = 0
    for recipe in tqdm(recipes_dataset, desc=f"Copying {phase} recipes"):
        # Copy rhe image from /RAW/cuisine/id.jpg to /IMAGES/phase/int_id____cuisine.jpg
        img_src_path = os.path.join(RAW_IMAGES_PATH, recipe['cuisine'], str(recipe['id']) + ".jpg")
        img_final_name = str(int_id) + tokenize_category(recipe['cuisine'].lower()) + ".jpg"
        img_dest_path = os.path.join(IMAGES_PATH, phase, img_final_name)

        if os.path.exists(img_src_path):
            shutil.copy(str(img_src_path), str(img_dest_path))
            recipe['old_id'] = recipe['id']
            recipe['id'] = int_id
            int_id += 1
        else:
            images_not_found += 1
            recipes_dataset.remove(recipe)

    print(f"Images not found during {phase} phase: {images_not_found}")
    json.dump(recipes_dataset, open(os.path.join(RECIPES_PATH, f"{phase}.json"), "w"), indent=4)
