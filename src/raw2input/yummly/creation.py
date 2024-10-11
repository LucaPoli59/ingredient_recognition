from sklearn.model_selection import train_test_split
import json
import shutil
from tqdm import tqdm

from settings.config import *
from settings.commons import tokenize_category

seed = 42

# Create the directories for train, val, and test data
for phase in ["train", "val", "test"]:  # delete and recreate the directories
    if os.path.exists(os.path.join(YUMMLY_PATH, phase)):
        shutil.rmtree(os.path.join(YUMMLY_PATH, phase))
    os.mkdir(os.path.join(YUMMLY_PATH, phase))

# load and split the recipes
recipes = json.load(open(os.path.join(YUMMLY_RECIPES_PATH, "recipes_sorted.json")))
id_encode_len = len(str(len(recipes))) + 1

train_size = len(recipes) - int(len(recipes) * (VAL_SIZE + TEST_SIZE))
val_size = int(len(recipes) * VAL_SIZE)
test_size = int(len(recipes) * TEST_SIZE)

recipes_train, recipes_test = train_test_split(recipes, shuffle=True, test_size=test_size, random_state=seed)
recipes_train, recipes_val = train_test_split(recipes_train, shuffle=True, test_size=val_size, random_state=seed)

int_id = 0
for phase, recipes_dataset in zip(["train", "val", "test"], [recipes_train, recipes_val, recipes_test]):
    images_not_found = 0
    out_recipes = []
    for recipe in tqdm(recipes_dataset, desc=f"Copying {phase} recipes"):
        # Copy rhe image from /RAW/cuisine/id.jpg to /IMAGES/phase/int_id____cuisine.jpg
        img_src_path = os.path.join(RAW_YUMMLY_IMAGES_PATH, recipe['cuisine'], str(recipe['id']) + ".jpg")
        img_final_name = str(int_id).zfill(id_encode_len) + tokenize_category(recipe['cuisine'].lower()) + ".jpg"
        img_dest_path = os.path.join(YUMMLY_PATH, phase, img_final_name)

        if os.path.exists(img_src_path):
            shutil.copy(str(img_src_path), str(img_dest_path))
            recipe['old_id'] = recipe['id']
            recipe['id'] = int_id
            recipe['image'] = img_final_name
            int_id += 1
            out_recipes.append(recipe)
        else:
            images_not_found += 1

    print(f"Images not found during {phase} phase: {images_not_found}")
    json.dump(out_recipes, open(os.path.join(YUMMLY_PATH, phase, METADATA_FILENAME), "w"), indent=4)

yummly_img_stats = os.path.join(YUMMLY_PATH, IMG_STATS_FILENAME)
if os.path.exists(yummly_img_stats):
    os.remove(yummly_img_stats)


# Remember to run the compute_img_stats.py script to compute the mean and std of the training images