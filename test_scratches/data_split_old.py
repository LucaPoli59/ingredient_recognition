import os
import pathlib
from sklearn.model_selection import train_test_split
import json
import shutil
from tqdm import tqdm

from config import *
from settings import VAL_SIZE, TEST_SIZE
from commons import tokenize_category

# Create the directories for train, val, and test data
for phase in ["train", "val", "test"]:  # delete and recreate the directories
    if os.path.exists(os.path.join(IMAGES_PATH, phase)):
        shutil.rmtree(os.path.join(IMAGES_PATH, phase))
    os.mkdir(os.path.join(IMAGES_PATH, phase))

recipes = json.load(open(os.path.join(RECIPES_PATH, "recipes_sorted.json")))

seed = 42
full_md_train, full_md_test, full_md_val = [], [], []
full_r_train, full_r_test, full_r_val = [], [], []

for category in os.listdir(RAW_IMAGES_PATH):

    metadata_s = json.load(open(os.path.join(RAW_METADATA_PATH, category + ".json")))
    images_s = os.listdir(os.path.join(RAW_IMAGES_PATH, category))
    recipes_cat = list(filter(lambda recipe: recipe['cuisine'] == category, recipes))

    # since some recipes are not present in the dataset, we need to filter them out
    id_recipes = [recipe['id'] for recipe in recipes_cat]
    images = [image for image in images_s if image.removesuffix(".jpg") in id_recipes]
    metadata = [md for md in metadata_s if md['id'] in id_recipes]

    print(f"Category: {category}\nAll images {len(images_s)}\nAll metadata {len(metadata_s)}\n"
          f"All recipes {len(recipes_cat)}\nAll immages filtered {len(images)}\nAll metadata filtered {len(metadata)}\n")

    # Split the data
    (images_train, images_test, metadata_train,
     metadata_test, recipes_train, recipes_test) = train_test_split(images, metadata, recipes_cat, shuffle=False,
                                                                                test_size=TEST_SIZE, random_state=seed)

    (images_train, images_val, metadata_train,
     metadata_val, recipes_train, recipes_val) = train_test_split(images_train, metadata_train, recipes_train,
                                                                              test_size=VAL_SIZE, random_state=seed,
                                                                              shuffle=False)

    # Copy the images and metadata to the new directories
    for phase, image_dataset in zip(["train", "val", "test"], [images_train, images_val, images_test]):
        for image in tqdm(image_dataset, desc=f"Copying {category} images of {phase} phase"):
            shutil.copy(str(os.path.join(RAW_IMAGES_PATH, category, image)),
                        str(os.path.join(IMAGES_PATH, phase, tokenize_category(category.lower()) + image)))

    full_md_train.extend(metadata_train)
    full_md_val.extend(metadata_val)
    full_md_test.extend(metadata_test)

    full_r_train.extend(recipes_train)
    full_r_val.extend(recipes_val)
    full_r_test.extend(recipes_test)


# Save the metadata
json.dump(full_md_train, open(os.path.join(METADATA_PATH, "train.json"), "w"))
json.dump(full_md_val, open(os.path.join(METADATA_PATH, "val.json"), "w"))
json.dump(full_md_test, open(os.path.join(METADATA_PATH, "test.json"), "w"))

json.dump(full_r_train, open(os.path.join(RECIPES_PATH, "train.json"), "w"))
json.dump(full_r_val, open(os.path.join(RECIPES_PATH, "val.json"), "w"))
json.dump(full_r_test, open(os.path.join(RECIPES_PATH, "test.json"), "w"))

# todo: alcune immagini e metadata non sono presenti nel dataset delle ricette