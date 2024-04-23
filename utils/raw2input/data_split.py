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

seed = 42
full_md_train, full_md_test, full_md_val = [], [], []

for category in os.listdir(RAW_IMAGES_PATH):
    metadata = json.load(open(os.path.join(RAW_METADATA_PATH, category + ".json")))
    images = os.listdir(os.path.join(RAW_IMAGES_PATH, category))

    # Split the data
    images_train, images_test, metadata_train, metadata_test = train_test_split(images, metadata,
                                                                                test_size=TEST_SIZE, random_state=seed)

    images_train, images_val, metadata_train, metadata_val = train_test_split(images_train, metadata_train,
                                                                              test_size=VAL_SIZE, random_state=seed)

    # Copy the images and metadata to the new directories
    for phase, image_dataset in zip(["train", "val", "test"], [images_train, images_val, images_test]):
        for image in tqdm(image_dataset, desc=f"Copying {category} images of {phase} phase"):
            shutil.copy(str(os.path.join(RAW_IMAGES_PATH, category, image)),
                        str(os.path.join(IMAGES_PATH, phase, tokenize_category(category.lower()) + image)))

    full_md_train.extend(metadata_train)
    full_md_val.extend(metadata_val)
    full_md_test.extend(metadata_test)


# Save the metadata
json.dump(full_md_train, open(os.path.join(METADATA_PATH, "train.json"), "w"))
json.dump(full_md_val, open(os.path.join(METADATA_PATH, "val.json"), "w"))
json.dump(full_md_test, open(os.path.join(METADATA_PATH, "test.json"), "w"))