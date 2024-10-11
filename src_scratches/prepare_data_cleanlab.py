import json
import os

import pandas as pd
import torch
import shutil

from src.data_processing.images_recipes import _recipes_filter_by_category, _compute_images_paths
from src.data_processing.labels_encoders import MultiLabelBinarizerRobust

from settings.config import YUMMLY_PATH, YUMMLY_RECIPES_PATH, DATA_PATH

CLEANLAB_DATA_PATH = os.path.join(DATA_PATH, "cleanlab", "images_recipes")





if __name__ == "__main__":

    # if os.path.exists(CLEANLAB_DATA_PATH):
    #     shutil.rmtree(CLEANLAB_DATA_PATH)
    
    # os.makedirs(CLEANLAB_DATA_PATH)

    CATEGORY = "all"
    INGREDIENTS_TO_KEEP = 100
    
    # coping images to the cleanlab directory
    # for img in _compute_images_paths(os.path.join(YUMMLY_PATH, "train"), category=CATEGORY):
    #     shutil.copy(img, os.path.join(CLEANLAB_DATA_PATH))

    # prepare the data
    images_paths = os.listdir(CLEANLAB_DATA_PATH)
    recipes = _recipes_filter_by_category(json.load(open(os.path.join(YUMMLY_PATH, "train", "metadata.json"))), category=CATEGORY)
    recipes_df = pd.DataFrame(recipes)
    ingredients_selected = recipes_df['ingredients_ok'].explode().value_counts()[:INGREDIENTS_TO_KEEP].index.tolist()

    recipes_fin = []
    for img, recipe in zip(images_paths, recipes):
        ingredients = set(recipe['ingredients_ok']).intersection(ingredients_selected)
        if len(ingredients) != 0:
            recipes_fin.append({"id": str(recipe['id']), "image": str(img), "label": ",".join(ingredients),
                                "cuisine": recipe['cuisine'], "name": recipe['name'],
                                "course": ",".join(recipe['course'])})

    json.dump(recipes_fin, open(os.path.join(CLEANLAB_DATA_PATH, "metadata.json"), "w"), indent=4)


    
    

