import json
import os

from settings.config import RECIPES_PATH, RAW_RECIPES_PATH, METADATA_FILENAME

if __name__ == "__main__":
    recipes_train = json.load(open(os.path.join(RAW_RECIPES_PATH, "train.json")))
    recipes_test = json.load(open(os.path.join(RAW_RECIPES_PATH, "test.json")))
    recipes_total = recipes_train + recipes_test
    json.dump(recipes_total, open(os.path.join(RECIPES_PATH, METADATA_FILENAME), "w"), indent=2)
    