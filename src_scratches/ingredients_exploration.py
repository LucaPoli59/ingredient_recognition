import json
import pandas as pd
from settings.config import *

# for category in os.listdir(RAW_IMAGES_PATH):
#     print(len(os.listdir(os.path.join(RAW_IMAGES_PATH, category))))

train_recipes = pd.DataFrame(json.load(open(os.path.join(YUMMLY_RECIPES_PATH, 'train.json'))))
val_recipes = pd.DataFrame(json.load(open(os.path.join(YUMMLY_RECIPES_PATH, 'val.json'))))
test_recipes = pd.DataFrame(json.load(open(os.path.join(YUMMLY_RECIPES_PATH, 'test.json'))))


normalize = True
print(train_recipes['ingredients_ok'].explode().value_counts(normalize=normalize), "\n\n")
print(val_recipes['ingredients_ok'].explode().value_counts(normalize=normalize), "\n\n")
print(test_recipes['ingredients_ok'].explode().value_counts(normalize=normalize), "\n\n")
