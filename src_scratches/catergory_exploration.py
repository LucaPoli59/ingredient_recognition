import json
import pandas as pd
from settings.config import *

# for category in os.listdir(RAW_IMAGES_PATH):
#     print(len(os.listdir(os.path.join(RAW_IMAGES_PATH, category))))

train_recipes = pd.DataFrame(json.load(open(os.path.join(RECIPES_PATH, 'train.json'))))
val_recipes = pd.DataFrame(json.load(open(os.path.join(RECIPES_PATH, 'val.json'))))
test_recipes = pd.DataFrame(json.load(open(os.path.join(RECIPES_PATH, 'test.json'))))


print("Train: ", train_recipes['cuisine'].value_counts(), "\n\n")
print("Val: ", val_recipes['cuisine'].value_counts(), "\n\n")
print("Test: ", test_recipes['cuisine'].value_counts(), "\n\n")


print("Train len ", len(train_recipes))
print("Val len ", len(val_recipes))
print("Test len ", len(test_recipes))
