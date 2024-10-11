import json

from settings.config import *

recipes = []
for name in os.listdir(RAW_YUMMLY_RECIPES_PATH):
    with open(os.path.join(RAW_YUMMLY_RECIPES_PATH, name), 'r') as f:
        recipes.extend(json.load(f))


with open(os.path.join(YUMMLY_RECIPES_PATH, 'all_recipes.json'), 'w') as f:
    json.dump(recipes, f, indent=4, sort_keys=True)
