import json

from config import *

recipes = []
for name in os.listdir(RAW_RECIPES_PATH):
    with open(os.path.join(RAW_RECIPES_PATH, name), 'r') as f:
        recipes.extend(json.load(f))


with open(os.path.join(RECIPES_PATH, 'all_recipes.json'), 'w') as f:
    json.dump(recipes, f, indent=4, sort_keys=True)
