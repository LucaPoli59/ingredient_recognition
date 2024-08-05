import json

from settings.config import *

all_recipes = json.load(open(os.path.join(YUMMLY_RECIPES_PATH, 'all_recipes.json')))

output_recipes = []
for category in os.listdir(RAW_IMAGES_PATH):
    images = os.listdir(os.path.join(RAW_IMAGES_PATH, category))

    recipes_cat = list(filter(lambda recipe: recipe['cuisine'] == category, all_recipes))
    recipes_cat_sorted = sorted(recipes_cat, key=lambda x: x['id'])
    output_recipes.extend(recipes_cat_sorted)

json.dump(output_recipes, open(os.path.join(YUMMLY_RECIPES_PATH, 'recipes_sorted.json'), 'w'), indent=4)

