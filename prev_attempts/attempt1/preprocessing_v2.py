import os,sys,math,time,io,argparse,json,traceback,collections, random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms, utils, models, ops
from multiprocessing import cpu_count, Pool
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import seaborn as sns
import re
import textdistance
import itertools
from multiprocessing import Pool

sns.set()

# sns.set(font_scale=0.8)


def sim(cur_combo):
	# get textdistance algorithms
	algs = textdistance.algorithms
	# define similarity
	similarity = algs.levenshtein.normalized_similarity(cur_combo[0], cur_combo[1])
	# create output
	out = {'ingredient1':cur_combo[0], 'ingredient2':cur_combo[1], 'similarity':similarity}
	# return it
	return out


def get_ingredient_similarity(ingredients):
	# get textdistance algorithms
	algs = textdistance.algorithms
	# get ingredients
	ings = ingredients['ingredient'].str.lower().to_list()
	# define all possible combinations
	combos = list(itertools.combinations(ings, 2))
	# define output
	out = []

	# debug
	pool = Pool(8)
	# out = pool.map(sim, combos)
	out = list(tqdm(pool.imap_unordered(sim, combos), total=len(combos)))
	# debug
	# # for each couple of ingredients
	# for cur_combo in tqdm(combos):
	# 	# compute similarity
	# 	similarity = algs.levenshtein.normalized_similarity(cur_combo[0], cur_combo[1])
	# 	# save it
	# 	out.append({'ingredient1':cur_combo[0], 'ingredient2':cur_combo[1], 'similarity':similarity})
	# at the end, convert to dataframe
	df = pd.DataFrame(out)
	# sort by similarity
	df = df.sort_values(by=['similarity'], ascending=False)
	# df[df['similarity']>0.85]
	# return it
	return df


def merge_with_ingredient_similarity(ingredients, mapping, threshold=0.85):
	# get ingredient similarity
	ingredient_similarity = get_ingredient_similarity(ingredients)
	# filter by threshold
	ingredient_similarity = ingredient_similarity[ingredient_similarity['similarity']>threshold]
	# for each similarity
	for i,cur_similarity in ingredient_similarity.iterrows():
		# get ingredients
		ingredient1 = cur_similarity['ingredient1']
		ingredient2 = cur_similarity['ingredient2']
		# replace all occurrences of ingredient2 with ingredient1 both in ingredients and mapping
		ingredients['ingredient'].replace(ingredient2, ingredient1, inplace=True)
		mapping['ingredient_ok'].replace(ingredient2, ingredient1, inplace=True)
	# join ingredients
	# ingredients = ingredients.groupby('ingredient').sum().reset_index()
	ingredients = ingredients.groupby('ingredient')['count'].sum().reset_index()
	ingredients = ingredients.sort_values(by='count', ascending=False)
	# return new mapping, new ingredients and ingredient similarity
	return mapping, ingredients, ingredient_similarity
	


def load_all(recipe_folder):
	# define all output
	out = []
	# find all json files inside the folder
	cuisine_files = [os.path.join(recipe_folder, f) for f in os.listdir(recipe_folder) if f.endswith('.json')]
	# initialize recipe count
	recipe_count = 0
	# for each cuisine
	for cur_cuisine_file in tqdm(cuisine_files):
		# read json
		with open(cur_cuisine_file) as f:
			cur_cuisine_data = json.load(f)
		# update output
		out.extend(cur_cuisine_data)
	# return all
	return out
		

def update_recipes(recipes, mapping, min_num_ingredients=3):
	# convert mapping in a dictionary
	mapping_dict = mapping.set_index('ingredient')['ingredient_ok'].to_dict()
	# keep track of recipes that have to be deleted
	to_be_deleted = []
	# for each recipe
	for i,cur_recipe in enumerate(tqdm(recipes)):
		# create new ingredient line
		cur_recipe['ingredients_ok'] = []
		# for each ingredient
		for cur_ingredient in cur_recipe['ingredients']:
			# check if considered
			if cur_ingredient in mapping_dict:
				new_name = mapping_dict[cur_ingredient]
				cur_recipe['ingredients_ok'].append(new_name)
			# # get index
			# idx = mapping[mapping['ingredient']==cur_ingredient].index
			# # get new name
			# if len(idx) > 0:
			# 	new_name = mapping.loc[idx, 'ingredient_ok'].iloc[0]
			# 	# append it
			# 	cur_recipe['ingredients_ok'].append(new_name)
		# remove duplicates
		cur_recipe['ingredients_ok'] = list(set(cur_recipe['ingredients_ok']))
		# at the end, check if there are more than min_num_ingredients
		if len(cur_recipe['ingredients_ok']) < min_num_ingredients:
			# delete this recipe
			to_be_deleted.append(i)
	# delete recipes with too few ingredients
	for cur_id in reversed(to_be_deleted):
		del recipes[cur_id]
	# return recipes
	return recipes



def count_and_filter_new_ingredients(df, min_occurrences=15):
	# get new ingredients
	new_ingredients = df['ingredient_ok'].value_counts()
	# threshold on count
	new_ingredients_thresholded = new_ingredients[new_ingredients>=min_occurrences]
	# rename
	new_ingredients_thresholded = new_ingredients_thresholded.to_frame().reset_index()
	new_ingredients_thresholded.rename({'index':'ingredient', 'ingredient_ok':'count'}, axis=1, inplace=True)
	# remove empty ingredient
	idx_to_drop = new_ingredients_thresholded[new_ingredients_thresholded['ingredient']==''].index
	new_ingredients_thresholded.drop(idx_to_drop, inplace = True)
	# return them
	return new_ingredients_thresholded
	

############################## BASE PROCESSING ##############################

def remove_paranthesis_and_markups(ingredient):
	# remove "
	ingredient = re.sub(r'"', '', ingredient)
	# remove *
	ingredient = re.sub(r'\*', '', ingredient)
	# remove trailing and leading spaces
	ingredient = ingredient.strip()
	# make it lowercase
	ingredient = ingredient.lower()
	# keep only first part if there is a dash surrounded by spaces
	ingredient = re.sub(r'\s+\-\s+', '', ingredient)
    # remove parenthesis
	ingredient = re.sub(r'\([^)]*\)?', '', ingredient)
	# if there is ;, keep only first
	ingredient = ingredient.split(';')[0]
	# remove tabs
	ingredient = re.sub(r'\t', '', ingredient)
	# return 
	return ingredient

############################## QUANTIFIERS ##############################

def remove_quantifiers(ingredient):
	# convert fractions
	fractions = [('½','1/2'), ('⅓', '1/3'), ('¼','1/4'), ('¾','3/4'), ('⅔','2/3'), ('⅛','1/8'), ('⅜','3/8'), ('⅝','5/8'), ('⅞','7/8')]
	for cur_fraction in fractions:
		ingredient = ingredient.replace(cur_fraction[0], cur_fraction[1])
	# remove quantifiers
	# ¼ to ½ tsp.
	# 1 / 1/4 / 2 1/2 / 1/4 to 1/2 / 4-5 / 1/2tbps
	frac_regex = r'(\d+\/\d+)'
	int_regex = r'\d+'
	# frac to frac / int frac / int to int / int - int / frac / int
	quant_regex = f'({frac_regex}\s+to\s+{frac_regex}|{int_regex}\s+{frac_regex}|{int_regex}\s+to\s+{int_regex}|{int_regex}\s?-\s?{int_regex}|{frac_regex}|{int_regex})'
	# print(quant_regex)
	# fraction_regex = r'((((\d+\s+)?\d+\/\d+|\d+)\s?(to|\-)\s?)?((\d+\s+)?\d+\/\d+|\d+)\s?)?\s{0,}'
	# 1 tablespoon / 3 tablespoons / 2 1/2 tablespoons / 1/4 tablespoon / 1table spoon / 1/4 to 1/2 tablespoon / 1/2 teaspoon / 1/2 to 1/4 tea spoon
	# print(ingredient)
	tablespoon_regex = f'{quant_regex}\s?(table|tea)\s?spoon(s)?'
	ingredient = re.sub(tablespoon_regex, '', ingredient, flags=re.IGNORECASE)
	# print(ingredient)
	# 6 Tbs. / 1 tbsp / 1 Tbsp / 2 Tbsp. / 1 TBSP / 1 tsp / 1/2 tsp.
	tbs_regex = f'{quant_regex}\s?(tbs(p)?|tsp)\.?'
	ingredient = re.sub(tbs_regex, '', ingredient, flags=re.IGNORECASE)
	# print(ingredient)
	# 1/2 cup / 1 cup / 4-5 cups / 1/2 cup / 11/2  cups / 2 1/2 cups
	cup_regex = f'{quant_regex}\s?cup(s)?'
	ingredient = re.sub(cup_regex, '', ingredient, flags=re.IGNORECASE)
	# print(ingredient)
	# 1/2 kg or 1 lb / 1 1/2 Lbs / 1/2 lb / 1/2 pound / 1/2 pounds / 1/2 lb. / 1/2 lbs /
	weight_regex = f'{quant_regex}\s{{0,}}(kg|lb(s)?|pound(s)?|(\-)?oz|gram(s)?|\%|ounce(s)?|g\s+|c\.)\.?'
	all_weights_regex = f'({weight_regex}\s+or\s+{weight_regex} | {weight_regex})'
	ingredient = re.sub(all_weights_regex, '', ingredient, flags=re.IGNORECASE)
	ingredient = re.sub(weight_regex, '', ingredient, flags=re.IGNORECASE)
	# print(ingredient)
	# 120 milliliters / 8 liters / lt. attenzione a spazio prima e dopo
	liquid_regex = f'{quant_regex}\s{{0,}}(milliliter(s)?|kilogram(s)?|ml|lt\.?)'
	ingredient = re.sub(liquid_regex, '', ingredient, flags=re.IGNORECASE)
	# print(ingredient)
	# Doses / cans / jars / jars of / box / package / pinch of - attenzione allo spazio prima e dopo can
	cans_regex = f'{quant_regex}\s{{0,}}(dose(s)?|can(s)?|jar(s)?(\s+of)?|box(es)?|package(s)?|pinch(\s+of)?|head(s)?|pkg(s)?|container(s)?)\s?'
	ingredient = re.sub(cans_regex, '', ingredient, flags=re.IGNORECASE)
	# print(ingredient)
	# remove numbers
	ingredient = re.sub(quant_regex, '', ingredient, flags=re.IGNORECASE)
	# remove trailing and leading spaces
	ingredient = ingredient.strip()
	# return
	return ingredient



def remove_list_of_elements(ingredient, elements):
	# set elements lowercase
	elements = [cur_element.lower() for cur_element in elements]
	# create regex
	elements_regex = '(' + '|'.join(elements) + ')'
	# remove elements
	ingredient = re.sub(elements_regex, '', ingredient, flags=re.IGNORECASE)
	# return
	return ingredient


def remove_textual_quantifiers(ingredient):
	# define quantifiers
	quantifiers = ["Small", "Medium", "Large", "Extra-large", "Handful", "Pinch", "Heap", "Dash", "Teaspoonful", "Tablespoonful", 
				"Cupful", "Quart", "Gallon", "Few", "Several", "Dozen", "Score", "Bulk", "Finely", "Fine", "Good quality", 
				"Premium", "Superior", "Excellent", "Choice", "Select", "Prime", "First-rate", "High-grade", "Coarsely", 
				"Coarse", "Thickly", "Thick", "Thinly", "Thin", "Sliced", "Chopped", "Diced", "Minced", "Shredded", "Grated", 
				"Ground", "Lightly", "Light", "Heavily", "Heavy", "Strong", "Weak", "Mild", "Intense", "Moderate", "Sparse", 
				"Dense", "Rich", "Sparse", "Occasional", "Regular", "Frequent", "Constant", "Intermittent", "Rare", "Seldom", 
				"Often", "Always", "Never", "Generous", "Sparse", "Abundant", "Scant", "Meager", "Plentiful", "Copious", "Vast", 
				"Minimal", "Substantial", "Significant", "Negligible", "big", "extralarge", "slices"]
	# remove them and return
	return remove_list_of_elements(ingredient, quantifiers)


############################## TREATMENTS AND TEMPERATURES ##############################

def remove_treatments(ingredient):
	treatments = ["grated", "chopped", "sliced", "diced", "minced", "shredded", "pitted", "halved", "quartered", "peeled", "crushed",
			   "ground", "toasted", "slivered", "trimmed", "cubed", "drained", "rinsed", "patted", "dry", "frozen", "melted", "shelled", 
			   "packed", "cold-pressed", "coldpressed", "pressed", "refrigerated", "cooked", "condensed", "toasted", "seasoned", 
			   "lightly", "low-fat", "lowfat", "fat-free", "fatfree", "unbleached", "unsweetened", "sweetened", "clarified", "plain", 
			   "granulated", "chunky", "in syrup", "unsalted", "refried", "dried", "thawed", "miscellaneous", "delicious", "boneless", 
			   "rounded", "coarsely", "chopped", "lightly", "light", "frozen", "shelled", "bulk", "halves", "rounded", "loosely"]
	# remove them and return
	return remove_list_of_elements(ingredient, treatments)


def remove_temperature(ingredient):
	# cold / hot / warm / room temperature / at room temperature / at room temp / at room temp.
	ingredient = re.sub(r'(cold|hot|warm|(at\s)?room temp(\.|erature)|freshly|fresh)', '', ingredient, flags=re.IGNORECASE)
	# define temperature strings
	temperatures = ["hot", "warm", "piping hot", "sizzling", "steaming", "toasted", "heated", "roasted", "grilled", "boiling", 
				"simmering", "blazing", "scalding", "broiled", "baked", "seared", "charred", "glowing", "molten", "cold", "cool",
				"chilled", "icy", "frozen", "refrigerated", "frosty", "crisp", "refreshing", "gelid", "frigid", "glacial", "frosted",
				"ice-cold", "frozen solid", "thawed", "room temperature", "ambient", "tempered", "lukewarm", "tepid", "mild"]
	# remove and return them
	return remove_list_of_elements(ingredient, temperatures)
	

############################## CATHEGORIES ##############################

def classify_ingredient(ingredient, patterns):
    for category, pattern in patterns.items():
        if re.search(pattern, ingredient, re.IGNORECASE):
            return True, category
    return False, None


def is_alcoholic(ingredient):
	# define sub category regex
	patterns = {
		'Liquor': r'\b(vodka|gin|rum|whiskey|tequila|brandy|cognac|bourbon)\b',
		'Wines': r'\b(vermouth|sherry|port|marsala)\b',
		'Beers and Ciders': r'\b(\sale|lager|stout|cider)\b',
		'Fortified Wines and Spirits': r'\b(champagne|prosecco|calvados|grappa)\b',
		'Liqueurs': r'\b(amaretto|cointreau|frangelico|schnapps|kirsch)\b',
	}
	# classify
	return classify_ingredient(ingredient, patterns)


def is_diary_and_eggs(ingredient):
	# define cheese subcategory regex
	cheese_types = ['ricotta', 'parmesan', 'parmigiano', 'mozzarella', 'pecorino', 'provolone', 'feta', 'gouda', 'manchego',
				  'feta', 'brie', 'camembert', 'gorgonzola', 'roquefort', 'emmental', 'gruyere', 'asiago', 'pecorino', 
				  'manchego', 'halloumi', 'paneer', 'cottage', 'goat', 'swiss', 'provolone', 'fontina', 'havarti',
				  'monterey', 'jack', 'colby', 'pepperjack', 'muenster', 'limburger', 'brick', 'edam', 'havarti', 'stilton',
				  'boursin', 'Monterey Jack', 'cheddar', 'gruyère', 'Philadelphia', 'fontina', 'Beaufort', 'Asiago', 
				  'Parmigano', 'Reggiano', 'grana', 'mascarpone', 'Cantal', 'Cotija', 'Brie', 'Jarlsberg', 'Stilton',
				  'Robiola', 'fior di latte', 'caciocavallo', 'jack', 'chèvre', 'Taleggio', 'Muenster', 'Havarti', 'cottage',
				  'Emmenthal', 'paneer', 'Chihuahua', 'Neufchâtel', 'Comté', 'stracchino', 'Pecorina', 'Boursin', 'Parrano',
				  'hard cheese', 'soft cheese', 'fermented cheese', 'cheddar', 'brie', 'camembert', 'queso', 'cheese']
	# define regex for cheese
	cheese_regex = r'\b(' + '|'.join(cheese_types) + r')\b'
	# Define regular expressions for each subcategory
	patterns = {
		'Milk': r'\b(cow milk|goat milk|almond milk|soy milk|oat milk|milk)\b',
		'Cheese': cheese_regex,
		'Yogurt': r'\b(regular yogurt|greek yogurt|plant-based yogurt)\b',
		'Eggs': r'\b(chicken egg|duck egg|quail egg|egg)\b',
		'Butter and Cream': r'\b(butter|cream|heavy cream|whipping cream|cream)\b',
	}


def is_fruit_and_vegetables(ingredient):
	patterns = {
		'Leafy Greens': r'\b(spinach|kale|lettuce)\b',
		'Root Vegetables': r'\b(carrots|potatoes|beets)\b',
		'Fruits': r'\b(apples|bananas|berries)\b',
		'Citrus': r'\b(oranges|lemons|limes)\b',
		'Nightshades': r'\b(tomatoes|peppers|eggplants)\b',
	}
	# classify
	return classify_ingredient(ingredient, patterns)



def process_ingredient(ingredient):
	# save original ingredient
	orig_ingredient = ingredient
	# remove paranethesis and markups
	ingredient = remove_paranthesis_and_markups(ingredient)
	# remove numeric quantifiers
	ingredient = remove_quantifiers(ingredient)
	# remove textual quantifiers
	ingredient = remove_textual_quantifiers(ingredient)
	# remove treatments
	ingredient = remove_treatments(ingredient)
	# remove temperature
	ingredient = remove_temperature(ingredient)
	# whole / brewed / to taste
	ingredient = re.sub(r'about|whole|brewed|to taste|reduced\-fat|lowfat|nonfat|homemade|prepared', '', ingredient, flags=re.IGNORECASE)
	# about two large handfuls of baby spinach
	# Fistful of finely chopped coriander leaves
	# chicken breast
	keep_as_is = ['chicken breast', 'chicken thighs', 'chicken broth', 'oregano', 'red wine', 'white wine', 'water']
	for cur_keep in keep_as_is:
		if cur_keep in ingredient:
			ingredient = cur_keep
			break
	# remove trademarks
	# trademarks = ["I Can't Believe It's Not Butter!®", "Country Crock®"]
	trademarks = ["i can't believe it's not butter!®", "country crock®", "I Can't Believe It's Not Butter!® Spread"]
	for cur_trademark in trademarks:
		ingredient = ingredient.replace(cur_trademark, '')
	# oil
	# extra virgin olive oil / extravirgin olive oil / 
	# change extra-virgin or extra virgin to extravirgin
	ingredient = re.sub(r'(extra\-virgin|extra\s+virgin)', 'extravirgin', ingredient, flags=re.IGNORECASE)
	ingredient = re.sub(r'extravirgin', '', ingredient, flags=re.IGNORECASE)
	# change allpurpose flour to flour
	ingredient = re.sub(r'allpurpose', '', ingredient, flags=re.IGNORECASE)
	# change all-purpose flour to flour
	ingredient = re.sub(r'all\-purpose', '', ingredient, flags=re.IGNORECASE)
	# change mehl to flour
	ingredient = re.sub(r'mehl', 'flour', ingredient, flags=re.IGNORECASE)
	# change öl to oil
	ingredient = re.sub(r'öl', 'oil', ingredient, flags=re.IGNORECASE)
	ingredient = re.sub(r'olivenöl', 'olive oil', ingredient, flags=re.IGNORECASE)
	# remove for garnish
	ingredient = re.sub(r'for\s+garnish', '', ingredient, flags=re.IGNORECASE)
	# remove +, -, –
	ingredient = re.sub(r'(\+|\-|\–)', '', ingredient, flags=re.IGNORECASE)
	# remove or, plus
	ingredient = re.sub(r'\s{1,}(or|plus)\s{1,}', '', ingredient, flags=re.IGNORECASE)
	# replace queso with cheese
	if 'queso' in ingredient:
		ingredient = 'cheese'
	# replace all types of cheese with cheese
	cheese_types = ['ricotta', 'parmesan', 'parmigiano', 'mozzarella', 'pecorino', 'provolone', 'feta', 'gouda', 'manchego',
				  'feta', 'brie', 'camembert', 'gorgonzola', 'roquefort', 'emmental', 'gruyere', 'asiago', 'pecorino', 
				  'manchego', 'halloumi', 'paneer', 'cottage', 'goat', 'swiss', 'provolone', 'fontina', 'havarti',
				  'monterey', 'jack', 'colby', 'pepperjack', 'muenster', 'limburger', 'brick', 'edam', 'havarti', 'stilton',
				  'boursin', 'Monterey Jack', 'cheddar', 'gruyère', 'Philadelphia', 'fontina', 'Beaufort', 'Asiago', 
				  'Parmigano', 'Reggiano', 'grana', 'mascarpone', 'Cantal', 'Cotija', 'Brie', 'Jarlsberg', 'Stilton',
				  'Robiola', 'fior di latte', 'caciocavallo', 'jack', 'chèvre', 'Taleggio', 'Muenster', 'Havarti', 'cottage',
				  'Emmenthal', 'paneer', 'Chihuahua', 'Neufchâtel', 'Comté', 'stracchino', 'Pecorina', 'Boursin', 'Parrano'] # , 'blue', 'cream cheese'
	for cur_cheese in cheese_types:
		if cur_cheese.lower() in ingredient.lower():
			ingredient = 'cheese'
	# plural to singular (e.g. eggs->egg, onions->onion)
	p2s = [('onions', 'onion'), ('eggs', 'egg'), ('carrots', 'carrot'), ('tomatoes', 'tomato')]
	for cur_p2s in p2s:
		ingredient = re.sub(cur_p2s[0], cur_p2s[1], ingredient, flags=re.IGNORECASE)
	# replace
	p2s = [('cloves of garlic', 'garlic'), ('cloves garlic', 'garlic'), ('garlic cloves', 'garlic'), ('sea salt', 'salt')]
	for cur_p2s in p2s:
		ingredient = re.sub(cur_p2s[0], cur_p2s[1], ingredient, flags=re.IGNORECASE)
	# cloves of garlic
	# remove article at the beginning
	ingredient = re.sub(r'^(an|a|the|ly|el|un|of|ei)\s{1,}', '', ingredient, flags=re.IGNORECASE)
	# simplify some ingredients
	simplify_ingredients = ['chicken', 'beef', 'pork', 'wine', 'onion', 'garlic', 'salt', 'potatoes', 'potato', 'sugar', 'pepper', 
						 'flour', 'oil', 'butter', 'tomato', 'milk', 'cheese', 'rice', 'lemon', 'egg', 'carrot', 
						 'pasta', 'bread', 'cream', 'corn', 'cucumber', 'lettuce', 'apple', 'banana', 'orange', 
						 'strawberry', 'cherry', 'blueberry', 'raspberry', 'blackberry', 'peach', 'pear', 'plum', 
						 'grape', 'watermelon', 'melon', 'pineapple', 'kiwi', 'mango', 'papaya', 'avocado', 'coconut', 
						 'peanut', 'almond', 'walnut', 'hazelnut', 'pistachio', 'cashew', 'pecan', 'macadamia', 
						 'chestnut', 'honey', 'syrup', 'jam', 'chocolate', 'vanilla', 'cinnamon', 'ginger', 'nutmeg', 
						 'clove', 'cardamom', 'turmeric', 'cumin', 'coriander', 'basil', 'parsley', 'oregano', 'thyme', 
						 'rosemary', 'sage', 'mint', 'dill', 'tarragon', 'chili', 'paprika', 'cayenne', 'curry', 
						 'mustard', 'soy', 'vinegar', 'wine', 'beer', 'rum', 'whiskey', 'vodka', 'gin', 'tequila', 
						 'brandy', 'liqueur', 'sake', 'cognac', 'vermouth', 'sherry', 'port', 'champagne', 'prosecco', 
						 'cider', 'juice', 'soda', 'water', 'coffee', 'tea', 'milk', 'yogurt', 'cream', 'butter', 'cheese', 
						 'egg', 'bread', 'pasta', 'rice', 'flour', 'sugar', 'honey', 'syrup', 'jam', 'chocolate', 'vanilla', 
						 'cinnamon', 'ginger', 'nutmeg', 'clove', 'cardamom', 'turmeric', 'cumin', 'coriander', 'basil', 
						 'parsley', 'oregano', 'thyme', 'rosemary', 'mashrooms', 'yeast', 'baking powder', 'baking soda', 
						 ' peas', 'mushrooms', 'beans', 'broccoli', 'cauliflower', 'asparagus', 'zucchini', 'eggplant', 
						 'lobster', 'bacon', 'spinach', 'noodles', 'pumpkin', 'cabbage', 'beet', 'radish', 'turnip', 'shrimp',
						 'artichoke', 'polenta', 'quinoa', 'couscous', 'bulgur', 'barley', 'millet', 'oat', 'wheat', 'rye', 'tapioca',
						 'mayonnaise', 'pancetta', 'celery', 'leek', 'cilantro', 'arugula', 'fennel']
	for cur_simplify in simplify_ingredients:
		if cur_simplify in ingredient:
			ingredient = cur_simplify
	# generalize liquor beverages (no 'wine', 'beer')
	liquor_beverages = ['rum', 'whiskey', 'vodka', 'gin', 'tequila', 'brandy', 'liqueur', 'sake', 'cognac', 'vermouth', 
						'sherry', 'port', 'champagne', 'prosecco', 'cider', 'vermouth', 'liqueur', 'brandy', 'whiskey', 'rum', 'burbon',
						'marsala', 'sherry', 'port', 'cognac', 'calvados', 'grappa', 'schnapps', 'kirsch', 'amaretto', 'frangelico', 'cointreau']
	for cur_liquor in liquor_beverages:
		if cur_liquor.lower() in ingredient.lower():
			ingredient = 'liquor'
	# generalize nuts (shell fruits)
	nuts = ['peanut', 'almond', 'walnut', 'hazelnut', 'pistachio', 'cashew', 'pecan', 'macadamia', 'chestnut']
	for cur_nut in nuts:
		if cur_nut.lower() in ingredient.lower():
			ingredient = 'nuts'
	# generalize spices
	spices = ['cinnamon', 'ginger', 'nutmeg', 'clove', 'cardamom', 'turmeric', 'cumin', 'coriander', 'parsley', 'oregano', 'thyme', 
		   'rosemary', 'sage', 'mint', 'dill', 'tarragon', 'chili', 'paprika', 'cayenne', 'curry', 'ancho chiles', 'chipotle chiles', 'powder']
	
	# 'honey', 'syrup', 'jam', 'chocolate', 'vanilla', 'basil', 'mustard', 'soy', 'vinegar'
	# nutmeg (noce moscata), clove (chiodo di garofano), cardamom (cardamomo), turmeric (curcuma), cumin (cumino), coriander (coriandolo), basil (basilico),
	# generalize mushrooms
	# mushrooms = ['mushrooms', 'mushroom', 'porcini', 'chanterelle', 'shiitake', 'portobello', 'cremini', 'oyster', 'enoki', 'maitake', 'morel', 'truffle']
	# generalize pasta
	pasta = ['pasta', 'spaghetti', 'penne', 'fusilli', 'rigatoni', 'linguine', 'tagliatelle', 'fettuccine', 'macaroni', 'farfalle',
		  	 'orecchiette', 'cavatappi', 'cavatelli', 'gnocchi', 'ravioli', 'tortellini', 'lasagna', 'cannelloni', 'manicotti', 'ziti', 
			 'ditalini', 'tortellini']
	for cur_pasta in pasta:
		if cur_pasta.lower() in ingredient.lower():
			ingredient = 'pasta'
	# remove double spaces
	ingredient = re.sub(r'\s{2,}', ' ', ingredient)
	# Olivenöl
	# for garnish
	# if there is a comma, keep only first part
	ingredient = ingredient.split(',')[0]
	# remove trailing spaces
	ingredient = ingredient.strip(' .')

	# 
	# if ingredient.startswith('ly'):
	# 	print(ingredient)
	# 	print(orig_ingredient)
	# 	print('ciao')
	# return ingredient
	return ingredient


def process_ingredient_in_dataframe(row):
	# get ingredients
	return process_ingredient(row['ingredient'])


def create_mapping(recipes, min_occurrences=15):
	# init ingredient dictionary
	ingredients = {}
	# initialize recipe count
	recipe_count = len(recipes)
	# for each recipe
	for cur_recipe in recipes: 
		# ['flavors', 'cuisine', 'name', 'ingredients', 'course', 'id']
		cur_flavors = cur_recipe['flavors']
		cur_cuisine = cur_recipe['cuisine']
		cur_name = cur_recipe['name']
		cur_ingredients = cur_recipe['ingredients']
		cur_course = cur_recipe['course']
		cur_id = cur_recipe['id']
		# add ingredients to dict
		for cur_ingredient in cur_ingredients:
			ingredients[cur_ingredient] = ingredients.get(cur_ingredient, 0) + 1
	# print recipe count and number of ingredients
	print('Number of recipes: {}'.format(recipe_count))
	print('Number of ingredients: {}'.format(len(ingredients)))

	# transform in dataframe
	df = pd.DataFrame.from_dict(ingredients, orient='index', columns=['count'])
	# reset index
	df = df.reset_index(names=['ingredient', 'count'])
	# sort by count
	df = df.sort_values(by=['count'], ascending=False)
	# filter
	tqdm.pandas()
	df['ingredient_ok'] = df.progress_apply(process_ingredient_in_dataframe, axis=1)
	# print the number of unique ingredients_ok
	print('Number of unique ingredients before union: {}'.format(len(df['ingredient_ok'].unique())))
	# switch axis
	df = df[['ingredient', 'ingredient_ok', 'count']]
	# debug: get all ingredients
	all_ingredients = df['ingredient_ok'].value_counts().to_frame().reset_index().rename({'index':'ingredient', 'ingredient_ok':'count'}, axis=1, inplace=False)	
	# recount all ingredients
	new_ingredients = count_and_filter_new_ingredients(df, min_occurrences = min_occurrences)
	# print
	print('Number of unique ingredients after union and threshold: {}'.format(len(new_ingredients['ingredient'].unique())))
	# create mask for surviving elements
	mask = df['ingredient_ok'].isin(new_ingredients['ingredient'])
	# print number of survivals
	print(mask.to_frame().value_counts())
	# filter ingredient mapping
	mapping_df = df[mask]
	# return mapping and new_ingredients
	return mapping_df, new_ingredients, all_ingredients


# 'totalTime', 'ingredientLines', 'attribution', 'name', 'rating', 'numberOfServings', 'yield',
# 'nutritionEstimates', 'source', 'flavors', 'images', 'attributes', 'id', 'totalTimeInSeconds'


if __name__ == '__main__':
	# define recipe folder
	recipe_folder = '../metadata'
	# get all recipes
	recipes = load_all(recipe_folder)
	print('ciao')
	# define minimum number of occurrences
	min_occurrences = 50 # 15
	# define minimum number of ingredients
	min_num_ingredients = 3
	# get mapping
	# mapping, new_ingredients = create_mapping(recipe_folder, min_occurrences = min_occurrences)
	mapping, new_ingredients, all_ingredients = create_mapping(recipes, min_occurrences = min_occurrences)
	# save all ingredients for debug
	all_ingredients.to_csv('all_ingredients.csv', index=False)
	all_ingredients.to_excel('all_ingredients.xlsx')
	# get ingredients similarity
	mapping, new_ingredients, ingredient_similarity = merge_with_ingredient_similarity(new_ingredients, mapping)
	# update recipes
	update_recipes(recipes, mapping, min_num_ingredients=min_num_ingredients)
	# save new ingredients
	new_ingredients.to_csv('ingredients_ok.csv', index=False)
	new_ingredients.to_excel('ingredients_ok.xlsx')
	# save ingredient similarity
	ingredient_similarity.to_csv('ingredient_similarity.csv', index=False)
	ingredient_similarity.to_excel('ingredient_similarity.xlsx')
	# save mapping
	mapping.to_csv('mapping.csv', index=False)
	mapping.to_excel('mapping.xlsx')
	# plot ingredients
	plt.figure(figsize=(10, 10))
	ax = sns.barplot(new_ingredients.iloc[:50], x="count", y="ingredient", orient='h', ax=plt.gca())
	# save figure
	plt.tight_layout()
	plt.savefig('ingredients_ok.png', bbox_inches='tight', dpi=600)
	# split in training, test and validation with seed
	random.seed(42)
	random.shuffle(recipes)
	# split
	n_train = int(0.8*len(recipes))
	n_val = int(0.1*len(recipes))
	n_test = len(recipes) - n_train - n_val
	# save
	with open('recipes_train.json', 'w') as f:
		json.dump(recipes[:n_train], f, indent=4)
	with open('recipes_val.json', 'w') as f:
		json.dump(recipes[n_train:n_train+n_val], f, indent=4)
	with open('recipes_test.json', 'w') as f:
		json.dump(recipes[n_train+n_val:], f, indent=4)
	# print('ciao')

	# # prova
	# # prova = {'ingredients': ['1 tablespoon / 3 tablespoons / 2 1/2 tablespoons / 1/4 tablespoon / 1table spoon / 1/4 to 1/2 tablespoon / 1/2 teaspoon / 1/2 to 1/4 tea spoon / 1/2 kg or 1 lb / 1 1/2 Lbs / 1/2 lb / 1/2 pound / 1/2 pounds / 1/2 lb. / 1/2 lbs / 28-oz. / 3 oz.']}
	# # process_ingredients(prova)
	# # print('ciao')
	# ingredients = {}
	# # find all json files inside the folder
	# recipe_folder = '../metadata'
	# cuisine_files = [os.path.join(recipe_folder, f) for f in os.listdir(recipe_folder) if f.endswith('.json')]
	# # initialize recipe count
	# recipe_count = 0
	# # for each cuisine
	# for cur_cuisine_file in tqdm(cuisine_files):
	# 	# read json
	# 	with open(cur_cuisine_file) as f:
	# 		cur_cuisine_data = json.load(f)
	# 	# update recipe count
	# 	recipe_count += len(cur_cuisine_data)
	# 	# for each recipe
	# 	for cur_recipe in cur_cuisine_data: 
	# 		# ['flavors', 'cuisine', 'name', 'ingredients', 'course', 'id']
	# 		cur_flavors = cur_recipe['flavors']
	# 		cur_cuisine = cur_recipe['cuisine']
	# 		cur_name = cur_recipe['name']
	# 		cur_ingredients = cur_recipe['ingredients']
	# 		cur_course = cur_recipe['course']
	# 		cur_id = cur_recipe['id']
	# 		# add ingredients to dict
	# 		for cur_ingredient in cur_ingredients:
	# 			ingredients[cur_ingredient] = ingredients.get(cur_ingredient, 0) + 1
	# 	# 	break
	# 	# break
	# # print recipe count and number of ingredients
	# print('Number of recipes: {}'.format(recipe_count))
	# print('Number of ingredients: {}'.format(len(ingredients)))

	# # transform in dataframe
	# df = pd.DataFrame.from_dict(ingredients, orient='index', columns=['count'])
	# # reset index
	# df = df.reset_index(names=['ingredient', 'count'])
	# # sort by count
	# df = df.sort_values(by=['count'], ascending=False)
	# # filter
	# tqdm.pandas()
	# df['ingredient_ok'] = df.progress_apply(process_ingredient_in_dataframe, axis=1)
	# # print the number of unique ingredients_ok
	# print('Number of unique ingredients: {}'.format(len(df['ingredient_ok'].unique())))
	# # switch axis
	# df = df[['ingredient', 'ingredient_ok', 'count']]
	# # recount all ingredients
	# new_ingredients = count_and_filter_new_ingredients(df, min_occurrences=15)
	# # create mask for surviving elements
	# mask = df['ingredient_ok'].isin(new_ingredients['ingredient'])
	# # print number of survivals
	# print(mask.to_frame().value_counts())
	# # filter ingredient mapping
	# mapping_df = df[mask]
	# # save to file
	# df.to_csv('ingredients.csv', index=False)
	# df.to_excel('ingredients.xlsx')
	# # keep ingredients with a count higher than 500
	# # df = df[df['count'] > 1000]
	# # keep first 50 ingredients
	# df = df.iloc[:50]
	# # save
	# # plot with seaborn
	# # ax = sns.barplot(df, x="ingredient", y="count", orient='v')
	# # plt.xticks(rotation=90)
	# # plt.figure(figsize=(10, 30))
	# # plt.figure(figsize=(10, 15))
	# plt.figure(figsize=(10, 10))
	# ax = sns.barplot(df, x="count", y="ingredient", orient='h', ax=plt.gca())
	# # save figure
	# plt.tight_layout()
	# plt.savefig('ingredients.png', bbox_inches='tight', dpi=600)
	# print('ciao')