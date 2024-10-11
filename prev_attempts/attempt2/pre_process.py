# from gpt4all import GPT4All
import pandas as pd
import os, json, re, random
from tqdm.rich import tqdm
import swifter
import matplotlib.pyplot as plt
from EasyPandas import EasyPandas

from utils import preprocess_ingredient, fast_classify

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
# sns.set_theme(style="darkgrid")
sns.set_style('darkgrid')
sns.set_theme(rc={'figure.figsize':(20,20)})



from swifter import set_defaults

set_defaults(
    npartitions=None,
    dask_threshold=1,
    scheduler="processes",
    progress_bar=True,
    progress_bar_desc=None,
    allow_dask_on_strings=False,
    force_parallel=False,
)


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



def create_mapping(recipes, min_occurrences=15):
	# define minimum number of occurrences
	min_occurrences = 50 # 15
	# define minimum number of ingredients
	min_num_ingredients = 3
	# init ingredient dictionary
	ingredients = {}
	# initialize recipe count
	recipe_count = len(recipes)
	# for each recipe
	for cur_recipe in recipes: 
		# get elements
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
	df['category'], df['subcategory'] = zip(*df['ingredient'].swifter.apply(fast_classify))
	# print
	# print(df.head())
	# switch axis and remove count
	df = df[['ingredient', 'category', 'subcategory']]
	# remove all ingredients that have null string as category
	df = df[df['category'] != '']
	# return
	return df


def update_recipes(recipes, mapping, min_num_ingredients=3):
	# convert mapping in a dictionary
	# mapping_dict = mapping.set_index('ingredient')['ingredient_ok'].to_dict()
	mapping_dict = mapping
	mapping_dict['combo'] = mapping_dict.apply(lambda x: [x['category'], x['subcategory']], axis=1)
	mapping_dict = mapping_dict.set_index('ingredient')
	mapping_dict = mapping_dict['combo'].to_dict()
	# keep track of recipes that have to be deleted
	to_be_deleted = []
	# for each recipe
	for i,cur_recipe in enumerate(tqdm(recipes)):
		# check if image exists
		# fn = os.path.join(self.root_dir, 'images', cuisine_name, recipe['id']+'.jpg')
		img_path = os.path.join('../images', cur_recipe['cuisine'], cur_recipe['id'] + '.jpg')
		if not os.path.exists(img_path):
			to_be_deleted.append(i)
			print(f'Image not found: {img_path}')
			continue
		# create new ingredient line
		cur_recipe['ingredients_ok'] = []
		# for each ingredient
		for cur_ingredient in cur_recipe['ingredients']:
			# check if considered
			if cur_ingredient in mapping_dict:
				new_name = mapping_dict[cur_ingredient]
				cur_recipe['ingredients_ok'].append(new_name)
		# remove duplicates
		unique_data = [list(x) for x in set(tuple(x) for x in cur_recipe['ingredients_ok'])]
		cur_recipe['ingredients_ok'] = unique_data
		# at the end, check if there are more than min_num_ingredients
		if len(cur_recipe['ingredients_ok']) < min_num_ingredients:
			# delete this recipe
			to_be_deleted.append(i)
	# delete recipes with too few ingredients
	for cur_id in reversed(to_be_deleted):
		del recipes[cur_id]
	# return recipes
	return recipes




if __name__ == '__main__':
	# define recipe folder
	recipe_folder = '../metadata'
	# define minimum number of ingredients
	min_num_ingredients = 3
	# get all recipes
	recipes = load_all(recipe_folder)
	# debug keep only 1000 recipes
	# recipes = recipes[:1000]
	# convert ingredients
	mapping = create_mapping(recipes)
	# save mapping
	mapping.to_csv('mapping.csv', index=False)
	mapping.to_excel('mapping.xlsx')
	# plot category
	counts = mapping.groupby(by=['category', 'subcategory']).count().reset_index().rename({'ingredient':'counts'}, axis=1)
	# print(mapping['category'].value_counts())
	# categories = mapping['category'].value_counts().to_frame().reset_index().rename({'index':'category', 'category':'counts'}, axis=1)
	categories = counts.groupby(by='category').sum('ingredient').reset_index().sort_values(by='counts', ascending=False)
	# import ipdb; ipdb.set_trace()
	# mapping['category'].value_counts().plot(kind='bar')
	sns.barplot(data=categories, y='category', x='counts', orient='h')
	plt.savefig('category.png')
	# plot subcategory
	# mapping['subcategory'].value_counts().plot(kind='bar')
	# sns.barplot(data=mapping['subcategory'].value_counts(), x='value', y='counts', orient='h')
	# subcategories = mapping['subcategory'].value_counts().to_frame().reset_index().rename({'index':'subcategory', 'subcategory':'counts'}, axis=1)
	# subcategories = counts.groupby(by='subcategory').sum('ingredient').reset_index().rename({'ingredient':'counts'}, axis=1)
	# sns.barplot(data=subcategories, y='subcategory', x='counts', orient='h')
	plt.clf()
	sns.barplot(data=counts.sort_values(by='counts', ascending=False), y='subcategory', x='counts', orient='h', hue='category') # , errorbar=None) #, colormap='Paired')
	# ax = counts.T.plot(kind='bar', x='subcategory', y='counts', label='category', colormap='Paired')
	plt.savefig('subcategory.png')
	# update recipes
	recipes = update_recipes(recipes, mapping, min_num_ingredients=min_num_ingredients)
	# get ingredients similarity
	# mapping, new_ingredients, ingredient_similarity = merge_with_ingredient_similarity(new_ingredients, mapping)

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