import os, re




############################# BASE PROCESSING ##############################

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
	# remove multiple spaces
	ingredient = re.sub(r'\s{2,}', ' ', ingredient)
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
				"Cupful", "Gallon", "Few", "Several", "Dozen", "Score", "Bulk", "Finely", "Fine", "Good quality", 
				"Premium", "Superior", "Excellent", "Choice", "Select", "Prime", "First-rate", "High-grade", "Coarsely", 
				"Coarse", "Thickly", "Thick", "Thinly", "Thin", "Sliced", "Chopped", "Diced", "Minced", "Shredded", "Grated", 
				"Ground", "Lightly", "Light", "Heavily", "Heavy", "Strong", "Weak", "Mild", "Intense", "Moderate", "Sparse", 
				"Dense", "Rich", "Sparse", "Occasional", "Regular", "Frequent", "Constant", "Intermittent", "Rare", "Seldom", 
				"Often", "Always", "Never", "Generous", "Sparse", "Abundant", "Scant", "Meager", "Plentiful", "Copious", "Vast", 
				"Minimal", "Substantial", "Significant", "Negligible", "big", "extralarge", "slices"]
	# remove them and return
	return remove_list_of_elements(ingredient, quantifiers)
	# "Quart",


############################## TREATMENTS AND TEMPERATURES ##############################

def remove_treatments(ingredient):
	treatments = ["grated", "chopped", "sliced", "diced", "minced", "shredded", "pitted", "halved", "quartered", "quart", "peeled", "crushed",
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



def preprocess_ingredient(cur_ingredient):
    cur_ingredient = remove_paranthesis_and_markups(cur_ingredient)
    cur_ingredient = remove_quantifiers(cur_ingredient)
    cur_ingredient = remove_textual_quantifiers(cur_ingredient)
    cur_ingredient = remove_treatments(cur_ingredient)
    cur_ingredient = remove_temperature(cur_ingredient)
    cur_ingredient = cur_ingredient.strip(" ;,./-()")
    return cur_ingredient





def is_in(ingredient, list_of_ingredients):
    for cur_ingredient in list_of_ingredients:
        if cur_ingredient.lower() in ingredient.lower():
            return True
    return False


def fast_classify(ingredient):

    # FUNDAMENTALS
    if is_in(ingredient, ['water', 'acqua', 'wasser', 'eau', 'agua', 'aqua']):
        return 'Fundamentals', 'Water'
    if is_in(ingredient, ['salt', 'salz', 'sale']):
        return 'Fundamentals', 'Salt'

    # ALCOL
    if is_in(ingredient, ['vodka', 'gin', 'rum', 'whiskey', 'tequila', 'brandy', 'cognac', 'bourbon']):
        return 'Alcohol', 'Liquor'
    
    if is_in(ingredient, ['wine', 'champagne', 'prosecco', 'moscato', 'riesling', 'pinot', 'merlot', 
                          'cabernet', 'sauvignon', 'chardonnay', 'zinfandel', 'syrah', 'shiraz', 'malbec',
                          'tempranillo', 'sangiovese', 'chenin', 'gewürztraminer', 'viognier', 'vermentino', 
                          'albariño', 'rosé', 'wein',
                          ]):
        return 'Alcohol', 'Wines'

    if is_in(ingredient, ['vermouth', 'port', 'marsala', 'sherry', 'calvados', 'grappa', 'madeira']):
        return 'Alcohol', 'Fortified Wines and Spirits'
    
    if is_in(ingredient, ['beer', 'lager', 'stout', 'cider']):
        return 'Alcohol', 'Beers and Ciders'
    
    
    if is_in(ingredient, ['liqueur', 'amaretto', 'cointreau', 'frangelico', 'schnapps', 'kirsch', 'Whiskey', 
                          'Scotch', 'Bourbon', 'Rye', 'Vodka', 'Rum', 'Tequila', 'Gin', 'Brandy', 'Cognac', 
                          'Armagnac', 'Mezcal', 'Absinthe', 'Irish Cream', 'Grand Marnier', 'Jägermeister',
                          'Kahlúa', 'Baileys', 'Sambuca', 'Pernod', 'Whisky',
                          ]):
        return 'Alcohol', 'Liqueurs'
    
    # DAIRY
    if is_in(ingredient, ['milk', 'Milch','latte']): #, 'lactose', 'lactaid', 'lactase', 'lactobacillus', 'lactobacilli', 'lactobacillus']):
        return 'Dairy and Eggs', 'Milk'
    
    if is_in(ingredient, ['cheese', 'cheddar', 'mozzarella', 'parmesan', 'gouda', 'brie', 'feta', 'goat', 'ricotta', 
                          'swiss', 'blue', 'provolone', 'colby', 'monterey', 'jack', 'pepper jack', 'muenster', 
                          'havarti', 'limburger', 'boursin', 'camembert', 'edam', 'emmental', 'gruyère', 'havarti', 
                          'jarlsberg', 'manchego', 'mascarpone', 'neufchâtel', 'paneer', 'queso', 'roquefort', 
                          'stilton', 'teleme', 'wensleydale', 'yarg', 'yorkshire', 'burrata', 'halloumi', 'queso',
                          'queso fresco', 'queso blanco', 'cotija', 'queso panela', 'queso oaxaca', 'queso asadero',
                          'queso chihuahua', 'queso manchego', 'queso ibérico', 'queso de bola', 'queso de cabra',
                          'parmigiano', 'reggiano', 'pecorino', 'asiago', 'grana padano', 'gorgonzola', 'gruyere',
                          'taleggio', 'fontina', 'bel paese', 'caciocavallo', 'caciotta', 'Raclette', 'scamorza',
                          'fromage']):
        return 'Dairy and Eggs', 'Cheese'

    if is_in(ingredient, ['yogurt', 'yoghurt', 'yoghourt', 'yogourt']):
        return 'Dairy and Eggs', 'Yogurt'

    if is_in(ingredient, ['egg', 'eggs', 'eier', 'eigelb']):
        return 'Dairy and Eggs', 'Eggs'

    if is_in(ingredient, ['butter', 'cream', 'margarine', 'half-and-half', 'half and half', 'ghee', 'sahne', 'panna',
                          'schlagsahne', 'creme fraiche', 'crème fraîche', 'crème fraiche', 'clotted cream', 'whipped cream',
                          'sour cream', 'half & half', 'beurre', 'country crock®', 'spread']):
        return 'Dairy and Eggs', 'Butter and Cream'

    # FRUITS AND VEGETABLES
    
    # Ortaggi da fiore: broccoli, carciofi, cavolfiori.
    # Flowering vegetables: broccoli, artichokes, cauliflower.
    # capers are wrong here but..
    if is_in(ingredient, ['broccoli', 'artichoke', 'cauliflower', 'broccolo', 'cavolfiore', 'cavolfiori', 'cabbage',
                          'sprouts', 'kale', 'broccol', 'blossom', 'argula', 'cavolo', 'cavoli', 'zucchini', 'capers',
                          'carciof', 'Artischocke', 'brokkoli']):
        return 'Fruits and Vegetables', 'Flowering Vegetables'
    
    # Ortaggi da frutto: cetriolo, melanzane, peperoni, pomodori, zucchine.
    # fruit vegetables: cucumber, eggplant, pepper, tomato, zucchini.
    if is_in(ingredient, ['cucumber', 'eggplant', 'pepper', 'tomato', 'zucchini', 'cetriolo', 'cetrioli', 'melanzane', 
                          'melanzana', 'peperone', 'peperoni', 'pomodoro', 'pomodori', 'zucchina', 'zucchine',
                          'chilies', 'chili', 'chili pepper', 'peperoncino', 'peperoncini', 'jalapeño', 'passata',
                          'pumpkin', 'tomatillos', 'gourd', 'jalapeno', 'masala', 'chilli', 'okra', 'cayenne',
                          'capsicum', 'kapern', 'capper']):
        return 'Fruits and Vegetables', 'Fruit Vegetables'
    
    # Ortaggi da seme: ceci, fagioli, fave, lenticchie, piselli, soia.
    # Seed vegetables: chickpeas, beans, broad beans, lentils, peas, soybeans.
    if is_in(ingredient, ['chickpea', 'beans', 'lentil', 'peas', 'soybeans', 'fagioli', 'fagiolo', 'lenticchie', 'piselli',
                          'soia', 'ceci', 'fave', 'lenticchie', 'moong dal', 'chana dal', 'urad dal', 'haricots verts']):
        return 'Fruits and Vegetables', 'Seed Vegetables'
    
    # Ortaggi da foglia: bietola, cicoria, lattuga, spinaci.
    # Leaf vegetables: chard, chicory, lettuce, spinach.
    if is_in(ingredient, ['chard', 'chicory', 'lettuce', 'spinach', 'bietola', 'cicoria', 'lattuga', 'spinaci', 'lattughe', 
                          'basil', 'coriander', 'cilantro', 'arugula', 'rucola', 'baby greens', 'dandelion greens',
                          'radicchio', 'endive', 'escarole', 'watercress', 'collard greens', 'kale', 'mustard greens', 'romaine',
                          'rhubarb', 'baby salad', 'salad', 'iceberg', 'mixed greens', 'frisee', 'mesclun', 'blattspinat']):
        return 'Fruits and Vegetables', 'Leaf Vegetables'

    # Ortaggi da radice: barbabietola, carota, rapa, ravanello, sedano rapa.
    # Root vegetables: beet, carrot, turnip, radish, celeriac
    if is_in(ingredient, ['beet', 'carrot', 'turnip', 'radish', 'celeriac', 'barbabietola', 'carota', 'rapa', 'ravanello', 
                          'sedano rapa', 'parsnip', 'rutabaga', 'salsify', 'carot', 'Karotte', 'jicama', 'möhren', 'rübe',
                          'möhre']):
        return 'Fruits and Vegetables', 'Root Vegetables'
    
    # Ortaggi da fusto: asparagi, finocchi, sedano.
    # Stem vegetables: asparagus, fennel, celery.
    if is_in(ingredient, ['asparagus', 'fennel', 'celery', 'asparagi', 'finocchi', 'sedano', 'asparagus', 'asparago', 'asparagi']):
        return 'Fruits and Vegetables', 'Stem Vegetables'
    
    # Ortaggi da tubero: batata, patata.
    # Tuber vegetables: yam, potato.
    if is_in(ingredient, ['yam', 'potato', 'patata', 'patate', 'kartoffel']):
        return 'Fruits and Vegetables', 'Tuber Vegetables'
    
    # Ortaggi da bulbo: aglio, cipolla, porro, scalogno.
    # Bulb vegetables: garlic, onion, leek, shallot.
    if is_in(ingredient, ['garlic', 'onion', 'leek', 'shallot', 'scallion', 'aglio', 'cipolla', 'cipolle', 'porro', 'scalogno',
                          'zwiebel', 'Knoblauch', 'schalotte']):
        return 'Fruits and Vegetables', 'Bulb Vegetables'
    
    if is_in(ingredient, ['tamarind', 'vanilla', 'cocoa', 'cacao']):
        return 'Fruits and Vegetables', 'Plants and Flowers'
    
    
    if is_in(ingredient, ['fruit', 'apple', 'apples', 'banana', 'bananas', 'berry', 'berries', 'ananas', 'ananas', 'avocado',
                          'mango', 'kiwi', 'pear', 'peach', 'pera', 'pesca', 'pesche', 'anguria', 'angurie', 'lemon', 'lime',
                          'squash', 'dates', 'Apricot', 'cherry', 'plum', 'olive', 'plantain', 'chilli', 'chillies', 'Tomate',
                          'cherries', 'orange', 'figs', 'prune', 'limone', 'Zitrone', 'raisins', 'grape', 'uva', 'clementine',
                          'Zitronat', 'mandarin', 'mandarino', 'mandarini', 'pomelo', 'goya', 'currants', 'cantaloupe',
                          'papaya', 'nectarines']):
        return 'Fruits and Vegetables', 'Fruits'
    
    
    if is_in(ingredient, ['mushroom', 'fungo', 'funghi', 'porcini', 'shiitake', 'portobello', 'truffle', 'chanterelle', 'tartufo']):
        return 'Mushrooms', 'Mushrooms'

    # GRAINS AND CEREALS    
    # cereals are missing
    if is_in(ingredient, ['flour', 'wheat', 'farina', 'mehl', 'harina', 'ladyfinger', 'lady finger', 'cereal', 'semolin']):
        return 'Grains and Cereals', 'Wheat'
    

    if is_in(ingredient, ['rice']):
        return 'Grains and Cereals', 'Rice'

    if is_in(ingredient, ['oat', 'avena', 'granola', 'muesli', 'porridge']):
        return 'Grains and Cereals', 'Oats'

    if is_in(ingredient, ['corn', 'maize', 'popcorn', 'pannocchia', 'pannocchie', 'popcorn', 'hominy', 'grits']):
        return 'Grains and Cereals', 'Corn'

    if is_in(ingredient, ['quinoa', 'amaranth', 'teff', 'spelt', 'farro', 'emmer', 'kamut', 'Khorasan', 'Freekeh',
                          'Millet', 'Sorghum', 'Einkorn', 'barley', 'orzo', 'bulgur']):
        return 'Grains and Cereals', 'Ancient Grains'

    # MEATS AND POULTRY
    if is_in(ingredient, ['beef', 'steak', 'ground', 'roast', 'tallow', 'oxtail', 'veal', 'brisket', 'meat', 'sirloin']):
        return 'Meats and Poultry', 'Beef'

    if is_in(ingredient, ['chicken', 'turkey', 'duck', 'Huhn', 'hähnchenfilet', 'hähnchenbrust', 'pollo',
                          'hähnchenfilets', 'hähnchenbrüste', 'hähnchen', 'huhn', 'poulet', 'poulets', 'pouletbrust', 'pouletbrüste',
                          ]):
        return 'Meats and Poultry', 'Poultry'

    if is_in(ingredient, ['pork', 'chops', 'bacon', 'ham', 'pancetta', 'prosciutto', 'coppa', 'guanciale', 'lardo', 
                          'salame', 'salumi', 'Mortadella', 'Salsiccia', 'Salsicce', 'chorizo', 'lard', 'sausage',
                          'salami', 'capicola', 'capocollo', 'capicol', 'schinken', 'speck']):
        return 'Meats and Poultry', 'Pork'

    if is_in(ingredient, ['lamb', 'goat']):
        return 'Meats and Poultry', 'Lamb and Goat'

    if is_in(ingredient, ['venison', 'bison', 'rabbit']):
        return 'Meats and Poultry', 'Game Meats'

    # SEAFOOD
    if is_in(ingredient, ['fish', 'salmon', 'tuna', 'cod', 'tonno', 'salmone', 'merluzzo', 'pesce', 'anchovy', 
                          'anchovies', 'acciuga', 'acciughe', 'squid', 'bass', 'sardell', 'halibut', 'fregola',
                          'sardines', 'trout', 'tilapia', 'flounder', 'platessa']):
        return 'Seafood', 'Fish'

    if is_in(ingredient, ['shrimp', 'crab', 'lobster', 'gamberetto', 'gamberetti', 'granchio', 'granchi', 'Garnelen', 'scallops', 'prawn']):
        return 'Seafood', 'Shellfish'

    if is_in(ingredient, ['clam', 'oyster', 'mussel', 'vongola', 'vongole', 'cozza', 'cozze', 'ostrica', 'ostriche']):
        return 'Seafood', 'Mollusks'

    if is_in(ingredient, ['seaweed', 'algae']):
        return 'Seafood', 'Seaweed and Algae'

    # FATS AND OILS
    if is_in(ingredient, ['oil', 'olive', 'canola', 'coconut', 'öl', 'cooking spray', 'shortening']):
        return 'Fats and Oils', 'Vegetable Oils'

    # if is_in(ingredient, ['tallow']): # butter, lard and tall removed
    #     return 'Fats and Oils', 'Animal Fats'

    if is_in(ingredient, ['sesame', 'peanut', 'almond', 'sesamo', 'nocciole', 'mandorle', 'nocciola', 'mandorla']):
        return 'Fats and Oils', 'Nut and Seed Oils'

    # SPICES AND HERBS
    if is_in(ingredient, ['spice', 'cumin', 'paprika', 'turmeric', 'curcuma', 'chipotle', 'chile', 'adobo', 
                          'powder', 'cinnamon', 'oregano', 'clove', 'pimento', 'cardamom', 'saffron', 'seitan', 
                          'asafoetida', 'laurel', 'alloro', 'bay leaves', 'bay leaf', 'nutmeg', 'kardamom', 'anise', 'anice',
                          'shahi jeera', 'asafetida', 'jeera', 'mace']):
        return 'Spices and Herbs', 'Dried Spices'

    if is_in(ingredient, ['fresh herb', 'rosemary', 'mint', 'thyme', 'tarragon', 'savory', 'rosmarin', 'lavender', 'sage',
                          'prezzemolo', 'parsley', 'Petersilie', 'dill', 'epazote', 'chives', 'fenugreek', 'marjoram',
                          'stiele thymian', 'chervil', 'blätter salbei']): # basilico 'prezzemolo', 'coriandolo' (aka 'cilantro')
        return 'Spices and Herbs', 'Fresh Herbs'

    if is_in(ingredient, ['curry', 'Seasoning', 'italian seasoning', 'erbsen', 'kräuter', 'herbes', 'herbes', 'herbes', 'masala']):
        return 'Spices and Herbs', 'Blends and Mixes'

    # SWEETENERS
    if is_in(ingredient, ['sugar', 'zucchero', 'Zucker']):
        return 'Sweeteners', 'Sugars'

    if is_in(ingredient, ['syrup', 'maple', 'agave', 'honey', 'sciroppo', 'sciroppi', 'sciroppo', 'sciroppi', 'molasses']):
        return 'Sweeteners', 'Syrups'

    if is_in(ingredient, ['stevia', 'sucralose']):
        return 'Sweeteners', 'Artificial Sweeteners'

    # NUTS AND SEEDS
    if is_in(ingredient, ['almond', 'walnut', 'pecan', 'mandorle', 'noci', 'hazelnut', 'chestnut', 'pistachio',
                          'cashew', 'pine nuts', 'pinenuts', 'peanut', 'arachidi', 'arachide', 'pinol', 'pinienkerne', 'nuts']): # DANGER: nuts
        return 'Nuts and Seeds', 'Nuts'

    if is_in(ingredient, ['seed', 'pumpkin seed', 'sunflower seed', 'sesame', 'zucca', 'girasole', 'sesamo']):
        return 'Nuts and Seeds', 'Seeds'


    # BAKING INGREDIENTS
    if is_in(ingredient, ['yeast', 'baking', 'baking powder', 'lievito', 'lieviti', 'würfel hefe']):
        return 'Baking Ingredients', 'Leavening Agents'

    if is_in(ingredient, ['cornstarch', 'gelatin', 'amido', 'tostada shell']):
        return 'Baking Ingredients', 'Thickening Agents'

    # if is_in(ingredient, ['chocolate', 'cocoa', 'cioccolato', 'cacao']):
    #     return 'Baking Ingredients', 'Chocolate and Cocoa'

    # CONDIMENTS AND SAUCES
    if is_in(ingredient, ['vinegar', 'balsamic', 'aceto', 'essig']):
        return 'Condiments and Sauces', 'Vinegars'

    if is_in(ingredient, ['soy', 'tomato', 'sauce', 'salsa', 'salse', 'Worcestershire', 'mustard', 'ketchup', 'mayonnaise', 
                          'pickles', 'guacamole', 'guacamol','chutney', 'hummus', 'aioli', 'pico de gallo', 'country crock', # DANGER: country crock
                          'mayo']):
        return 'Condiments and Sauces', 'Sauces'

    if is_in(ingredient, ['dressing', 'marinade', 'vinaigrette']):
        return 'Condiments and Sauces', 'Dressings and Marinades'

    # BEVERAGES (NON-ALCOHOLIC)
    if is_in(ingredient, ['coffee', 'tea', 'matcha', 'caffè', 'tè', 'espresso']):
        return 'Beverages (Non-Alcoholic)', 'Coffees and Teas'
    
    if is_in(ingredient, ['juice', 'succo', 'succhi']):
        return 'Beverages (Non-Alcoholic)', 'Juices'

    if is_in(ingredient, ['soft drink', 'energy', 'soda', 'sodas', 'cola', 'pepsi', 'fanta', 'sprite', '7up', 'red bull',
                          'monster', 'rockstar', 'gatorade', 'powerade', 'vitamin water', 'aquarius', 'lucozade', 'tizer',
                          'irn bru', 'faygo', 'jolt', 'tab', 'surge', 'vault', 'mello yello', 'fresca', 'squirt', 'sunkist',
                          'crush', 'faygo', 'fanta', 'mirinda', 'schweppes', 'tonic', 'tonic water', 'ginger ale', 'ginger beer',
                          'root beer', 'rootbeer', 'dr pepper', 'seltzer', 'seltz'
                          ]):
        return 'Beverages (Non-Alcoholic)', 'Soft Drinks and Energy Drinks'
    

    # at the end, ice
    if is_in(ingredient, ['ice', 'ghiaccio', 'eis', 'glace']):
        return 'Fundamentals', 'Water'
    

    # COMPOSITE INGREDIENTS

    # Processed culinary ingredients
    # Ingredients that have been processed but are used as basic cooking ingredients
    if is_in(ingredient, ['pasta', 'spaghetti', 'penne', 'fusilli', 'maccheroni', 'maccheroni', 'farfalle', 'noodle', 
                          'gnocchi', 'dough', 'fettuccine', 'macaroni', 'tortilla', 'tacos', 'rigatoni', 'frosting',
                          'pesto', 'broth', 'pita', 'naan','roll', 'bun', 'ramen', 'soba', 'udon', 'lasagna', 'lasagne',
                          'linguin', 'bucatini', 'sauerkraut', 'krauti', 'relish',
                          'coloring', 'topping', 'pappardelle', 'tagliatelle', 'couscous', 'cous cous', 'cous-cous',
                          'Nudeln', 'Schokolade', 'cioccolato', 'chocolate', 'cacao', 'cocoa', 'tempeh', 'Gemüsebrühe', 
                          'brodo', 'raviol', 'orecchiette', 'cavatelli', 'cavatappi', 'Ragù', 'vanilla extract',
                          'croutons', 'panko', 'wonton', 'ziti', 'taco shell', 'cannelloni', 'manicotti', 'tortellini', 
                          'taco shells', 'tostada shells', 'shells', 'hühnerbrühe']):
        return 'Composite Ingredients', 'Processed Culinary Ingredients'

    # Processed Food Products
    # more complex products made from combining processed ingredients with minimal processing ingredients.
    if is_in(ingredient, ['bread', 'baguette', 'brioche', 'ciabatta', 'focaccia', 'sourdough', 'chapati',
                          'pizza', 'pizzas', 'pizze', 'pizze', 'meatball', 'polpett', 'cookies', 'biscotti', 'biscuit',
                          'marshmallow', 'candy', 'candies', 'caramelle', 'caramella', 'ciabatta', 'cappuccino', 'coleslaw',
                          'kimchi', 'muffin', 'brownie', 'pie', 'tart', 'torte', 'torta', 'cake', 'french fries', 'fries',
                          'polenta', 'pastry', 'pretzel', 'waffle', 'pancake', 'Sonntagsbrötchen', 'pane', 'panini', 'panino',
                          'tofu', 'dulce de leche', 'wafer', 'biskuit', 'kekse']):
        return 'Composite Ingredients', 'Processed Food Products'
    
    return '', ''