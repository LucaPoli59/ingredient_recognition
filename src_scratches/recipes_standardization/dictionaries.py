# Next, build a list of adjectives, like "diced" or "blanched"

mods_1 = ['baked', 'blanched', 'blackened', 'braised', 'breaded', 'broiled', 'caramelized', 'charred', 'fermented',
         'fried',
         'glazed', 'infused', 'marinated', 'poached', 'roasted', 'sauteed', 'seared', 'smoked', 'whipped']

# -ed adjectives.  Found another good list online.
mods_2 = ['diced', 'battered', 'blackened', 'blanched', 'blended', 'boiled', 'boned', 'braised', 'brewed', 'broiled',
         'browned', 'butterflied', 'candied', 'canned', 'caramelized', 'charred', 'chilled', 'chopped', 'clarified',
         'condensed', 'creamed', 'crystalized', 'curdled', 'cured', 'curried', 'dehydrated', 'deviled', 'diluted',
         'dredged', 'drenched', 'dried', 'drizzled', 'dry roasted', 'dusted', 'escalloped', 'evaporated',
         'fermented',
         'filled', 'folded', 'freeze dried', 'fricaseed', 'fried', 'glazed', 'granulated', 'grated', 'griddled',
         'grilled',
         'hardboiled', 'homogenized', 'kneaded', 'malted', 'mashed', 'minced', 'mixed', 'medium', 'small', 'large',
         'packed', 'pan-fried', 'parboiled', 'parched', 'pasteurized', 'peppered', 'pickled', 'powdered',
         'preserved',
         'pulverized', 'pureed', 'redolent', 'reduced', 'refrigerated', 'chilled', 'roasted', 'rolled', 'salted',
         'saturated', 'scalded', 'scorched', 'scrambled', 'seared', 'seasoned', 'shredded', 'skimmed', 'sliced',
         'slivered', 'smothered', 'soaked', 'soft-boiled', 'hard-boiled', 'stewed', 'stuffed', 'toasted', 'whipped',
         'wilted', 'wrapped']

modifiers = list(set(mods_1 + mods_2))  # set to avoid overlapping words

units = list(
    {'l', 'dl', 'milliliter', 'liter', 'deciliter', 'teaspoon', 't.', 'tsp.', 'milliliters', 'liters', 'deciliters',
     'teaspoons', 't.', 'tsp.', 'tablespoon', 'T.', 'tbsp.', 'ounce', 'fl oz', 'cup', 'c.', 'pint', 'pt.',
     'tablespoons', 'ounces', 'fl ozs', 'cups', 'pints', 'quarts', 'gallons', 'grams', 'kilograms', 'quart', 'qt.',
     'gallon', 'gal', 'mg', 'milligram', 'g', 'gram', 'kg', 'kilogram', 'milligrams', 'pound', 'lb', 'ounce', 'oz',
     'count', 'pints', 'quarts', 'cups', 'tablespoons', 'pounds', 'lbs', 'ounces', 'units', 'drops', 'tsps.', 'tbsps.',
     'Ts.', 'ts.', 'teaspoons', 'dash', 'pinch', 'drop', 'dram', 'smidgeon', 'dashes', 'pinches', 'drops', 'drams',
     'smidgeons'})

quantities_dict = {
    "1/2": 0.5,
    "1/4": 0.25,
    "1/3": 0.333,
    "2/3": 0.666,
    "3/4": 0.75,
    "half": 0.5,
    "third": 0.333,
    "quarter": 0.25,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "11": 11,
    "12": 12,
    "a dozen": 12,
    "a baker's dozen": 13,
    "two dozen": 24,
    "three dozen": 36,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12
}