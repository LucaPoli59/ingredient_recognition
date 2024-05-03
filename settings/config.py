import os
import matplotlib

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
INPUT_PATH = os.path.join(DATA_PATH, 'input')
RAW_INPUT_PATH = os.path.join(DATA_PATH, 'raw_input')

IMAGES_PATH = os.path.join(INPUT_PATH, 'images')
RECIPES_PATH = os.path.join(INPUT_PATH, 'recipes')
METADATA_PATH = os.path.join(INPUT_PATH, 'metadata')

RAW_IMAGES_PATH = os.path.join(RAW_INPUT_PATH, 'images')
RAW_RECIPES_PATH = os.path.join(RAW_INPUT_PATH, 'recipes')
RAW_METADATA_PATH = os.path.join(RAW_INPUT_PATH, 'metadata')

EXPERIMENTS_PATH = os.path.join(PROJECT_PATH, 'experiments')
EXPERIMENTS_TRASH_PATH = os.path.join(EXPERIMENTS_PATH, '__trash__')

for path in [INPUT_PATH, IMAGES_PATH, RECIPES_PATH, METADATA_PATH, EXPERIMENTS_PATH, EXPERIMENTS_TRASH_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

# Data settings
VAL_SIZE = 0.08
TEST_SIZE = 0.08
FOOD_CATEGORIES = ['american', 'chinese', 'french', 'greek', 'indian', 'italian', 'japanese', 'mexican',
                   'spanish', 'thai']
DEF_BATCH_SIZE = 128
DEF_LR = 1e-3

# matplotlib.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (5, 5)
matplotlib.rcParams['axes.grid'] = False

