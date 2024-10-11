import os
import matplotlib

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
INPUT_PATH = os.path.join(DATA_PATH, 'input')
RAW_INPUT_PATH = os.path.join(DATA_PATH, 'raw_input')
METADATA_FILENAME = 'metadata.json'

YUMMLY_PATH = os.path.join(INPUT_PATH, 'yummly')
YUMMLY_RECIPES_PATH = os.path.join(YUMMLY_PATH, 'recipes_general')
YUMMLY_METADATA_PATH = os.path.join(YUMMLY_PATH, 'metadata_general')  # not used anymore

RAW_YUMMLY_PATH = os.path.join(RAW_INPUT_PATH, 'yummly')
RAW_YUMMLY_IMAGES_PATH = os.path.join(RAW_YUMMLY_PATH, 'images')
RAW_YUMMLY_RECIPES_PATH = os.path.join(RAW_YUMMLY_PATH, 'recipes')
RAW_YUMMLY_METADATA_PATH = os.path.join(RAW_YUMMLY_PATH, 'metadata')

RAW_RECIPES_PATH = os.path.join(RAW_INPUT_PATH, 'recipes')
RAW_RECIPES1M_PATH = os.path.join(RAW_INPUT_PATH, 'recipes1M')

RECIPES_PATH = os.path.join(INPUT_PATH, 'recipes')
RECIPES1M_PATH = os.path.join(INPUT_PATH, 'recipes1M')

IMG_STATS_FILENAME = 'train_images_stats.csv'

APP_DATA = os.path.join(DATA_PATH, 'app_data')
BLANK_IMG_PATH = os.path.join(APP_DATA, 'blank.jpg')
DASH_PATH = os.path.join(PROJECT_PATH, 'src', 'dashboards', 'dash')

EXPERIMENTS_PATH = os.path.join(PROJECT_PATH, 'experiments')
EXPERIMENTS_TRASH_PATH = os.path.join(EXPERIMENTS_PATH, '__trash__')
EXPERIMENTS_WANDB_PATH = os.path.join(EXPERIMENTS_PATH, 'wandb')


GLOVE_EMBEDDINGS_PATH = os.path.join(DATA_PATH, 'glove_embeddings')

for path in [INPUT_PATH, YUMMLY_PATH, YUMMLY_RECIPES_PATH, YUMMLY_METADATA_PATH, RECIPES_PATH, RECIPES1M_PATH,
             EXPERIMENTS_PATH, EXPERIMENTS_TRASH_PATH, EXPERIMENTS_WANDB_PATH, GLOVE_EMBEDDINGS_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

# Data settings
VAL_SIZE = 0.08
TEST_SIZE = 0.08
FOOD_CATEGORIES = ['american', 'chinese', 'french', 'greek', 'indian', 'italian', 'japanese', 'mexican',
                   'spanish', 'thai', 'all']

DEF_UNKNOWN_TOKEN: str = '<UNK>'
DEF_PAD_TOKEN: str = '<PAD>'
DEF_MAKS_TOKEN: str = '<MASK>'
DEF_NONE_TOKEN: str = '<NONE>'

DEF_BATCH_SIZE = 128
DEF_IMAGE_SHAPE = (224, 224)
DEF_LR = 1e-3
DEF_LR_INIT = min(DEF_LR * 100, 1e-1)
DISABLE_RESUME = False
OPTUNA_JOURNAL_FILENAME = 'journal.log'
OPTUNA_JOURNAL_PATH = os.path.join(EXPERIMENTS_PATH, OPTUNA_JOURNAL_FILENAME)
DEF_N_TRIALS = 20
HTUNER_CONFIG_FILE = 'hparam_config.json'

WANDB_PROJECT_NAME = 'ingredient_recognition'


LP_MAX_PHASE = 4  # LAYER-WISE PRETRAINING MAX PHASES

# matplotlib.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (5, 5)
matplotlib.rcParams['axes.grid'] = False

