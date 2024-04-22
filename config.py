import os
import matplotlib

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
IMAGES_PATH = os.path.join(DATA_PATH, 'images')

matplotlib.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (5, 5)
matplotlib.rcParams['axes.grid'] = False

