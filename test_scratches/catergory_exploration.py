import os

from config import *


for category in os.listdir(RAW_IMAGES_PATH):
    print(len(os.listdir(os.path.join(RAW_IMAGES_PATH, category))))

