from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from external_code.helpers import plot
import os
import pandas as pd

from data_processing.ImagesRecipesDataset import ImagesRecipesDataset
from data_processing.MultiLabelBinarizerRobust import MultiLabelBinarizerRobust
from config import *

# for category in os.listdir(RAW_IMAGES_PATH):
#     print(len(os.listdir(os.path.join(RAW_IMAGES_PATH, category))))

encoder = MultiLabelBinarizerRobust()


train_dataset = ImagesRecipesDataset(os.path.join(IMAGES_PATH, "train"), os.path.join(RECIPES_PATH, 'train.json'),
                                     label_encoder=encoder)

val_dataset = ImagesRecipesDataset(os.path.join(IMAGES_PATH, "val"), os.path.join(RECIPES_PATH, 'val.json'),
                                   label_encoder=encoder)


print(pd.DataFrame(train_dataset.recipes)['cuisine'].value_counts())
print(pd.DataFrame(val_dataset.recipes)['cuisine'].value_counts())  # Todo fix the imbalance in the dataset classes