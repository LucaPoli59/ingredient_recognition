import pandas as pd

from src.data_processing.ImagesRecipesDataset import ImagesRecipesDataset
from src.data_processing.MultiLabelBinarizerRobust import MultiLabelBinarizerRobust
from settings.config import *

# for category in os.listdir(RAW_IMAGES_PATH):
#     print(len(os.listdir(os.path.join(RAW_IMAGES_PATH, category))))

encoder = MultiLabelBinarizerRobust()


train_dataset = ImagesRecipesDataset(os.path.join(IMAGES_PATH, "train"), os.path.join(RECIPES_PATH, 'train.json'),
                                     label_encoder=encoder)

val_dataset = ImagesRecipesDataset(os.path.join(IMAGES_PATH, "val"), os.path.join(RECIPES_PATH, 'val.json'),
                                   label_encoder=encoder)

test_dataset = ImagesRecipesDataset(os.path.join(IMAGES_PATH, "test"), os.path.join(RECIPES_PATH, 'test.json'),
                                    label_encoder=encoder)


print("Train: ", pd.DataFrame(train_dataset.recipes)['cuisine'].value_counts(), "\n\n")
print("Val: ", pd.DataFrame(val_dataset.recipes)['cuisine'].value_counts(), "\n\n")
print("Test: ", pd.DataFrame(test_dataset.recipes)['cuisine'].value_counts(), "\n\n")

print("Train len ", len(train_dataset))
print("Val len ", len(val_dataset))
print("Test len ", len(test_dataset))
