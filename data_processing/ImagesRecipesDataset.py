import os
import pathlib
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from typing import Tuple
from PIL import Image

from config import *
from commons import *
from data_processing.MultiLabelBinarizerRobust import MultiLabelBinarizerRobust


class ImagesRecipesDataset(Dataset):
    def __init__(self, image_dir, recipes_file, transform=None, category=None, label_encoder=None,
                 recipe_feature_label="ingredients_ok"):
        self.image_dir = image_dir
        self.recipes_file = recipes_file

        if transform is None:
            transform = v2.Compose([v2.ToImage(), v2.Resize((224, 224)), v2.ToDtype(torch.float32, scale=True)])
        self.transform = transform

        if label_encoder is None:
            label_encoder = MultiLabelBinarizerRobust()  # default encoder
        self.label_encoder = label_encoder

        if category is not None:
            category = category.lower()
            if category not in FOOD_CATEGORIES:
                raise ValueError(f'Invalid category: {category}')
        self.category = category

        self.feature_label = recipe_feature_label
        self._compute_images_paths()
        self._compute_recipes()

    def _compute_images_paths(self):
        if self.category is None:
            self.images_paths = list(pathlib.Path(self.image_dir).glob('*.jpg'))
        else:
            self.images_paths = list(pathlib.Path(self.image_dir).glob(f'*{tokenize_category(self.category)}.jpg'))

    def _compute_recipes(self):
        if self.category is None:
            self.recipes = json.load(open(self.recipes_file))
        else:
            recipes = json.load(open(self.recipes_file))
            # Filter recipes by category chosen
            self.recipes = list(filter(lambda recipe: recipe['cuisine'].lower() == self.category, recipes))
            if not self.recipes:
                raise ValueError(f"No recipes found for category {self.category}")

        self._encode_label_data()

    def _encode_label_data(self):
        # Fit the encoder to the label feature if it is not already fitted, and then transform it
        label_data_raw = pd.DataFrame(self.recipes)[self.feature_label].values
        if not self.label_encoder.is_fitted():
            self.label_encoder.fit(label_data_raw)
        self.label_data = self.label_encoder.transform(label_data_raw)

    def load_image(self, idx) -> Image.Image:
        """Opens an image via its path and returns it as a PIL image."""
        image_path = self.images_paths[idx]
        return Image.open(image_path)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, torch.tensor]:
        image = self.load_image(idx)
        label = self.label_data[idx]
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)
