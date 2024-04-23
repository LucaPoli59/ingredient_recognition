import os
import pathlib
import json
from torch.utils.data import Dataset
from typing import Tuple
from PIL import Image

from config import *
from commons import *


class ImageDataset(Dataset):
    def __init__(self, image_dir, metadata_file, transform=None, category=None):
        self.image_dir = image_dir
        self.metadata_file = metadata_file
        self.transform = transform

        if category is not None:
            category = category.lower()
            if category not in FOOD_CATEGORIES:
                raise ValueError(f'Invalid category: {category}')
        self.category = category

        self._compute_images_paths()
        self._compute_metadata()

    def _compute_images_paths(self):
        if self.category is None:
            self.images_paths = list(pathlib.Path(self.image_dir).glob('*.jpg'))
        else:
            self.images_paths = list(pathlib.Path(self.image_dir).glob(f'*{tokenize_category(self.category)}.jpg'))

    def _compute_metadata(self):
        if self.category is None:
            self.metadata = json.load(open(self.metadata_file))
        else:
            metadata_file = json.load(open(self.metadata_file))
            self.metadata = list(filter(lambda recipe: recipe['cuisine'].lower() == self.category, metadata_file))

    def load_image(self, idx) -> Image.Image:
        """Opens a image via its path and returns it as a PIL image."""
        image_path = self.images_paths[idx]
        return Image.open(image_path)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, list]:
        image = self.load_image(idx)
        metadata = self.metadata[idx]
        if self.transform:
            image = self.transform(image)

        return image, metadata['name']


class ImagesRecipesDataset(Dataset):
    def __init__(self, image_dir, recipes_file, transform=None, category=None):
        self.image_dir = image_dir
        self.recipes_file = recipes_file
        self.transform = transform

        if category is not None:
            category = category.lower()
            if category not in FOOD_CATEGORIES:
                raise ValueError(f'Invalid category: {category}')
        self.category = category

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
            metadata_file = json.load(open(self.recipes_file))
            self.recipes = list(filter(lambda recipe: recipe['cuisine'].lower() == self.category, metadata_file))

    def load_image(self, idx) -> Image.Image:
        """Opens a image via its path and returns it as a PIL image."""
        image_path = self.images_paths[idx]
        return Image.open(image_path)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, list]:
        image = self.load_image(idx)
        recipe = self.recipes[idx]
        if self.transform:
            image = self.transform(image)

        return image, recipe['ingredients_ok']


train_dataset = ImagesRecipesDataset(os.path.join(IMAGES_PATH, "train"), os.path.join(RECIPES_PATH, 'train.json'))

train_dataset_iter = iter(train_dataset)
for _ in range(20):
    image, y = next(train_dataset_iter)
    show_image(image, title=f"{y}")
