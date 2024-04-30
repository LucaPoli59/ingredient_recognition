import pathlib
import json

import pandas as pd
from numpy import ndarray
from torch.utils.data import Dataset
from torchvision.transforms import v2
from typing import Tuple, List, Dict, Iterable
from PIL import Image
from sklearn.base import TransformerMixin as anySkTransformer

from settings.config import *
from settings.commons import *
from src.data_processing.MultiLabelBinarizerRobust import MultiLabelBinarizerRobust


class _ImagesRecipesDataset(Dataset):
    """Base Dataset class for the images and labels. It loads the images and recipes data and applies the
    transformations to the images."""

    def __init__(self, images_paths, label_data, transform):
        self.images_paths = images_paths
        self.label_data = label_data
        self.transform = transform
        super().__init__()

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


class ImagesRecipesDataset(_ImagesRecipesDataset):
    """Dataset class wrapper for the base class _ImagesRecipesDataset.
    It loads the images from a general directory and the recipes from a json file, filters them by category,
    encodes the recipes to pass everything to the base class."""

    def __init__(self, image_dir, recipes_file, transform=None, category=None, label_encoder=None,
                 recipe_feature_label="ingredients_ok"):

        # Check validity of parameters
        if transform is None:
            transform = v2.Compose([v2.ToImage(), v2.Resize((224, 224)), v2.ToDtype(torch.float32, scale=True)])
        if label_encoder is None:
            label_encoder = MultiLabelBinarizerRobust()  # default encoder
        if category is not None:
            category = category.lower()
            if category not in FOOD_CATEGORIES:
                raise ValueError(f'Invalid category: {category}')

        # Compute images_path, Load recipes filter them by category and encode them to get the label data
        images_paths, label_data, label_encoder = images_recipes_processing(image_dir, recipes_file, category,
                                                                            label_encoder, recipe_feature_label)

        super().__init__(images_paths, label_data, transform)


def images_recipes_processing(images_dir: os.path, recipes_file: os.path, category: str | None = None,
                              label_encoder: anySkTransformer | MultiLabelBinarizerRobust = None,
                              recipe_feature_label: str = "ingredients_ok"
                              ) -> Tuple[List[pathlib.Path], ndarray, anySkTransformer | MultiLabelBinarizerRobust]:
    """Function that processes the images and recipes data, filtering them by category, encoding the recipes and
    returning the images paths, the label data and the label encoder."""

    images_paths = _compute_images_paths(images_dir, category)
    recipes = json.load(open(recipes_file))
    recipes = _recipes_filter_by_category(recipes, category)
    label_data, label_encoder = _encode_recipes(recipes, label_encoder, recipe_feature_label)
    return images_paths, label_data, label_encoder


def _recipes_filter_by_category(recipes: List[Dict], category: str | None = None) -> List[Dict]:
    if category is None or category == "all":
        return recipes
    return list(filter(lambda recipe: recipe['cuisine'].lower() == category, recipes))


def _encode_recipes(recipes: List[Dict],
                    label_encoder: anySkTransformer | MultiLabelBinarizerRobust,
                    feature_label: str) -> Tuple[ndarray, anySkTransformer | MultiLabelBinarizerRobust]:
    # Fit the encoder to the label feature if it is not already fitted, and then transform it
    label_data_raw = pd.DataFrame(recipes)[feature_label].values
    if not label_encoder.fitted: # warning: this doesn't work for anySkTransformer
        label_encoder.fit(label_data_raw)
    return label_encoder.transform(label_data_raw), label_encoder


def _compute_images_paths(images_dir: os.path, category: str) -> List[pathlib.Path]:
    if category is None:
        return list(pathlib.Path(images_dir).glob('*.jpg'))
    else:
        return list(pathlib.Path(images_dir).glob(f'*{tokenize_category(category)}.jpg'))
