import json
import pathlib

import lightning as lgn
import pandas as pd
from PIL import Image
from numpy import ndarray
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.base import TransformerMixin as anySkTransformer
from typing import Tuple, List, Dict

from settings.config import FOOD_CATEGORIES, IMAGES_PATH, RECIPES_PATH, DEF_BATCH_SIZE
from settings.commons import tokenize_category
from src.commons.utils import register_hparams

from src.data_processing.labels_encoders import MultiLabelBinarizerRobust


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


class ImagesRecipesDataModule(lgn.LightningDataModule):
    def __init__(
            self,
            global_images_dir: os.path = IMAGES_PATH,
            recipes_dir: os.path = RECIPES_PATH,
            food_categories: List[str] = FOOD_CATEGORIES,
            category: str = None,
            recipe_feature_label: str = "ingredients_ok",
            label_encoder: None | MultiLabelBinarizerRobust | anySkTransformer = None,
            image_size: Tuple[int, int] = (224, 224),
            batch_size: int = DEF_BATCH_SIZE,
            num_workers: int | None = None
    ):
        super().__init__()  # Setting parameters
        self.images_dir, self.recipes_dir, = global_images_dir, recipes_dir,
        self.recipe_feature_label, self.food_categories = recipe_feature_label, food_categories
        self.image_size, self.batch_size = image_size, batch_size
        self.label_encoder, self.category, self.num_workers = label_encoder, category, num_workers
        self._set_def_params()

        register_hparams(self, ["global_images_dir", "recipes_dir", "category", "recipe_feature_label",
                                {"label_encoder": self.label_encoder.to_config()}, {"type": self.__class__},
                                {"num_workers": self.num_workers}], log=False)

        self.transform_aug = self._get_transform_aug()
        self.transform_plain = self._get_transform_plain()

        self._images_dir, self._recipes_files = {}, {}  # Local paths for each stage
        self._check_paths_and_set_locals()

        # Images paths and the label data for each stage (final data used by the datasets)
        self._images_paths, self._label_data = {}, {}

        self.train_dataset, self.val_dataset, self.test_dataset, self.predict_dataset = None, None, None, None

    def _set_def_params(self):
        """Sets some default parameters if not provided"""
        if self.label_encoder is None:
            self.label_encoder = MultiLabelBinarizerRobust()  # default encoder

        if self.category is not None:
            self.category = self.category.lower()
            if self.category not in self.food_categories:
                raise ValueError(f'Invalid category: {self.category}')

        if self.num_workers is None:
            self.num_workers = os.cpu_count()

    def _check_paths_and_set_locals(self):
        """Checks if the global images and recipes directories exist and sets the local paths for each stage"""
        # Checks if the global images and recipes directories exist
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f'Images directory not found: {self.images_dir}')
        if not os.path.exists(self.recipes_dir):
            raise FileNotFoundError(f'Recipes directory not found: {self.recipes_dir}')

        # Checks if the images and recipes directories for each stage exist and sets the local paths
        for stage in ["train", "val", "test"]:
            imgs_path = os.path.join(self.images_dir, stage)
            recipes_file = os.path.join(self.recipes_dir, f"{stage}.json")
            if not os.path.exists(imgs_path):
                raise FileNotFoundError(f'Images directory for {stage} stage not found: {imgs_path}')
            if not os.path.exists(recipes_file):
                raise FileNotFoundError(f'Recipes file for {stage} stage not found: {recipes_file}')

            self._images_dir[stage] = imgs_path
            self._recipes_files[stage] = recipes_file

        self._set_local_path_predict()

    def _set_local_path_predict(self):
        """Sets the local paths for the predict stage"""
        predict_imgs_path = os.path.join(self.images_dir, "predict")
        predict_recipes_file = os.path.join(self.recipes_dir, "predict.json")
        if not os.path.exists(predict_imgs_path) and not os.path.exists(predict_recipes_file):
            # If the predict directories do not exist, set the predict dataset equal to the test dataset
            self._images_dir["predict"] = self._images_dir["test"]
            self._recipes_files["predict"] = self._recipes_files["test"]

        else:  # If one of the paths exists, both of them must exist
            if not os.path.exists(predict_imgs_path):
                raise FileNotFoundError(f'Images directory for predict stage not found: {predict_imgs_path}')
            if not os.path.exists(predict_recipes_file):
                raise FileNotFoundError(f'Recipes file for predict stage not found: {predict_recipes_file}')
            self._images_dir["predict"] = predict_imgs_path
            self._recipes_files["predict"] = predict_recipes_file

    def _get_transform_aug(self, num_magnitude_bins=31):
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(self.image_size),
            v2.TrivialAugmentWide(num_magnitude_bins=num_magnitude_bins),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def _get_transform_plain(self):
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(self.image_size),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def prepare_data(self): # todo: fare il sistema che salva i risultati in un file, in modo che non vengano ricalcolati ogni volta (e che si possano rimuovere volendo dal checkpointing)
        """Prepares the data for the datasets by processing the images and recipes data."""
        for stage in ['train', 'val', 'test', 'predict']:
            res = images_recipes_processing(self._images_dir[stage], self._recipes_files[stage], self.category,
                                            self.label_encoder, self.recipe_feature_label)
            self._images_paths[stage], self._label_data[stage], self.label_encoder = res

        self.hparams['label_encoder'] = self.label_encoder.to_config()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = _ImagesRecipesDataset(self._images_paths['train'], self._label_data['train'],
                                                       self.transform_aug)
            self.val_dataset = _ImagesRecipesDataset(self._images_paths['val'], self._label_data['val'],
                                                     self.transform_plain)
        if stage == 'test' or stage is None:
            self.test_dataset = _ImagesRecipesDataset(self._images_paths['test'], self._label_data['test'],
                                                      self.transform_plain)
        if stage == 'predict' or stage is None:
            self.predict_dataset = _ImagesRecipesDataset(self._images_paths['predict'], self._label_data['predict'],
                                                         self.transform_plain)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=True)

    def get_num_classes(self):
        return len(self.label_encoder.classes) + 1

    @classmethod
    def load_from_config(cls, config: Dict[str, any], image_size: Tuple[int, int], batch_size: int, **kwargs
                         ) -> 'ImagesRecipesDataModule':
        image_dir_path, recipe_dir_path = config['global_images_dir'], config['recipes_dir']
        category, recipe_feature_label = config['category'], config['recipe_feature_label']
        num_workers = config['num_workers']
        # le_type = str_to_class(json.loads(config['label_encoder'])['type'])
        le_type = config['label_encoder']['type']

        label_encoder = le_type.load_from_config(config['label_encoder'])
        return cls(global_images_dir=image_dir_path, recipes_dir=recipe_dir_path, category=category,
                   image_size=image_size, batch_size=batch_size, recipe_feature_label=recipe_feature_label,
                   num_workers=num_workers, label_encoder=label_encoder, **kwargs)


def images_recipes_processing(
        images_dir: os.path, recipes_file: os.path, category: str | None = None,
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


def _encode_recipes(
        recipes: List[Dict],
        label_encoder: anySkTransformer | MultiLabelBinarizerRobust,
        feature_label: str) -> Tuple[ndarray, anySkTransformer | MultiLabelBinarizerRobust]:
    # Fit the encoder to the label feature if it is not already fitted, and then transform it
    label_data_raw = pd.DataFrame(recipes)[feature_label].values
    if not label_encoder.fitted:  # warning: this doesn't work for anySkTransformer
        label_encoder.fit(label_data_raw)
    return label_encoder.transform(label_data_raw), label_encoder


def _compute_images_paths(images_dir: os.path, category: str) -> List[pathlib.Path]:
    if category is None or category == "all":
        return list(pathlib.Path(images_dir).glob('*.jpg'))
    else:
        return list(pathlib.Path(images_dir).glob(f'*{tokenize_category(category)}.jpg'))


