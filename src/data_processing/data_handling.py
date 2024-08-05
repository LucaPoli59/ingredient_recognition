import json
import pathlib

import lightning as lgn
import numpy as np
import pandas as pd
from PIL import Image
from numpy import ndarray
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader, Dataset
import os
from typing import Tuple, List, Dict, Optional, Any, Callable, Sequence
from typing_extensions import Self
from abc import ABC, abstractmethod

from settings.config import FOOD_CATEGORIES, YUMMLY_PATH, YUMMLY_RECIPES_PATH, DEF_BATCH_SIZE, YUMMLY_IMG_STATS_PATH, \
    METADATA_FILENAME
from settings.commons import tokenize_category
from src.commons.utils import register_hparams

from src.data_processing.labels_encoders import MultiLabelBinarizerRobust, LabelEncoderInterface
from src.data_processing.transformations import transform_plain_base, transform_aug_base, transformations_wrapper


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

    def to_light_dataset(self, label_encoder: Optional[LabelEncoderInterface] = None
                         ) -> 'LightImagesRecipesDataset':
        label_data = self.label_data
        if label_encoder is not None and label_encoder.fitted:
            label_data = label_encoder.inverse_transform(label_data)
            label_data = [label.tolist() for label in label_data if isinstance(label, ndarray)]
        return LightImagesRecipesDataset(self.images_paths, label_data)


class LightImagesRecipesDataset(_ImagesRecipesDataset):
    def __init__(self, images_paths: List[pathlib.Path], label_data: ndarray):
        super().__init__(images_paths, label_data, None)

    def to_json(self) -> Dict[str, Any]:
        return {"images_paths": [str(p) for p in self.images_paths], "label_data": self.label_data}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'LightImagesRecipesDataset':
        return cls([pathlib.Path(p) for p in data['images_paths']], data['label_data'])

    def to_light_dataset(self, label_encoder: Optional[LabelEncoderInterface] = None
                         ) -> 'LightImagesRecipesDataset':
        return self


class ImagesRecipesDataset(_ImagesRecipesDataset):
    """Dataset class wrapper for the base class _ImagesRecipesDataset.
    It loads the images from a general directory and the recipes from a json file, filters them by category,
    encodes the recipes to pass everything to the base class."""

    def __init__(self, data_dir, transform=None, category=None, label_encoder=None,
                 metadata_filename="metadata.json", feature_label="ingredients_ok"):

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
        images_paths, label_data, label_encoder = images_recipes_processing(data_dir, metadata_filename, category,
                                                                            label_encoder, feature_label)

        super().__init__(images_paths, label_data, transform)


class BaseDataModule(ABC, lgn.LightningDataModule):
    t_transform = Callable[[Image.Image | np.ndarray | torch.Tensor], torch.Tensor]

    def __init__(self, images_stats_path: str | os.PathLike, transform_aug: Optional[t_transform] = None,
                 transform_plain: Optional[t_transform] = None):
        super().__init__()
        self.classes_weights = None
        self.images_stats_path = images_stats_path
        self._transform_aug = self._def_transform_aug() if transform_aug is None else transform_aug
        self._transform_plain = self._def_transform_plain() if transform_plain is None else transform_plain

        self.prepared = False

    @staticmethod
    def _def_transform_aug(num_magnitude_bins=31):
        return transform_aug_base(num_magnitude_bins=num_magnitude_bins)

    @staticmethod
    def _def_transform_plain():
        return transform_plain_base()

    @abstractmethod
    def get_num_classes(self):
        pass

    @abstractmethod
    def _compute_classes_weights(self) -> torch.tensor:
        pass

    @staticmethod
    def _init_transform(transform: list[v2.Transform] | v2.Transform, mean: Sequence[float], std: Sequence[float]
                        ) -> v2.Transform:
        if type(transform) is list:
            return transformations_wrapper(transform, mean, std)
        if isinstance(transform, v2.Transform):
            return transform
        raise ValueError("Invalid transform type")

    def prepare_data(self) -> None:
        self.classes_weights = self._compute_classes_weights()

        if not os.path.exists(self.images_stats_path):
            raise FileNotFoundError(f'Images stats file not found: {self.images_stats_path}')
        mean, std = pd.read_csv(self.images_stats_path, index_col=0).values  # TODO: quando ci sarà il sistema che salva i risultati in un file, anche questi dati verranno calcolati e salvati in quel file, e poi caricati

        self._transform_aug = self._init_transform(self._transform_aug, mean, std)
        self._transform_plain = self._init_transform(self._transform_plain, mean, std)

        self.prepared = True

    @property
    def transform_aug(self):
        if not self.prepared:
            raise ValueError("prepare_data() must be called first")
        return self._transform_aug

    @property
    def transform_plain(self):
        if not self.prepared:
            raise ValueError("prepare_data() must be called first")
        return self._transform_plain


class ImagesRecipesBaseDataModule(BaseDataModule):
    def __init__(
            self,
            data_dir: os.path = YUMMLY_PATH,
            metadata_filename: str = METADATA_FILENAME,
            images_stats_path: str | os.PathLike = YUMMLY_IMG_STATS_PATH,
            food_categories: List[str] = FOOD_CATEGORIES,
            category: str = None,
            feature_label: str = "ingredients_ok",
            label_encoder: None | LabelEncoderInterface = None,
            batch_size: int = DEF_BATCH_SIZE,
            num_workers: int | None = None,
            transform_aug: Optional[BaseDataModule.t_transform] = None,
            transform_plain: Optional[BaseDataModule.t_transform] = None,
    ):
        super().__init__(images_stats_path, transform_aug=transform_aug, transform_plain=transform_plain)  # Setting parameters
        self.data_dir, self.metadata_filename, = data_dir, metadata_filename,
        self.recipe_feature_label, self.food_categories = feature_label, food_categories
        self.batch_size, self.num_workers = batch_size, num_workers
        self.label_encoder, self.category = label_encoder, category
        self._set_def_params()

        register_hparams(self, ["data_dir", "metadata_filename", "category", "feature_label",
                                {"label_encoder": self.label_encoder.to_config()}, {"type": self.__class__},
                                {"num_workers": self.num_workers}, {}],
                         log=False)

        self._stage_data_dir = {}  # Local paths for each stage
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
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f'Dataset directory not found: {self.data_dir}')

        # Checks if the images and recipes directories for each stage exist and sets the local paths
        for stage in ["train", "val", "test"]:
            stage_data_dir = os.path.join(self.data_dir, stage)
            recipes_file = os.path.join(stage_data_dir, self.metadata_filename)
            if not os.path.exists(recipes_file):
                raise FileNotFoundError(f'Recipes file for {stage} stage not found: {recipes_file}')

            self._stage_data_dir[stage] = stage_data_dir

        self._set_local_path_predict()

    def _set_local_path_predict(self):
        """Sets the local paths for the predict stage"""
        predict_data_dir = os.path.join(self.data_dir, "predict")
        predict_recipes_file = os.path.join(predict_data_dir, self.metadata_filename)
        if not os.path.exists(predict_data_dir) and not os.path.exists(predict_recipes_file):
            # If the predict directories do not exist, set the predict dataset equal to the test dataset
            self._stage_data_dir["predict"] = self._stage_data_dir["test"]

        else:  # If one of the paths exists, both of them must exist
            if not os.path.exists(predict_data_dir):
                raise FileNotFoundError(f'Dataset directory for predict stage not found: {predict_data_dir}')
            if not os.path.exists(predict_recipes_file):
                raise FileNotFoundError(f'Recipes file for predict stage not found: {predict_recipes_file}')
            self._stage_data_dir["predict"] = predict_data_dir

    def _compute_classes_weights(self, stage_target="train", minority_inversion=True, standardize=True
                                 ) -> torch.tensor:
        """Computes the class weights for the dataset labels"""

        classes_occ = np.sum(self._label_data[stage_target], axis=0, dtype=np.float32)
        classes_occ[classes_occ == 0] = np.NaN  # Put NaNs in the classes that are not present in the dataset

        class_weights = np.nansum(classes_occ) / classes_occ

        if not minority_inversion:
            class_weights = 1 / class_weights

        if standardize:
            class_weights = class_weights / class_weights[~np.isnan(class_weights)].min()

        return torch.tensor(np.nan_to_num(class_weights), dtype=torch.float32)

    def prepare_data(
            self):  # todo: fare il sistema che salva i risultati in un file, in modo che non vengano ricalcolati ogni volta (e che si possano rimuovere volendo dal checkpointing)
        """Prepares the data for the datasets by processing the images and recipes data."""
        for stage in ['train', 'val', 'test', 'predict']:
            res = images_recipes_processing(self._stage_data_dir[stage], self.metadata_filename, self.category,
                                            self.label_encoder, self.recipe_feature_label)
            self._images_paths[stage], self._label_data[stage], self.label_encoder = res

        self.hparams['label_encoder'] = self.label_encoder.to_config()

        super().prepare_data()

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
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=True)

    def get_num_classes(self):
        return self.label_encoder.num_classes

    @classmethod
    def load_from_config(cls, config: Dict[str, any], batch_size: int,
                         transform_aug: Optional[BaseDataModule.t_transform] = None,
                         transform_plain: Optional[BaseDataModule.t_transform] = None,
                         **kwargs
                         ) -> Self:
        data_dir_path, metadata_filename = config['data_dir'], config['metadata_filename']
        category, feature_label = config['category'], config['feature_label']
        num_workers, le_type = config['num_workers'], config['label_encoder']['type']

        label_encoder = le_type.load_from_config(config['label_encoder'])
        return cls(data_dir=data_dir_path, metadata_filename=metadata_filename, category=category,
                   batch_size=batch_size, feature_label=feature_label,
                   num_workers=num_workers, label_encoder=label_encoder,
                   transform_plain=transform_plain, transform_aug=transform_aug, **kwargs)


def images_recipes_processing(
        data_dir: os.path, metadata_filename: str = METADATA_FILENAME, category: str | None = None,
        label_encoder: LabelEncoderInterface = None, recipe_feature_label: str = "ingredients_ok",
        image_field: str = "image"
) -> Tuple[List[pathlib.Path], ndarray, LabelEncoderInterface]:
    """Function that processes the images and recipes data, filtering them by category, encoding the recipes and
    returning the images paths, the label data and the label encoder."""

    recipes = json.load(open(os.path.join(data_dir, metadata_filename)))
    recipes = _recipes_filter_by_category(recipes, category)

    images_paths = _compute_images_paths(recipes, data_dir, image_field)
    label_data, label_encoder = _encode_recipes(recipes, label_encoder, recipe_feature_label)

    return images_paths, label_data, label_encoder


def _recipes_filter_by_category(recipes: List[Dict], category: str | None = None) -> List[Dict]:
    if category is None or category == "all":
        return recipes
    return list(filter(lambda recipe: recipe['cuisine'].lower() == category, recipes))


def _encode_recipes(
        recipes: List[Dict],
        label_encoder: LabelEncoderInterface,
        feature_label: str) -> Tuple[ndarray, LabelEncoderInterface]:
    # Fit the encoder to the label feature if it is not already fitted, and then transform it
    label_data_raw = pd.DataFrame(recipes)[feature_label].values
    if not label_encoder.fitted:  # warning: this doesn't work for anySkTransformer
        label_encoder.fit(label_data_raw)
    return label_encoder.transform(label_data_raw), label_encoder


def _compute_images_paths(metadata: List[Dict], data_dir: str | os.PathLike, image_field: str = "image"
                          ) -> List[pathlib.Path]:
    metadata_df = pd.DataFrame(metadata)
    metadata_df[image_field] = metadata_df[image_field].apply(lambda img_path: os.path.join(data_dir, img_path))
    return metadata_df[image_field].values.tolist()
