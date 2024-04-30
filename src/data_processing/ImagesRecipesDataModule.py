import lightning as lgn
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader
import os
from sklearn.base import TransformerMixin as anySkTransformer
from typing import Tuple

from src.data_processing.ImagesRecipesDataset import _ImagesRecipesDataset, images_recipes_processing
from src.data_processing.MultiLabelBinarizerRobust import MultiLabelBinarizerRobust
from settings.config import FOOD_CATEGORIES, IMAGES_PATH, RECIPES_PATH


class ImagesRecipesDataModule(lgn.LightningDataModule):
    def __init__(self,
                 global_images_dir: os.path = IMAGES_PATH,
                 recipes_dir: os.path = RECIPES_PATH,
                 category: str = None,
                 recipe_feature_label: str = "ingredients_ok",
                 label_encoder: None | MultiLabelBinarizerRobust | anySkTransformer = None,
                 image_size: Tuple[int, int] = (224, 224),
                 batch_size: int = 32,
                 num_workers: int | None = None):
        super().__init__()  # Setting parameters
        self.images_dir = global_images_dir
        self.recipes_dir = recipes_dir
        self.recipe_feature_label = recipe_feature_label
        self.image_size = image_size
        self.batch_size = batch_size
        self.label_encoder, self.category, self.num_workers = None, None, None
        self._set_def_init_params(label_encoder, category, num_workers)

        self.transform_aug = self._get_transform_aug()
        self.transform_plain = self._get_transform_plain()

        self._images_dir, self._recipes_files = {}, {}  # Local paths for each stage
        self._check_paths_and_set_locals()

        # Images paths and the label data for each stage (final data used by the datasets)
        self._images_paths, self._label_data = {}, {}

        self.train_dataset, self.val_dataset, self.test_dataset, self.predict_dataset = None, None, None, None

    def _set_def_init_params(self, label_encoder, category, num_workers):
        """Sets some default parameters if not provided"""
        if label_encoder is None:
            label_encoder = MultiLabelBinarizerRobust()  # default encoder
        self.label_encoder = label_encoder

        if category is not None:
            category = category.lower()
            if category not in FOOD_CATEGORIES:
                raise ValueError(f'Invalid category: {category}')
        self.category = category

        if num_workers is None:
            num_workers = os.cpu_count()
        self.num_workers = num_workers

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

    def prepare_data(self):
        """Prepares the data for the datasets by processing the images and recipes data."""
        for stage in ['train', 'val', 'test', 'predict']:
            res = images_recipes_processing(self._images_dir[stage], self._recipes_files[stage], self.category,
                                            self.label_encoder, self.recipe_feature_label)
            self._images_paths[stage], self._label_data[stage], self.label_encoder = res

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
