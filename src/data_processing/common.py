import os
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

import lightning as lgn
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2

from src.data_processing.transformations import transform_aug_base, transform_plain_base, transformations_wrapper, t_transform


class BaseDataModule(ABC, lgn.LightningDataModule):

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

    def prepare_data(self) -> None:  # Prepare data, by computing classes weights and initializing transformations
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

    @transform_plain.setter
    def transform_plain(self, value):
        self._transform_plain = value

    @transform_aug.setter
    def transform_aug(self, value):
        self._transform_aug = value
