from typing import List, Dict, Any
import json
import numpy as np
from typing_extensions import Self
from abc import ABC, abstractmethod

from settings.config import DEF_UNKNOWN_TOKEN


class LabelEncoderInterface(ABC):
    def __init__(self, classes: List[str] = None, fitted: bool = False):
        self.classes: List[str] = classes
        self.fitted = fitted

    def fit(self, labels_array: np.ndarray[str] | None = None):
        """
        Fit the encoder either to the labels array or to the classes list (giving priority to the labels array)
        """
        if self.classes is None and labels_array is None:
            raise ValueError("Either classes or labels must be provided")
        if labels_array is not None:
            self.classes = np.unique(np.concatenate(labels_array))

        self._fit()
        self.fitted = True

    def transform(self, labels_array: List[List[str]]):
        if not self.fitted:
            raise ValueError("fit() must be called first")
        return np.array(list(map(self._encode, labels_array)))

    def fit_transform(self, labels_array: List[List[str]]):
        self.fit(labels_array)
        return self.transform(labels_array)

    def inverse_transform(self, encoded_labels_array: List[List[int]] | np.ndarray | List[
        np.ndarray]):  # todo: debugga che non funziona
        if not self.fitted:
            raise ValueError("fit() must be called first")
        return np.array(list(map(self._inverse, encoded_labels_array)), dtype=object)

    def decode_labels(self, encoded_labels_array: List[int] | List[List[int]] | np.ndarray):
        if not self.fitted:
            raise ValueError("fit() must be called first")
        if isinstance(encoded_labels_array[0], int):
            encoded_labels_array = [encoded_labels_array]
        return np.array(list(map(self._decode_labels, encoded_labels_array)), dtype=object)

    def get_classes(self):
        return self.classes

    def is_fitted(self):
        return self.fitted

    @classmethod
    def load_from_config(cls, config: str | Dict[str, Any]) -> Self:
        if isinstance(config, str):
            config = json.loads(config)

        params = cls._load_config(config)
        return cls(**params)

    def to_config(self):
        if self.classes is not None:
            classes = self.classes.tolist() if isinstance(self.classes, np.ndarray) else list(self.classes)
        else:
            classes = None
        return {
            'type': self.__class__,
            'classes': classes,
            'fitted': self.fitted,
        }

    @classmethod
    def _load_config(cls, config: Dict[str, Any]):
        if cls is not config['type']:
            raise ValueError(f"Config type {config['type']} does not match class type {cls}")
        params = {key: value for key, value in config.items() if key in ['classes', 'fitted']}
        return params

    @abstractmethod
    def _fit(self):
        pass

    @abstractmethod
    def _encode(self, labels: List[str]):
        pass

    @abstractmethod
    def _inverse(self, encoded_labels: List[int]):
        pass

    @abstractmethod
    def _decode_labels(self, encoded_labels: List[int]):
        pass


class MultiLabelBinarizerRobust(LabelEncoderInterface):
    def __init__(self, classes: List[str] = None, unknown_token: str = DEF_UNKNOWN_TOKEN,
                 encode_map: Dict[str, int] = None, fitted: bool = False):
        super().__init__(classes, fitted)
        self.encode_map: Dict[str, int] | None = encode_map
        self._inverted_encode_map: Dict[int, str] | None = None
        self.unknown_token = unknown_token

    @property
    def inverted_encode_map(self):
        if self._inverted_encode_map is None:
            self._inverted_encode_map = {index: label for label, index in self.encode_map.items()}
        return self._inverted_encode_map

    @classmethod
    def _load_config(cls, config: Dict[str, Any]):
        params = super()._load_config(config)
        params['encode_map'] = config['encode_map']
        params['unknown_token'] = config['unknown_token']
        return params

    def to_config(self):
        config = super().to_config()
        config['encode_map'] = self.encode_map
        config['unknown_token'] = self.unknown_token
        return config

    def _fit(self):
        self.encode_map = {label: index for index, label in enumerate(self.classes)}
        self.encode_map[self.unknown_token] = len(self.classes)
        self._inverted_encode_map = None

    def get_index(self, label: str):
        return self.encode_map.get(label, len(self.encode_map) - 1)

    def _encode(self, labels: List[str]):
        encoded_labels = np.zeros(len(self.encode_map))
        indices = [self.get_index(label) for label in labels]
        encoded_labels[indices] = 1
        return encoded_labels

    def _inverse(self, encoded_labels: List[int]):
        return np.array([label for label, index in self.encode_map.items() if encoded_labels[index] == 1])

    def _decode_labels(self, encoded_labels: List[int]):
        return np.array([self.inverted_encode_map[value] for value in encoded_labels])
