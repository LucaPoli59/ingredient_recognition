from typing import List, Dict, Any
import json
import numpy as np

from settings.config import DEF_UNKNOWN_TOKEN


class MultiLabelBinarizerRobust:
    def __init__(self, classes: List[str] = None, unknown_token: str = DEF_UNKNOWN_TOKEN,
                 encode_map: Dict[str, int] = None, fitted: bool = False):
        self.unknown_token = unknown_token
        self.classes: List[str] = classes
        self.encode_map: Dict[str, int] | None = encode_map
        self.fitted = fitted

        self._inverted_encode_map: Dict[int, str] | None = None

    def fit(self, labels_array: np.ndarray[str] | None = None):
        """
        Fit the encoder either to the labels array or to the classes list (giving priority to the labels array)
        """
        if self.classes is None and labels_array is None:
            raise ValueError("Either classes or labels must be provided")
        if labels_array is not None:
            self.classes = np.unique(np.concatenate(labels_array))

        self.encode_map = {label: index for index, label in enumerate(self.classes)}
        # noinspection PyTypeChecker
        self.encode_map[self.unknown_token] = len(self.classes)

        self._inverted_encode_map = None
        self.fitted = True

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

    def transform(self, labels_array: List[List[str]]):
        if self.encode_map is None:
            raise ValueError("fit() must be called first")
        return np.array(list(map(self._encode, labels_array)))

    def fit_transform(self, labels_array: List[List[str]]):
        self.fit(labels_array)
        return self.transform(labels_array)

    def inverse_transform(self, encoded_labels_array: List[List[int]] | np.ndarray | List[np.ndarray]):  #todo: debugga che non funziona
        if not self.fitted:
            raise ValueError("fit() must be called first")
        return np.array(list(map(self._inverse, encoded_labels_array)), dtype=object)

    def decode_labels(self, encoded_labels_array: List[int] | List[List[int]] | np.ndarray):
        if not self.fitted:
            raise ValueError("fit() must be called first")
        if isinstance(encoded_labels_array[0], int):
            encoded_labels_array = [encoded_labels_array]
        return np.array(list(map(self._decode_labels, encoded_labels_array)), dtype=object)

    @classmethod
    def load_from_config(cls, config: str | Dict[str, Any]):
        if isinstance(config, str):
            config = json.loads(config)

        classes = config['classes'] if config['classes'] is not None else None
        encode_map = config['encode_map']
        fitted = config['fitted']
        unknown_token = config['unknown_token']
        return cls(classes=classes, encode_map=encode_map, fitted=fitted, unknown_token=unknown_token)

    def to_config(self):
        if self.classes is not None:
            classes = self.classes.tolist() if isinstance(self.classes, np.ndarray) else list(self.classes)
        else:
            classes = None
        return {
            'classes': classes,
            'encode_map': self.encode_map,
            'fitted': self.fitted,
            'unknown_token': self.unknown_token,
            'type': self.__class__
        }

    def get_classes(self):
        return self.classes

    def is_fitted(self):
        return self.fitted

    @property
    def inverted_encode_map(self):
        if self._inverted_encode_map is None:
            self._inverted_encode_map = {index: label for label, index in self.encode_map.items()}
        return self._inverted_encode_map
