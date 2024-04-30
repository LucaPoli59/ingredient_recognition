from typing import List, Dict

import numpy as np


class MultiLabelBinarizerRobust:
    def __init__(self, classes: List[str] = None, unknown_token: str = '<UNK>'):
        self.unknown_token = unknown_token
        self.classes: List[str] = classes
        self.encode_map = None
        self.fitted = False

    def fit(self, labels_array=None):
        """
        Fit the encoder either to the labels array or to the classes list (giving priority to the labels array)
        """
        if self.classes is None and labels_array is None:
            raise ValueError("Either classes or labels must be provided")
        if labels_array is not None:
            # noinspection PyTypeChecker
            self.classes: List[str] = np.unique(np.concatenate(labels_array))

        self.encode_map: Dict[str, int] = {label: index for index, label in enumerate(self.classes)}
        self.encode_map[self.unknown_token] = len(self.classes)
        self.fitted = True

    def _get_index(self, label: str):
        return self.encode_map.get(label, len(self.encode_map) - 1)

    def _encode(self, labels: List[str]):
        encoded_labels = np.zeros(len(self.encode_map))
        indices = [self._get_index(label) for label in labels]
        encoded_labels[indices] = 1
        return encoded_labels

    def _decode(self, encoded_labels: List[int]):
        return [label for label, index in self.encode_map.items() if encoded_labels[index] == 1]

    def transform(self, labels_array: List[List[str]]):
        if self.encode_map is None:
            raise ValueError("fit() must be called first")
        return np.array(list(map(self._encode, labels_array)))

    def fit_transform(self, labels_array: List[List[str]]):
        self.fit(labels_array)
        return self.transform(labels_array)

    def inverse_transform(self, encoded_labels_array: List[List[int]] | np.ndarray | List[np.ndarray]):
        if not self.fitted:
            raise ValueError("fit() must be called first")
        return np.array(list(map(self._decode, encoded_labels_array)))

    def get_classes(self):
        return self.classes

    def is_fitted(self):
        return self.fitted
