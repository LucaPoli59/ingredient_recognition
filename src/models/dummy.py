import torch
from torch import nn
from typing import Dict, Any
from typing_extensions import Self
from abc import ABC, abstractmethod

from settings.config import DEF_IMAGE_SHAPE
from src.models.commons import BaseModel


class _BaseDummy(BaseModel, ABC):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape)

    @classmethod
    def load_from_config(cls, config: Dict[str, Any], **kwargs) -> Self:
        return cls(config["num_classes"], config["input_shape"])


class DummyBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 3, padding="same"), nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, 3, padding="same"), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.block(x)

    @property
    def conv_target_layer(self):
        return self.block[4]


class DummyModel(_BaseDummy):

    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE):
        super().__init__(num_classes, input_shape)
        self.block_1 = DummyBlock(3, 16)
        self.block_2 = DummyBlock(16, 32)
        self.block_3 = DummyBlock(32, 64)
        self.num_blocks = 3

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((self.input_shape[0] // (2 ** self.num_blocks)) ** 2 * 64, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.block_3(self.block_2(self.block_1(x))))

    @property
    def conv_target_layer(self):
        last_block = getattr(self, f"block_{self.num_blocks}")
        return last_block.conv_target_layer

    @property
    def classifier_target_layer(self):
        return self.classifier[-1]


class DummyNBBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 3, padding="same"), nn.BatchNorm2d(output_dim), nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, 3, padding="same"), nn.BatchNorm2d(output_dim), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.block(x)

    @property
    def conv_target_layer(self):
        return self.block[6]


class DummyBNModel(_BaseDummy):
    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE):
        super().__init__(num_classes, input_shape)
        self.block_1 = DummyNBBlock(3, 16)
        self.block_2 = DummyNBBlock(16, 32)
        self.block_3 = DummyNBBlock(32, 64)

        self.num_blocks = 3

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((self.input_shape[0] // (2 ** self.num_blocks)) ** 2 * 64, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.classifier_target_layer = self.classifier[-1]

    def forward(self, x):
        return self.classifier(self.block_3(self.block_2(self.block_1(x))))

    @property
    def conv_target_layer(self):
        last_block = getattr(self, f"block_{self.num_blocks}")
        return last_block.conv_target_layer

    @property
    def classifier_target_layer(self):
        return self.classifier[-1]


if __name__ == "__main__":
    model = DummyModel(10, 32)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    print(out.shape)
    print(model)
