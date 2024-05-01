import torch
from torch import nn
import logging
from typing import Any, Tuple
import lightning as lgn
from lightning.pytorch.utilities.types import OptimizerLRScheduler


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


class DummyModel(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape)
        elif len(input_shape) > 2 or len(input_shape) < 1:
            raise ValueError("The input_shape must be a tuple of 2 integers")
        elif input_shape[0] != input_shape[1]:
            raise ValueError("The input_shape must be a tuple of 2 equal integers")

        super().__init__()
        self.block_1 = DummyBlock(3, 16)
        self.block_2 = DummyBlock(16, 32)
        self.block_3 = DummyBlock(32, 64)

        self.num_blocks = 3
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((input_shape[0] // (2 ** self.num_blocks)) ** 2 * 64, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.block_3(self.block_2(self.block_1(x))))
