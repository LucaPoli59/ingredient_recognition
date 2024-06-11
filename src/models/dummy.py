import torch
from torch import nn
from typing import Dict, Any

from typing_extensions import Self

from src.models.commons import BaseModel


class _BaseDummy(BaseModel):
    def __init__(self, input_shape, num_classes):
        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape)
        elif len(input_shape) > 2 or len(input_shape) < 1:
            raise ValueError("The input_shape must be a tuple of 2 integers")
        elif input_shape[0] != input_shape[1]:
            raise ValueError("The input_shape must be a tuple of 2 equal integers")

        super().__init__(input_shape=input_shape, num_classes=num_classes)
        self.num_classes = num_classes
        self.input_shape = input_shape
    
    @classmethod
    def load_from_config(cls, config: Dict[str, Any], **kwargs) -> Self:
        return cls(config["input_shape"], config["num_classes"])

class DummyBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 3, padding="same"), nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, 3, padding="same"), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_target_layer = self.block[4]

    def forward(self, x):
        return self.block(x)


class DummyModel(_BaseDummy):
    def __init__(self, input_shape, num_classes):
        super().__init__(input_shape, num_classes)
        self.block_1 = DummyBlock(3, 16)
        self.block_2 = DummyBlock(16, 32)
        self.block_3 = DummyBlock(32, 64)

        self.num_blocks = 3

        last_block = getattr(self, f"block_{self.num_blocks}")
        self.conv_target_layer = last_block.conv_target_layer

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((self.input_shape[0] // (2 ** self.num_blocks)) ** 2 * 64, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.classifier_target_layer = self.classifier[-1]

    def forward(self, x):
        return self.classifier(self.block_3(self.block_2(self.block_1(x))))



class DummyNBBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 3, padding="same"), nn.BatchNorm2d(output_dim), nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, 3, padding="same"), nn.BatchNorm2d(output_dim), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_target_layer = self.block[6]

    def forward(self, x):
        return self.block(x)


class DummyBNModel(_BaseDummy):
    def __init__(self, input_shape, num_classes):
        super().__init__(input_shape, num_classes)
        self.block_1 = DummyNBBlock(3, 16)
        self.block_2 = DummyNBBlock(16, 32)
        self.block_3 = DummyNBBlock(32, 64)

        self.num_blocks = 3

        last_block = getattr(self, f"block_{self.num_blocks}")
        self.conv_target_layer = last_block.conv_target_layer

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((self.input_shape[0] // (2 ** self.num_blocks)) ** 2 * 64, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.classifier_target_layer = self.classifier[-1]

    def forward(self, x):
        return self.classifier(self.block_3(self.block_2(self.block_1(x))))
    

if __name__ == "__main__":
    model = DummyModel(32, 10)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    print(out.shape)
    print(model)
