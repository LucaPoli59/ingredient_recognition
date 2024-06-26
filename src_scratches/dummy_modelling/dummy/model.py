import torch
from torch import nn
from torchinfo import summary

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
        super().__init__()
        self.block_1 = DummyBlock(3, 16)
        self.block_2 = DummyBlock(16, 32)
        self.block_3 = DummyBlock(32, 64)

        self.num_blocks = 3
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((input_shape // (2 ** self.num_blocks))**2 * 64, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x = self.block_1(x)
        # x = self.block_2(x)
        # x = self.block_3(x)
        # x = self.classifier(x)
        # return x

        return self.classifier(self.block_3(self.block_2(self.block_1(x))))


if __name__ == "__main__":

    input_shape = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyModel(input_shape, 10).to(device)

    dummy_data = torch.randn(32, 3, input_shape, input_shape).to(device)
    summary(model, input_size=(32, 3, input_shape, input_shape))

    print(device)