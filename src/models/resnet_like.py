import torch
from torch import nn


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding="same", bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.final_relu = nn.ReLU()

        if in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.downsample = nn.Identity()
        self.conv_target_layer = self.final_relu

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out += self.downsample(x)  # Skip connection
        return self.final_relu(out)


class ResnetLikeV1(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape)
        elif len(input_shape) > 2 or len(input_shape) < 1:
            raise ValueError("The input_shape must be a tuple of 2 integers")
        elif input_shape[0] != input_shape[1]:
            raise ValueError("The input_shape must be a tuple of 2 equal integers")

        super().__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1)
        )
        self.layer1 = self._make_layer(ResidualBlock, 64, 64, 2)
        self.layer2 = self._make_layer(ResidualBlock, 64, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 128, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 256, 512, 2, stride=2)

        self.num_halving = 5
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((input_shape[0] // (2 ** self.num_halving)) ** 2 * 512, num_classes)
        )

        self.classifier_target_layer = self.classifier[-1]

    @staticmethod
    def _make_layer(block_type, in_channel, out_channel, num_blocks, stride=1):
        layers = [block_type(in_channel, out_channel, stride)]
        for _ in range(1, num_blocks):
            layers.append(block_type(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(self.layer4(self.layer3(self.layer2(self.layer1(self.conv1(x))))))
