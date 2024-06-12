import torch
import torch.functional as F
import torchvision
from torch import nn
from typing import Dict, Any

from typing_extensions import Self

from src.models.commons import BaseModel


class _BaseResnetLike(BaseModel):
    def __init__(self, num_classes):
        super().__init__(num_classes=num_classes)
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1)
        )

    @classmethod
    def load_from_config(cls, config: Dict[str, Any], **kwargs) -> Self:
        if "num_classes" not in config:
            raise ValueError("The configuration must contain the key 'num_classes' with the number of classes")
        return cls(config["num_classes"])

    @staticmethod
    def _make_layer(block_type, in_channel, out_channel, num_blocks, stride=1):
        blocks = [block_type(in_channel, out_channel, stride)]
        in_channel = out_channel * getattr(block_type, "expansion", 1)
        for _ in range(1, num_blocks):
            blocks.append(block_type(in_channel, out_channel))
        return nn.Sequential(*blocks)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()

        if in_channel != out_channel or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)  # Skip connection
        return self.relu(out)


class BottleneckBlock(torch.nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        # 1x1 Convolution [Reduce the dimensionality]
        self.conv1 = nn.Conv2d(in_channel, out_channel, 1, padding=0, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        # 3x3 Convolution [Regular Convolution]
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        # 1x1 Convolution [Increase the dimensionality]
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, 1, padding=0, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU()

        if in_channel != out_channel * self.expansion or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * self.expansion)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)  # Skip connection

        return self.relu(out)


class ResnetLikeV1(_BaseResnetLike):  #Like Resnet18
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 64, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 512, 2, stride=2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes, bias=True)
        )

        self.classifier_target_layer = self.classifier[-1]

        self.conv_target_layer = self.layer4[-1]

    def forward(self, x):
        out = self.layer4(self.layer3(self.layer2(self.layer1(self.conv1(x)))))
        return self.classifier(out)


class ResnetLikeV2(_BaseResnetLike):  #Like Resnet50
    def __init__(self, num_classes):
        super().__init__(num_classes)

        self.layer1 = self._make_layer(BottleneckBlock, 64, 64, 3)
        self.layer2 = self._make_layer(BottleneckBlock, 256, 128, 4, stride=2)
        self.layer3 = self._make_layer(BottleneckBlock, 512, 256, 6, stride=2)
        self.layer4 = self._make_layer(BottleneckBlock, 1024, 512, 3, stride=2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(int(512 * BottleneckBlock.expansion), num_classes, bias=True)
        )

        self.classifier_target_layer = self.classifier[-1]

        self.conv_target_layer = self.layer4[-1]

    def forward(self, x):
        out = self.layer4(self.layer3(self.layer2(self.layer1(self.conv1(x)))))
        return self.classifier(out)


class _BaseResnet(BaseModel):
    def __init__(self, num_classes, pretrained):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.num_classes = num_classes

    @classmethod
    def load_from_config(cls, config: Dict[str, Any], **kwargs) -> Self:
        for key in ["num_classes", "pretrained"]:
            if key not in config:
                raise ValueError(f"The configuration must contain the key '{key}'")
        return cls(config["num_classes"], config["pretrained"])


class Resnet18(_BaseResnet):
    def __init__(self, num_classes, pretrained=True):
        super().__init__(num_classes, pretrained)
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = torchvision.models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.conv_target_layer = self.model.layer4[-1]
        self.classifier_target_layer = self.model.fc

    def forward(self, x):
        return self.model(x)


class Resnet50(_BaseResnet):
    def __init__(self, num_classes, pretrained=True):
        super().__init__(num_classes, pretrained)

        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = torchvision.models.resnet50(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.conv_target_layer = self.model.layer4[-1]
        self.classifier_target_layer = self.model.fc

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18 = Resnet18(183).to(device)
    resnet_like1 = ResnetLikeV1(183).to(device)
    x = torch.randn(1, 3, 224, 224).to(device)

    from torchinfo import summary

    print(resnet18)
    print(resnet_like1)

    summary(resnet18, input_size=(1, 3, 224, 224), depth=2)
    summary(resnet_like1, input_size=(1, 3, 224, 224), depth=1)


