import torch
import torch.functional as F
import torchvision
from torch import nn
from typing import Dict, Any, Tuple

from typing_extensions import Self

from src.models.commons import BaseModel


class _BaseDensenetLike(BaseModel):
    def __init__(self, num_classes):
        """NB: densenet convs follow the pattern of bn -> relu -> conv"""
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
    def _make_block(layer_type, in_channel, num_layers) -> Tuple[nn.Sequential, int]:
        layers = []
        for _ in range(num_layers):
            layers.append(layer_type(in_channel))
            in_channel += layer_type.growth_rate
        return nn.Sequential(*layers), in_channel


class DenseLayer(torch.nn.Module):
    reduction_factor = 0.5
    growth_rate = 32
    growth_rate_factor = 4
    mid_channels = growth_rate * growth_rate_factor

    def __init__(self, in_channel):
        super().__init__()

        # 1x1 Convolution [Reduce dimensionality (IN -> 128)]
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, self.mid_channels, 1, stride=1, padding=0, bias=False)

        # 3x3 Convolution [Extract features (128 -> 32)]
        self.bn2 = nn.BatchNorm2d(self.mid_channels)
        self.conv2 = nn.Conv2d(self.mid_channels, self.growth_rate, 3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return torch.cat([x, out], 1)  # Concatenate the input with the output (IN + 32)


class TransitionLayer(torch.nn.Module):
    def __init__(self, in_channel, reduction_factor):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channel, int(in_channel * reduction_factor), 1, stride=1, padding=0, bias=False)
        self.avg_pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return self.avg_pool(out)


class DensenetLikeV1(_BaseDensenetLike):  # Like DenseNet-121
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.num_classes = num_classes

        self.dense_1, out_chs = self._make_block(DenseLayer, 64, 6)  # 64 -> 256
        self.transition_1 = TransitionLayer(out_chs, DenseLayer.reduction_factor)  # 256 -> 128

        self.dense_2, out_chs = self._make_block(DenseLayer, 128, 12)  # 128 -> 512
        self.transition_2 = TransitionLayer(out_chs, DenseLayer.reduction_factor)  # 512 -> 256

        self.dense_3, out_chs = self._make_block(DenseLayer, 256, 24)  # 256 -> 1024
        self.transition_3 = TransitionLayer(out_chs, DenseLayer.reduction_factor)  # 1024 -> 512

        self.dense_4, out_chs = self._make_block(DenseLayer, 512, 16)  # 512 -> 1024
        self.bn_final = nn.BatchNorm2d(out_chs)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_chs, num_classes, bias=True)
        )

        self.classifier_target_layer = self.classifier[-1]
        self.conv_target_layer = self.dense_4[-1]

    def forward(self, x):
        out = self.conv1(x)
        out = self.transition_1(self.dense_1(out))
        out = self.transition_2(self.dense_2(out))
        out = self.transition_3(self.dense_3(out))
        out = self.bn_final(self.dense_4(out))

        return self.classifier(out)


class DensenetLikeV2(_BaseDensenetLike):  # Like DenseNet-201
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.num_classes = num_classes

        self.dense_1, out_chs = self._make_block(DenseLayer, 64, 6)  # 64 -> 256
        self.transition_1 = TransitionLayer(out_chs, DenseLayer.reduction_factor)  # 256 -> 128

        self.dense_2, out_chs = self._make_block(DenseLayer, 128, 12)  # 128 -> 512
        self.transition_2 = TransitionLayer(out_chs, DenseLayer.reduction_factor)  # 512 -> 256

        self.dense_3, out_chs = self._make_block(DenseLayer, 256, 48)  # 256 -> 1792
        self.transition_3 = TransitionLayer(out_chs, DenseLayer.reduction_factor)  # 1792 -> 896

        self.dense_4, out_chs = self._make_block(DenseLayer, 896, 32)  # 896 -> 1920
        self.bn_final = nn.BatchNorm2d(out_chs)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_chs, num_classes, bias=True)
        )

        self.classifier_target_layer = self.classifier[-1]
        self.conv_target_layer = self.dense_4[-1]

    def forward(self, x):
        out = self.conv1(x)
        out = self.transition_1(self.dense_1(out))
        out = self.transition_2(self.dense_2(out))
        out = self.transition_3(self.dense_3(out))
        out = self.bn_final(self.dense_4(out))

        return self.classifier(out)


class _BaseDensenet(BaseModel):
    def __init__(self, num_classes, pretrained):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.num_classes = num_classes

    @classmethod
    def load_from_config(cls, config: Dict[str, Any], **kwargs) -> Self:
        for key in ["num_classes", "pretrained"]:
            if key not in config:
                raise ValueError(f"The configuration must contain the key '{key}'")
        return cls(config["num_classes"], config["pretrained"])


class Densenet121(_BaseDensenet):
    def __init__(self, num_classes, pretrained=True):
        super().__init__(num_classes, pretrained)
        weights = torchvision.models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = torchvision.models.densenet121(weights=weights)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

        self.classifier_target_layer = self.model.classifier
        self.conv_target_layer = self.model.features[-1]

    def forward(self, x):
        return self.model(x)


class Densenet201(_BaseDensenet):
    def __init__(self, num_classes, pretrained=True):
        super().__init__(num_classes, pretrained)
        weights = torchvision.models.DenseNet201_Weights.DEFAULT if pretrained else None
        self.model = torchvision.models.densenet201(weights=weights)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

        self.classifier_target_layer = self.model.classifier
        self.conv_target_layer = self.model.features[-1]

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Densenet121(183).to(device)
    x = torch.randn(1, 3, 224, 224).to(device)

    from torchinfo import summary

    # summary(model, input_size=(1, 3, 224, 224))
    print(model(x).shape)

    print(type(model))