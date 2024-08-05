import torch
import torchvision
from torch import nn
from typing import Dict, Any, Tuple
from typing_extensions import Self
from abc import ABC

from settings.config import DEF_IMAGE_SHAPE
from src.models.commons import BaseModel
from src.data_processing.transformations import transform_aug_imagenet, transform_plain_imagenet


class _BaseDensenetLike(BaseModel, ABC):
    """NB: densenet convs follow the pattern of bn -> relu -> conv"""

    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1)
        )

    @classmethod
    def load_from_config(cls, config: Dict[str, Any], **kwargs) -> Self:
        for key in ["num_classes", "input_shape"]:
            if key not in config:
                raise ValueError(f"The configuration must contain the key '{key}'")
        return cls(config["num_classes"], config["input_shape"])

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
    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE):
        super().__init__(num_classes, input_shape)

        self.dense_1, out_chs = self._make_block(DenseLayer, 64, 6)  # 64 -> 256
        self.transition_1 = TransitionLayer(out_chs, DenseLayer.reduction_factor)  # 256 -> 128

        self.dense_2, out_chs = self._make_block(DenseLayer, 128, 12)  # 128 -> 512
        self.transition_2 = TransitionLayer(out_chs, DenseLayer.reduction_factor)  # 512 -> 256

        self.dense_3, out_chs = self._make_block(DenseLayer, 256, 24)  # 256 -> 1024
        self.transition_3 = TransitionLayer(out_chs, DenseLayer.reduction_factor)  # 1024 -> 512

        self.dense_4, out_chs = self._make_block(DenseLayer, 512, 16)  # 512 -> 1024

        self.bn_final = nn.BatchNorm2d(out_chs)
        self.final_relu = nn.ReLU()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_chs, num_classes, bias=True)
        )


    def forward(self, x):
        out = self.conv1(x)
        out = self.transition_1(self.dense_1(out))
        out = self.transition_2(self.dense_2(out))
        out = self.transition_3(self.dense_3(out))
        out = self.final_relu(self.bn_final(self.dense_4(out)))

        return self.classifier(out)

    @property
    def conv_target_layer(self):
        return self.final_relu

    @property
    def classifier_target_layer(self):
        return self.classifier[-1]


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
        self.final_relu = nn.ReLU()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_chs, num_classes, bias=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.transition_1(self.dense_1(out))
        out = self.transition_2(self.dense_2(out))
        out = self.transition_3(self.dense_3(out))
        out = self.final_relu(self.bn_final(self.dense_4(out)))

        return self.classifier(out)

    @property
    def conv_target_layer(self):
        return self.final_relu

    @property
    def classifier_target_layer(self):
        return self.classifier[-1]


class _BaseDensenet(BaseModel, ABC):
    def __init__(self, num_classes, input_shape, pretrained):
        super().__init__(num_classes=num_classes, input_shape=input_shape, pretrained=pretrained)
        self.pretrained = pretrained

    @classmethod
    def load_from_config(cls, config: Dict[str, Any], **kwargs) -> Self:
        for key in ["num_classes", "input_shape", "pretrained"]:
            if key not in config:
                raise ValueError(f"The configuration must contain the key '{key}'")
        return cls(config["num_classes"], config["input_shape"], config["pretrained"])

    @property
    def transform_aug(self):
        return transform_aug_imagenet(self.tr_weights, self.input_shape)

    @property
    def transform_plain(self):
        return transform_plain_imagenet(self.tr_weights, self.input_shape)


class Densenet121(_BaseDensenet):
    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE, pretrained=True):
        super().__init__(num_classes, input_shape, pretrained)
        weights = torchvision.models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = torchvision.models.densenet121(weights=weights)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    @property
    def conv_target_layer(self):
        return self.model.features[-1]

    @property
    def classifier_target_layer(self):
        return self.model.classifier


class Densenet201(_BaseDensenet):
    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE, pretrained=True):
        super().__init__(num_classes, input_shape, pretrained)
        weights = torchvision.models.DenseNet201_Weights.DEFAULT if pretrained else None
        self.model = torchvision.models.densenet201(weights=weights)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)


    def forward(self, x):
        return self.model(x)

    @property
    def conv_target_layer(self):
        return self.model.features[-1]

    @property
    def classifier_target_layer(self):
        return self.model.classifier


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 224, 224).to(device)
    densenet121 = Densenet121(183).to(device)
    densenet_like1 = DensenetLikeV1(183).to(device)

    from torchinfo import summary

    # summary(model, input_size=(1, 3, 224, 224))
    print(densenet121(x).shape)
    print(densenet_like1(x).shape)

    print(densenet121)
    print(densenet_like1)

    summary(densenet121, input_size=(1, 3, 224, 224), depth=3)
    summary(densenet_like1, input_size=(1, 3, 224, 224), depth=1)
