import torch
import torchvision
from torch import nn
from typing import Dict, Any, List, Optional
from typing_extensions import Self
from abc import ABC

from settings.config import DEF_IMAGE_SHAPE, LP_MAX_PHASE
from src.models.commons import BaseModel
from src.data_processing.transformations import transform_aug_imagenet, transform_plain_imagenet


class _BaseResnetLike(BaseModel, ABC):
    def __init__(self, num_classes, input_shape, lp_phase=None, layers_expansion=1):
        self.layers_expansion = layers_expansion
        super().__init__(num_classes=num_classes, input_shape=input_shape, lp_phase=lp_phase)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1)
        )


    @classmethod
    def load_from_config(cls, config: Dict[str, Any], **kwargs) -> Self:
        for key in ["num_classes", "input_shape", "lp_phase"]:
            if key not in config:
                raise ValueError(f"The configuration must contain the key '{key}'")
        return cls(config["num_classes"], config["input_shape"], lp_phase=config["lp_phase"])

    @staticmethod
    def _make_layer(block_type, in_channel, out_channel, num_blocks, stride=1):
        blocks = [block_type(in_channel, out_channel, stride)]
        in_channel = out_channel * getattr(block_type, "expansion", 1)
        for _ in range(1, num_blocks):
            blocks.append(block_type(in_channel, out_channel))
        return nn.Sequential(*blocks)

    def _make_classifier(self, in_features, num_classes, device=None):
        if device is None:
            device = self.device
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(int(in_features * self.layers_expansion), num_classes, bias=True)
        ).to(device)

    def _lp_init_layer(self, phase, as_placeholder=True, placeholder=torch.nn.Identity()):
        if as_placeholder:
            setattr(self, f"layer{phase}", placeholder)
            if phase == 0:
                self.classifier = self._make_classifier(self.conv1[0].weight.shape[0], self.num_classes)
        else:
            if phase > 0:
                setattr(self, f"layer{phase}", getattr(self, f"_layer{phase}"))

    def _lp_post_reset(self):
        # in the normal case, don't need to replace the classifier
        if not self._lp_phase == LP_MAX_PHASE:
            last_layer = getattr(self, f"layer{self._lp_phase}")
            out_channel = list(last_layer.parameters())[-1].shape[0]
            self.classifier = self._make_classifier(out_channel, self.num_classes)

    def _lp_get_last_trained_layers(self) -> List[nn.Module] | None:
        if self._lp_phase == 0:
            return [getattr(self, "conv1")]
        return [getattr(self, f"layer{self._lp_phase}")]

    def _lp_step_phase(self):
        new_layer = getattr(self, f"_layer{self._lp_phase + 1}")
        setattr(self, f"layer{self._lp_phase + 1}", new_layer)
        out_channel = list(new_layer.parameters())[-1].shape[0]
        self.classifier = self._make_classifier(out_channel, self.num_classes)



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

        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)  # Skip connection
        return self.relu(out)


class BasicBlockLVariant(BasicBlock):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__(in_channel, out_channel, stride)
        self.relu = nn.LeakyReLU()


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
    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE, lp_phase=-1):
        if lp_phase is None:
            lp_phase = -1
        super().__init__(num_classes, input_shape, lp_phase=lp_phase)

        self._layer1 = self._make_layer(BasicBlock, 64, 64, 2)
        self._layer2 = self._make_layer(BasicBlock, 64, 128, 2, stride=2)
        self._layer3 = self._make_layer(BasicBlock, 128, 256, 2, stride=2)
        self._layer4 = self._make_layer(BasicBlock, 256, 512, 2, stride=2)

        self.classifier = self._make_classifier(512, num_classes)
        self._lp_init_layers()


    def forward(self, x):
        out = self.layer4(self.layer3(self.layer2(self.layer1(self.conv1(x)))))
        return self.classifier(out)

    @property
    def conv_target_layer(self):
        return self._layer4[-1]

    @property
    def classifier_target_layer(self):
        return self.classifier[-1]


class ResnetLikeV1LVariant(ResnetLikeV1):  #Like Resnet18
    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE, lp_phase=-1):
        if lp_phase is None:
            lp_phase = -1
        super().__init__(num_classes, input_shape, lp_phase=lp_phase)

        self._layer1 = self._make_layer(BasicBlockLVariant, 64, 64, 2)
        self._layer2 = self._make_layer(BasicBlockLVariant, 64, 128, 2, stride=2)
        self._layer3 = self._make_layer(BasicBlockLVariant, 128, 256, 2, stride=2)
        self._layer4 = self._make_layer(BasicBlockLVariant, 256, 512, 2, stride=2)

        self.classifier = self._make_classifier(512, num_classes)
        self._lp_init_layers()


class ResnetLikeV2(_BaseResnetLike):  #Like Resnet50
    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE, lp_phase=-1):
        if lp_phase is None:
            lp_phase = -1
        super().__init__(num_classes, input_shape, lp_phase=lp_phase)
        self.layers_expansion = BottleneckBlock.expansion

        self._layer1 = self._make_layer(BottleneckBlock, 64, 64, 3)
        self._layer2 = self._make_layer(BottleneckBlock, 256, 128, 4, stride=2)
        self._layer3 = self._make_layer(BottleneckBlock, 512, 256, 6, stride=2)
        self._layer4 = self._make_layer(BottleneckBlock, 1024, 512, 3, stride=2)

        self.classifier = self._make_classifier(512, num_classes)
        self._lp_init_layers()


    def forward(self, x):
        out = self.layer4(self.layer3(self.layer2(self.layer1(self.conv1(x)))))
        return self.classifier(out)

    @property
    def conv_target_layer(self):
        return self.layer4[-1]

    @property
    def classifier_target_layer(self):
        return self.classifier[-1]


class _BaseResnet(BaseModel, ABC):
    def __init__(self, num_classes, input_shape, pretrained):
        super().__init__(num_classes=num_classes, input_shape=input_shape, pretrained=pretrained, lp_phase=None)
        self.pretrained = pretrained

    @classmethod
    def load_from_config(cls, config: Dict[str, Any], **kwargs) -> Self:
        for key in ["num_classes", "input_shape", "pretrained"]:
            if key not in config:
                raise ValueError(f"The configuration must contain the key '{key}'")
        return cls(config["num_classes"], config["input_shape"], config["pretrained"], lp_phase=None)

    @property
    def transform_aug(self):
        return transform_aug_imagenet(self.tr_weights, self.input_shape)

    @property
    def transform_plain(self):
        return transform_plain_imagenet(self.tr_weights, self.input_shape)


class Resnet18(_BaseResnet):
    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE, pretrained=True):
        super().__init__(num_classes, input_shape, pretrained)
        self.tr_weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = torchvision.models.resnet18(weights=self.tr_weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    @property
    def conv_target_layer(self):
        return self.model.layer4[-1]

    @property
    def classifier_target_layer(self):
        return self.model.fc

    @property
    def transform_aug(self):
        if self.tr_weights is None:
            return super().transform_aug
        pass

    @property
    def transform_plain(self):
        if self.tr_weights is None:
            return super().transform_plain
        return self.tr_weights.transforms(crop_size=self.input_shape[0])


class Resnet50(_BaseResnet):
    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE, pretrained=True):
        super().__init__(num_classes, input_shape, pretrained)

        self.tr_weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = torchvision.models.resnet50(weights=self.tr_weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    @property
    def conv_target_layer(self):
        return self.model.layer4[-1]

    @property
    def classifier_target_layer(self):
        return self.model.fc

    @property
    def transform_aug(self):
        if self.tr_weights is None:
            return super().transform_aug
        pass

    @property
    def transform_plain(self):
        if self.tr_weights is None:
            return super().transform_plain
        return self.tr_weights.transforms(crop_size=self.input_shape[0])


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
