import torch
import torchvision
from torch import nn
from typing import Dict, Any, List, Optional, Tuple, Callable

from torchvision.transforms import v2
from typing_extensions import Self
from abc import ABC

from settings.config import DEF_IMAGE_SHAPE, LP_MAX_PHASE
from src.models.commons import BaseModel
from src.data_processing.transformations import transform_aug_imagenet, transform_plain_imagenet, transform_core_imagenet


class _BaseResnetLike(BaseModel, ABC):
    PRETTY_NAME = "BaseResnetLike"
    LAYER_EXPANSION = 1
    def __init__(self, num_classes, input_shape, trns_aug=None, trns_bld_aug=None, trns_bld_plain=None, lp_phase=None):
        super().__init__(num_classes=num_classes, input_shape=input_shape, trns_aug=trns_aug, trns_bld_aug=trns_bld_aug,
                         trns_bld_plain=trns_bld_plain, lp_phase=lp_phase)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1)
        )

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
            nn.Linear(int(in_features * self.LAYER_EXPANSION), num_classes, bias=True)
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
    PRETTY_NAME = "ResnetLikeV1"
    LAYER_EXPANSION = BasicBlock.expansion
    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE, trns_aug=None, trns_bld_aug=None, trns_bld_plain=None,
                 lp_phase=-1):
        if lp_phase is None:
            lp_phase = -1
        super().__init__(num_classes=num_classes, input_shape=input_shape, trns_aug=trns_aug, trns_bld_aug=trns_bld_aug,
                         trns_bld_plain=trns_bld_plain, lp_phase=lp_phase)

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
    PRETTY_NAME = "ResnetLikeV1LeakyReLUVariant"
    LAYER_EXPANSION = BasicBlockLVariant.expansion

    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE, trns_bld_aug=None, trns_bld_plain=None, lp_phase=-1):
        if lp_phase is None:
            lp_phase = -1
        super().__init__(num_classes=num_classes, input_shape=input_shape, trns_bld_aug=trns_bld_aug,
                         trns_bld_plain=trns_bld_plain, lp_phase=lp_phase)

        self._layer1 = self._make_layer(BasicBlockLVariant, 64, 64, 2)
        self._layer2 = self._make_layer(BasicBlockLVariant, 64, 128, 2, stride=2)
        self._layer3 = self._make_layer(BasicBlockLVariant, 128, 256, 2, stride=2)
        self._layer4 = self._make_layer(BasicBlockLVariant, 256, 512, 2, stride=2)

        self.classifier = self._make_classifier(512, num_classes)
        self._lp_init_layers()


class ResnetLikeV2(_BaseResnetLike):  #Like Resnet50
    PRETTY_NAME = "ResnetLikeV2"
    LAYER_EXPANSION = BottleneckBlock.expansion

    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE, trns_aug=None, trns_bld_aug=None, trns_bld_plain=None,
                 lp_phase=-1):
        if lp_phase is None:
            lp_phase = -1
        super().__init__(num_classes=num_classes, input_shape=input_shape, trns_aug=trns_aug, trns_bld_aug=trns_bld_aug,
                         trns_bld_plain=trns_bld_plain, lp_phase=lp_phase)

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
    PRETTY_NAME = "BaseResnet"

    trns_bld_form = Callable[[Optional[torch.tensor], Tuple[int, int]], List[v2.Transform]]
    DEF_TRNS_BLD_AUG = transform_aug_imagenet
    DEF_TRNS_BLD_PLAIN = transform_plain_imagenet
    DEF_TRNS_BLD_BASE = transform_core_imagenet

    def __init__(self, num_classes, input_shape, pretrained, trns_aug=None, trns_bld_aug: Optional[trns_bld_form] = None,
                 trns_bld_plain: Optional[trns_bld_form] = None, lp_phase=None):
        super().__init__(num_classes=num_classes, input_shape=input_shape, trns_aug=trns_aug, trns_bld_aug=trns_bld_aug,
                         trns_bld_plain=trns_bld_plain, lp_phase=None)  # LP not supported
        self.pretrained = pretrained

    def to_config(self):
        config = super().to_config()
        config["pretrained"] = self.pretrained
        return config

    @classmethod
    def _load_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        params = super()._load_config(config)
        params["pretrained"] = config["pretrained"]
        params["lp_phase"] = None  # default value
        return params

    @property
    def transform_aug(self):
        if self.trns_aug is not None:
            return self.__class__.DEF_TRNS_BLD_BASE(self.tr_weights, self.input_shape, augmentations=self.trns_aug())
        return self.trns_bld_aug(self.tr_weights, self.input_shape)

    @property
    def transform_plain(self):
        if self.trns_aug is not None:
            return self.__class__.DEF_TRNS_BLD_BASE(self.tr_weights, self.input_shape, augmentations=None)
        return self.trns_bld_plain(self.tr_weights, self.input_shape)


class Resnet18(_BaseResnet):
    PRETTY_NAME = "Resnet18"
    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE, trns_aug=None,
                 trns_bld_aug: Optional[_BaseResnet.trns_bld_form] = None,
                 trns_bld_plain: Optional[_BaseResnet.trns_bld_form] = None,
                 pretrained=True, lp_phase=None):
        super().__init__(num_classes, input_shape, trns_aug=trns_aug, trns_bld_aug=trns_bld_aug, trns_bld_plain=trns_bld_plain,
                         lp_phase=lp_phase, pretrained=pretrained)

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


class Resnet50(_BaseResnet):
    PRETTY_NAME = "Resnet50"
    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE, trns_aug=None,
                 trns_bld_aug: Optional[_BaseResnet.trns_bld_form] = None,
                 trns_bld_plain: Optional[_BaseResnet.trns_bld_form] = None,
                 pretrained=True, lp_phase=None):
        super().__init__(num_classes, input_shape, trns_aug=trns_aug, trns_bld_aug=trns_bld_aug, trns_bld_plain=trns_bld_plain,
                         lp_phase=lp_phase, pretrained=pretrained)

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


from src.data_processing.transformations import transform_aug_imagenet_adv

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