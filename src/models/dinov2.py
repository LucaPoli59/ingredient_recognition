import torch
import torch.nn as nn
from abc import ABC
from typing import Callable, Optional, Tuple, List, Any, Dict

from torchinfo import summary
from torchvision.transforms import v2

from config import DEF_IMAGE_SHAPE
from data_processing.transformations import transform_aug_dino, transform_plain_dino, transform_core_dino
from src.models.commons import BaseModel

class _BaseDinoV2(BaseModel, ABC):
    PRETTY_NAME = "BaseDinoV2"

    trns_bld_form = Callable[[Optional[torch.tensor], Tuple[int, int]], List[v2.Transform]]
    DEF_TRNS_BLD_AUG = transform_aug_dino
    DEF_TRNS_BLD_PLAIN = transform_plain_dino
    DEF_TRNS_BLD_BASE = transform_core_dino

    def __init__(self, weights, num_classes, input_shape, pretrained, freeze_backbone=True,
                 trns_aug=None, trns_bld_aug: Optional[trns_bld_form] = None, 
                 trns_bld_plain: Optional[trns_bld_form] = None, lp_phase=None):
        super().__init__(num_classes=num_classes, input_shape=input_shape, 
                         trns_aug=trns_aug, trns_bld_aug=trns_bld_aug, 
                         trns_bld_plain=trns_bld_plain, lp_phase=lp_phase)
        self.pretrained = pretrained
        if freeze_backbone is None:
            freeze_backbone = True
        self.freeze_backbone_flag = freeze_backbone

        self.model = torch.hub.load("facebookresearch/dinov2", weights + "_lc")
        self.model.linear_head = nn.Linear(self.model.linear_head.weight.shape[1], num_classes)

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x):
        return self.model(x)

    def to_config(self):
        config = super().to_config()
        config["pretrained"] = self.pretrained
        config["freeze_backbone"] = self.freeze_backbone_flag
        return config

    @classmethod
    def _load_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        params = super()._load_config(config)
        params["pretrained"] = config["pretrained"]
        params["freeze_backbone"] = config["freeze_backbone"]
        params["lp_phase"] = None  # default value
        return params

    @property
    def transform_aug(self):
        if self.trns_aug is not None:
            return self.__class__.DEF_TRNS_BLD_BASE(self.input_shape, augmentations=self.trns_aug(), random_crop=True)
        return self.trns_bld_aug(self.input_shape)

    @property
    def transform_plain(self):
        if self.trns_aug is not None:
            return self.__class__.DEF_TRNS_BLD_BASE(self.input_shape, augmentations=None, random_crop=False)
        return self.trns_bld_plain(self.input_shape)

    def freeze_backbone(self):
        """
        Freeze the backbone parameters to prevent gradient updates.
        """
        for name, param in self.model.backbone.named_parameters():
            if not name.startswith("linear_head"):
                param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreeze the backbone parameters to allow gradient updates.
        """
        for name, param in self.model.backbone.named_parameters():
            if not name.startswith("linear_head"):
                param.requires_grad = True

    def conv_target_layer(self):
        """
        Target convolutional-like layer (can be used for visualization).
        """
        return self.model.backbone.norm


    @property
    def classifier_target_layer(self):
        """
        Target classifier layer.
        """

        return self.model.linear_head


class DinoV2B14(_BaseDinoV2):
    PRETTY_NAME = "DinoV2_B14"

    def __init__(self, num_classes, input_shape=DEF_IMAGE_SHAPE, pretrained=True, freeze_backbone=True, 
                 trns_aug=None, trns_bld_aug: Optional[_BaseDinoV2.trns_bld_form] = None, 
                 trns_bld_plain: Optional[_BaseDinoV2.trns_bld_form] = None, lp_phase=-1):
        if lp_phase is None:
            lp_phase = -1

        weights = "dinov2_vitb14_reg"
        super().__init__(weights, num_classes, input_shape, pretrained=pretrained, freeze_backbone=freeze_backbone,
                         trns_aug=trns_aug, trns_bld_aug=trns_bld_aug, trns_bld_plain=trns_bld_plain, 
                         lp_phase=lp_phase)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoV2B14(num_classes=180).to(device)
    print(model, "\n")
    print(model.to_config(), "\n")
    print(model.transform_aug, "\n")
    print(model.transform_plain, "\n")
    print(model.conv_target_layer, "\n")
    print(model.classifier_target_layer, "\n")
    x = torch.randn(1, 3, 224, 224).to(device)
    summary(model, input_data=x)
    print(model(x), "\n")