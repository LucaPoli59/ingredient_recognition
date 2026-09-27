"""Torchvision EfficientNet integrations used by maintained experiments."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from src.data_processing.transformations import (
    transform_aug_selector_efficientnet_v2_s,
    transform_plain_selector_efficientnet_v2_s,
)
from src.models.commons import BaseModel


class EfficientNetV2SSelector(BaseModel):
    """Exact model-side adapter frozen by decision 4B-D1."""

    PRETTY_NAME = "EfficientNetV2-S Phase 3 selector"
    MAX_ALLOWED_BATCH_SIZE = 8  # RTX 4060, 384 px, full fine-tuning, true FP32.
    WEIGHTS_ENUM = "EfficientNet_V2_S_Weights.IMAGENET1K_V1"
    WEIGHTS_URL = EfficientNet_V2_S_Weights.IMAGENET1K_V1.url
    WEIGHTS_SIZE = 86_721_253
    WEIGHTS_SHA256 = "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"
    DEF_TRNS_BLD_AUG = transform_aug_selector_efficientnet_v2_s
    DEF_TRNS_BLD_PLAIN = transform_plain_selector_efficientnet_v2_s

    def __init__(
            self,
            num_classes: int = 165,
            input_shape: int | Tuple[int, int] = (384, 384),
            pretrained: bool = True,
            trns_aug=None,
            trns_bld_aug=None,
            trns_bld_plain=None,
            lp_phase: Optional[int] = None,
    ):
        shape = (input_shape, input_shape) if isinstance(input_shape, int) else tuple(input_shape)
        if shape != (384, 384):
            raise ValueError("EfficientNetV2SSelector requires the frozen 384x384 input")
        if not pretrained:
            raise ValueError("the Phase 3 selector requires IMAGENET1K_V1 initialization")
        if lp_phase is not None:
            raise ValueError("the Phase 3 selector does not support layer-wise pretraining")
        super().__init__(
            num_classes=num_classes,
            input_shape=shape,
            trns_aug=trns_aug,
            trns_bld_aug=trns_bld_aug,
            trns_bld_plain=trns_bld_plain,
            lp_phase=None,
        )
        self.pretrained = True
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        dropout, classifier = self.model.classifier
        if not isinstance(dropout, torch.nn.Dropout) or dropout.p != 0.2 or not dropout.inplace:
            raise AssertionError("unexpected torchvision EfficientNetV2-S classifier dropout")
        self.model.classifier[1] = torch.nn.Linear(classifier.in_features, num_classes, bias=True)
        if not all(parameter.requires_grad for parameter in self.model.parameters()):
            raise AssertionError("4B-D1 requires every backbone and head parameter to be trainable")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    @property
    def conv_target_layer(self):
        return self.model.features[-1]

    @property
    def classifier_target_layer(self):
        return self.model.classifier[1]

    @property
    def max_allowed_batch_size(self):
        return self.MAX_ALLOWED_BATCH_SIZE

    def to_config(self) -> Dict[str, Any]:
        config = super().to_config()
        config["pretrained"] = True
        config["weights"] = self.WEIGHTS_ENUM
        return config

    @classmethod
    def _load_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        params = super()._load_config(config)
        if config.get("weights", cls.WEIGHTS_ENUM) != cls.WEIGHTS_ENUM:
            raise ValueError("only the frozen IMAGENET1K_V1 selector weights are supported")
        params["pretrained"] = config.get("pretrained", True)
        return params
