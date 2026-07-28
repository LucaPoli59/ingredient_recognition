import ctypes
import math
import os

from abc import ABC
from typing import Callable, Optional, Tuple, List, Any, Dict

import torch
import torch.nn as nn
from torchinfo import summary
from torchvision.transforms import v2
from typing_extensions import override

from config import DEF_IMAGE_SHAPE
from data_processing.transformations import transform_aug_dino, transform_plain_dino, transform_core_dino
from src.models.commons import BaseModel


class _BaseDinoV2(BaseModel, ABC):
    PRETTY_NAME = "BaseDinoV2"
    MAX_ALLOWED_BATCH_SIZE = 32

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

    @override
    @property
    def max_allowed_batch_size(self) -> int | None:
        return self.MAX_ALLOWED_BATCH_SIZE

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

    @property
    def conv_target_layer(self):
        """
        Last pre-attention normalization. Its token activations directly affect
        the final transformer block and therefore the classifier output.
        """
        return self.model.backbone.blocks[-1].norm1

    @property
    def gradcam_reshape_transform(self):
        """Map ViT tokens to the patch grid expected by Grad-CAM and DFF."""
        num_register_tokens = self.model.backbone.num_register_tokens

        def reshape_transform(tokens: torch.Tensor) -> torch.Tensor:
            # Drop [CLS] and the register tokens: neither has a location in the image.
            patch_tokens = tokens[:, 1 + num_register_tokens:, :]
            batch_size, num_patches, channels = patch_tokens.shape
            side = math.isqrt(num_patches)
            if side * side != num_patches:
                raise ValueError(f"DINOv2 Grad-CAM expects a square patch grid, got {num_patches} patches")
            return patch_tokens.reshape(batch_size, side, side, channels).permute(0, 3, 1, 2).contiguous()

        return reshape_transform


    @property
    def classifier_target_layer(self):
        """
        Target classifier layer.
        """

        return self.model.linear_head

    @property
    def factorization_classifier_layer(self):
        """Score a 768-D patch concept through the patch-pooling slice of the LC head."""
        return _DinoV2PatchClassifier(self.model.linear_head, self.model.backbone.embed_dim)


class _DinoV2PatchClassifier(nn.Module):
    """Projection used by DFF for patch-token concepts.

    The hub ``*_lc`` head receives four CLS tokens plus the mean of final patch
    tokens. A DFF concept is a single patch-channel vector, hence only the last
    (mean-patch) slice of the trained linear head is applicable.
    """

    def __init__(self, linear_head: nn.Linear, embed_dim: int):
        super().__init__()
        self.linear_head = linear_head
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DeepFeatureFactorization constructs its NMF concepts on CPU even when
        # the model is on CUDA.
        x = x.to(self.linear_head.weight.device)
        return torch.nn.functional.linear(x, self.linear_head.weight[:, -self.embed_dim:], self.linear_head.bias)


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
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())

    # Test bare CUDA first, before loading the model
    t = torch.randn(1, 3, 224, 224).to(device)
    t2 = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
    with torch.no_grad():
        out = t2(t)
    print("Bare CUDA conv works:", out.shape)  # if this fails, it's a system issue

    model = DinoV2B14(num_classes=10).to(device)

    # Now test on GPU
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        out_gpu = model(x)
    print("GPU forward pass works:", out_gpu.shape)

    summary(model, input_data=x, device=device)
