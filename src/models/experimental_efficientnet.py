"""The 224-pixel 4A EfficientNet experiment, separate from the Phase 3 selector."""

from __future__ import annotations

import json
from typing import Any

import torch
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from src.data_processing.experimental_transforms import ExperimentalFitPad224
from src.models.commons import BaseModel
from src.models.experimental_contract import ExperimentalModelContract, make_experimental_linear


class EfficientNetV2SExperiment(BaseModel):
    """Intact EfficientNetV2-S features followed by GAP and a new linear head.

    ``initialize_pretrained=False`` is operational: use it only to reconstruct
    a complete saved state offline or for a labelled no-network engineering
    fixture. It never changes the saved original ImageNet weight identity.
    There is no qualified physical-batch cap until the Phase 5.5 measurement.
    """

    PRETTY_NAME = "EfficientNetV2-S 4A experiment"
    WEIGHTS_ENUM = "EfficientNet_V2_S_Weights.IMAGENET1K_V1"
    WEIGHTS_URL = EfficientNet_V2_S_Weights.IMAGENET1K_V1.url

    def __init__(
            self, num_classes: int | None = None, input_shape=(224, 224),
            trns_aug=None, trns_bld_aug=None, trns_bld_plain=None, lp_phase=None,
            experimental_contract: dict | None = None,
            contract: ExperimentalModelContract | None = None,
            adaptation: str | None = None, head_seed: int | None = None,
            horizontal_flip_probability: float | None = None,
            initialize_pretrained: bool = True,
    ):
        if type(initialize_pretrained) is not bool:
            raise ValueError("initialize_pretrained must be a boolean operational flag")
        shape = self._validate_input_shape(input_shape)
        if any(value is not None for value in (trns_aug, trns_bld_aug, trns_bld_plain, lp_phase)):
            raise ValueError("the experimental contract owns transforms and does not support layer pretraining")
        if contract is not None and experimental_contract is not None:
            raise ValueError("provide contract or experimental_contract, not both")
        if experimental_contract is not None:
            contract = ExperimentalModelContract.from_config(experimental_contract)
        if contract is not None and not isinstance(contract, ExperimentalModelContract):
            raise ValueError("contract must be a validated ExperimentalModelContract")
        if contract is None:
            contract = ExperimentalModelContract(
                "efficientnet_v2_s", 165 if num_classes is None else num_classes,
                "full" if adaptation is None else adaptation,
                42 if head_seed is None else head_seed,
                0.0 if horizontal_flip_probability is None else horizontal_flip_probability,
            )
        if contract.model_id != "efficientnet_v2_s":
            raise ValueError("EfficientNetV2SExperiment requires its own model identity")
        for name, value, expected in (
                ("num_classes", num_classes, contract.num_classes),
                ("adaptation", adaptation, contract.adaptation),
                ("head_seed", head_seed, contract.head_seed),
                ("horizontal_flip_probability", horizontal_flip_probability, contract.horizontal_flip_probability)):
            if value is not None and (isinstance(value, bool) or value != expected):
                raise ValueError(f"{name} disagrees with the experimental contract")
        # Validate redundant integer values even when Python equality would accept 59.0.
        for name, value in (("num_classes", num_classes), ("head_seed", head_seed)):
            if value is not None and type(value) is not int:
                raise ValueError(f"{name} must be an integer")

        super().__init__(num_classes=contract.num_classes, input_shape=shape, lp_phase=None)
        self.experimental_contract = contract
        self.initialize_pretrained = initialize_pretrained
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if initialize_pretrained else None
        # Construct the unmodified 1000-way library model first. Loading its
        # complete original weights precedes replacement of its entire readout.
        self.model = efficientnet_v2_s(weights=weights)
        stock = self.model.classifier
        if not (isinstance(stock, torch.nn.Sequential) and len(stock) == 2
                and isinstance(stock[0], torch.nn.Dropout) and stock[0].p == 0.2
                and isinstance(stock[1], torch.nn.Linear) and stock[1].in_features == 1280
                and stock[1].out_features == 1000 and stock[1].bias is not None
                and isinstance(self.model.avgpool, torch.nn.AdaptiveAvgPool2d)):
            raise RuntimeError("unsupported torchvision EfficientNetV2-S readout structure")
        self.model.classifier = make_experimental_linear(1280, self.num_classes, contract.head_seed)
        for parameter in self.model.features.parameters():
            parameter.requires_grad_(contract.adaptation == "full")
        # Hold every feature buffer/stochastic operation in eval, not only BN.
        self.train(self.training)

    @staticmethod
    def _validate_input_shape(input_shape):
        if type(input_shape) is int:
            shape = (input_shape, input_shape)
        elif isinstance(input_shape, (tuple, list)) and len(input_shape) == 2:
            shape = tuple(input_shape)
        else:
            raise ValueError("EfficientNetV2SExperiment requires the declared 224x224 input")
        if any(type(side) is not int for side in shape) or shape != (224, 224):
            raise ValueError("EfficientNetV2SExperiment requires the declared 224x224 input")
        return shape

    @property
    def adaptation(self):
        return self.experimental_contract.adaptation

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "model") and self.adaptation == "frozen_encoder":
            self.model.features.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if (not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] <= 0
                or tuple(image.shape[1:]) != (3, 224, 224)):
            raise ValueError("experimental EfficientNet input must have shape [B,3,224,224], B>0")
        # Do not wrap a frozen encoder in no_grad(): Grad-CAM/input-gradient
        # diagnostics remain valid when callers explicitly request gradients.
        return self.model(image)

    @property
    def conv_target_layer(self):
        return self.model.features[-1]

    @property
    def classifier_target_layer(self):
        return self.model.classifier

    @property
    def transform_aug(self):
        return ExperimentalFitPad224(self.experimental_contract.horizontal_flip_probability)

    @property
    def transform_plain(self):
        return ExperimentalFitPad224(0.0)

    def to_config(self) -> dict[str, Any]:
        config = super().to_config()
        config["experimental_contract"] = self.experimental_contract.to_config()
        return config

    @classmethod
    def _load_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        expected = {"type", "num_classes", "input_shape", "trns_aug", "trns_bld_aug",
                    "trns_bld_plain", "lp_phase", "experimental_contract"}
        if not isinstance(config, dict) or set(config) != expected:
            raise ValueError("experimental model configuration has missing or unsupported fields")
        params = super()._load_config(config)
        contract = ExperimentalModelContract.from_config(config["experimental_contract"])
        if (contract.model_id != "efficientnet_v2_s" or type(params["num_classes"]) is not int
                or params["num_classes"] != contract.num_classes):
            raise ValueError("model configuration disagrees with its experimental identity")
        params["input_shape"] = cls._validate_input_shape(params["input_shape"])
        if any(params[name] is not None for name in ("trns_aug", "trns_bld_aug", "trns_bld_plain", "lp_phase")):
            raise ValueError("unsupported transform or layer-pretraining override")
        params["contract"] = contract
        return params

    @classmethod
    def load_from_config(cls, config: str | dict[str, Any], *, initialize_pretrained: bool = True):
        if isinstance(config, str):
            config = json.loads(config)
        return cls(**cls._load_config(config), initialize_pretrained=initialize_pretrained)
