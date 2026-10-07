"""DICA-Net-S approved initialization, adaptation and offline persistence."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import torch
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from src.data_processing.experimental_transforms import ExperimentalFitPad224
from src.models.commons import BaseModel
from src.models.experimental_contract import ExperimentalModelContract
from src.models.dica_net import DICANetSCore


class DICANetSExperiment(BaseModel):
    """Intact EfficientNetV2-S features with the approved dual-scale query head.

    ``initialize_pretrained=False`` is operational: use it only to reconstruct
    a complete saved state offline or for a labelled no-network engineering
    fixture. It never changes the saved original ImageNet weight identity.
    There is no qualified physical-batch cap until the Phase 5.5 measurement.
    """

    PRETTY_NAME = "DICA-Net-S"
    WEIGHTS_ENUM = "EfficientNet_V2_S_Weights.IMAGENET1K_V1"
    WEIGHTS_URL = "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth"
    WEIGHTS_SHA256 = "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"
    WEIGHTS_SIZE_BYTES = 86_721_253

    def __init__(
            self, num_classes: int | None = None, input_shape=(224, 224),
            trns_aug=None, trns_bld_aug=None, trns_bld_plain=None, lp_phase=None,
            experimental_contract: dict | None = None,
            contract: ExperimentalModelContract | None = None,
            adaptation: str | None = None, head_seed: int | None = None,
            horizontal_flip_probability: float | None = None,
            initialize_pretrained: bool = True,
            encoder_provenance: dict | None = None,
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
                "p2_s", 165 if num_classes is None else num_classes,
                "full" if adaptation is None else adaptation,
                42 if head_seed is None else head_seed,
                0.0 if horizontal_flip_probability is None else horizontal_flip_probability,
            )
        if contract.model_id != "p2_s":
            raise ValueError("DICA-Net-S requires its retained p2_s model identity")
        if encoder_provenance is not None:
            self._validate_encoder_provenance(encoder_provenance)
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
        if EfficientNet_V2_S_Weights.IMAGENET1K_V1.url != self.WEIGHTS_URL:
            raise RuntimeError("the library weight URL differs from the pinned DICA-Net artifact")
        backbone = efficientnet_v2_s(weights=weights)
        if initialize_pretrained:
            self._verify_original_artifact()
        stock = backbone.classifier
        if not (isinstance(stock, torch.nn.Sequential) and len(stock) == 2
                and isinstance(stock[0], torch.nn.Dropout) and stock[0].p == 0.2
                and isinstance(stock[1], torch.nn.Linear) and stock[1].in_features == 1280
                and stock[1].out_features == 1000 and stock[1].bias is not None
                and isinstance(backbone.avgpool, torch.nn.AdaptiveAvgPool2d)):
            raise RuntimeError("unsupported torchvision EfficientNetV2-S readout structure")
        norms = [module for module in backbone.features.modules() if isinstance(module, torch.nn.BatchNorm2d)]
        if not norms or any(module.eps != 1e-3 or module.momentum != .1
                            or not module.affine or not module.track_running_stats for module in norms):
            raise RuntimeError("unsupported EfficientNetV2-S normalization policy")
        self.model = DICANetSCore(backbone.features, self.num_classes, contract.head_seed)
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
            raise ValueError("DICA-Net-S requires the declared 224x224 input")
        if any(type(side) is not int for side in shape) or shape != (224, 224):
            raise ValueError("DICA-Net-S requires the declared 224x224 input")
        return shape

    @classmethod
    def _encoder_provenance(cls):
        return {
            "schema_version": 1, "constructor": "torchvision.models.efficientnet_v2_s",
            "weights_enum": cls.WEIGHTS_ENUM, "url": cls.WEIGHTS_URL,
            "sha256": cls.WEIGHTS_SHA256, "size_bytes": cls.WEIGHTS_SIZE_BYTES,
            "batch_norm_eps": 1e-3, "batch_norm_momentum": .1,
            "pretrained_running_statistics": "preserve",
        }

    @classmethod
    def _validate_encoder_provenance(cls, provenance):
        if json.dumps(provenance, sort_keys=True, allow_nan=False) != json.dumps(cls._encoder_provenance(), sort_keys=True):
            raise ValueError("DICA-Net encoder provenance or normalization differs from the approved protocol")

    @classmethod
    def _verify_original_artifact(cls):
        # The public constructor has loaded the original cache entry. Prefix
        # checking by TorchVision alone is not a complete artifact identity.
        path = Path(torch.hub.get_dir()) / "checkpoints" / cls.WEIGHTS_URL.rsplit("/", 1)[1]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != cls.WEIGHTS_SIZE_BYTES or digest.hexdigest() != cls.WEIGHTS_SHA256:
            raise ValueError("DICA-Net original weight artifact fails its complete SHA-256/size check")

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
            raise ValueError("DICA-Net-S input must have shape [B,3,224,224], B>0")
        # Do not wrap a frozen encoder in no_grad(): Grad-CAM/input-gradient
        # diagnostics remain valid when callers explicitly request gradients.
        return self.model(image)

    @property
    def conv_target_layer(self):
        return self.model.features[-1]

    @property
    def classifier_target_layer(self):
        return self.model.readout

    @property
    def supports_feature_factorization(self):
        return False

    @property
    def feature_factorization_unavailable_reason(self):
        return ("DICA-Net-S scores ingredient queries using both feature scales and pooled context. "
                "Standalone feature concepts cannot be scored by its complete classifier.")

    @property
    def factorization_classifier_layer(self):
        raise NotImplementedError(self.feature_factorization_unavailable_reason)

    @property
    def transform_aug(self):
        return ExperimentalFitPad224(self.experimental_contract.horizontal_flip_probability)

    @property
    def transform_plain(self):
        return ExperimentalFitPad224(0.0)

    def to_config(self) -> dict[str, Any]:
        config = super().to_config()
        config["experimental_contract"] = self.experimental_contract.to_config()
        config["encoder_provenance"] = self._encoder_provenance()
        return config

    @classmethod
    def _load_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        expected = {"type", "num_classes", "input_shape", "trns_aug", "trns_bld_aug",
                    "trns_bld_plain", "lp_phase", "experimental_contract", "encoder_provenance"}
        if not isinstance(config, dict) or set(config) != expected:
            raise ValueError("experimental model configuration has missing or unsupported fields")
        params = super()._load_config(config)
        contract = ExperimentalModelContract.from_config(config["experimental_contract"])
        if (contract.model_id != "p2_s" or type(params["num_classes"]) is not int
                or params["num_classes"] != contract.num_classes):
            raise ValueError("model configuration disagrees with its experimental identity")
        params["input_shape"] = cls._validate_input_shape(params["input_shape"])
        if any(params[name] is not None for name in ("trns_aug", "trns_bld_aug", "trns_bld_plain", "lp_phase")):
            raise ValueError("unsupported transform or layer-pretraining override")
        params["contract"] = contract
        cls._validate_encoder_provenance(config["encoder_provenance"])
        params["encoder_provenance"] = config["encoder_provenance"]
        return params

    @classmethod
    def load_from_config(cls, config: str | dict[str, Any], *, initialize_pretrained: bool = True):
        if isinstance(config, str):
            config = json.loads(config)
        return cls(**cls._load_config(config), initialize_pretrained=initialize_pretrained)
