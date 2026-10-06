"""The 224-pixel 4A MaxViT-T experiment with the shared pooled linear readout."""

from __future__ import annotations

import json
from typing import Any

import torch
from torchvision.models import MaxVit_T_Weights, maxvit_t
from torchvision.models.maxvit import PartitionAttentionLayer

from src.data_processing.experimental_transforms import ExperimentalFitPad224
from src.models.commons import BaseModel
from src.models.experimental_contract import ExperimentalModelContract, make_experimental_linear


class MaxViTTExperiment(BaseModel):
    """Intact MaxViT-T stem/blocks followed by GAP, flatten and a new linear head.

    ``initialize_pretrained=False`` is operational: complete offline restoration
    or a labelled no-network fixture. The saved ImageNet identity stays fixed.
    BatchNorm uses TorchVision's runtime eps=1e-3/momentum=0.01 while retaining
    every loaded running statistic. The historical pretraining momentum of
    0.99 is provenance, not a runtime override. No physical cap is qualified yet.
    """

    PRETTY_NAME = "MaxViT-T 4A experiment"
    WEIGHTS_ENUM = "MaxVit_T_Weights.IMAGENET1K_V1"
    WEIGHTS_URL = MaxVit_T_Weights.IMAGENET1K_V1.url

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
                "maxvit_t", 165 if num_classes is None else num_classes,
                "full" if adaptation is None else adaptation,
                42 if head_seed is None else head_seed,
                0.0 if horizontal_flip_probability is None else horizontal_flip_probability,
            )
        if contract.model_id != "maxvit_t":
            raise ValueError("MaxViTTExperiment requires its own model identity")
        for name, value, expected in (
                ("num_classes", num_classes, contract.num_classes),
                ("adaptation", adaptation, contract.adaptation),
                ("head_seed", head_seed, contract.head_seed),
                ("horizontal_flip_probability", horizontal_flip_probability, contract.horizontal_flip_probability)):
            if value is not None and (isinstance(value, bool) or value != expected):
                raise ValueError(f"{name} disagrees with the experimental contract")
        for name, value in (("num_classes", num_classes), ("head_seed", head_seed)):
            if value is not None and type(value) is not int:
                raise ValueError(f"{name} must be an integer")

        super().__init__(num_classes=contract.num_classes, input_shape=shape, lp_phase=None)
        self.experimental_contract = contract
        self.initialize_pretrained = initialize_pretrained
        weights = MaxVit_T_Weights.IMAGENET1K_V1 if initialize_pretrained else None
        # The original complete 1000-way state is loaded by the library before
        # replacing its entire classifier. No pretrained backbone is reinitialized.
        self.model = maxvit_t(weights=weights)
        self._validate_backbone_policy()
        stock = self.model.classifier
        if not (isinstance(stock, torch.nn.Sequential) and len(stock) == 6
                and isinstance(stock[0], torch.nn.AdaptiveAvgPool2d) and stock[0].output_size == 1
                and isinstance(stock[1], torch.nn.Flatten) and stock[1].start_dim == 1
                and stock[1].end_dim == -1
                and isinstance(stock[2], torch.nn.LayerNorm) and stock[2].normalized_shape == (512,)
                and isinstance(stock[3], torch.nn.Linear) and stock[3].in_features == 512
                and stock[3].out_features == 512 and stock[3].bias is not None
                and isinstance(stock[4], torch.nn.Tanh)
                and isinstance(stock[5], torch.nn.Linear) and stock[5].in_features == 512
                and stock[5].out_features == 1000 and stock[5].bias is None):
            raise RuntimeError("unsupported torchvision MaxViT-T readout structure")
        self.model.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1),
            make_experimental_linear(512, self.num_classes, contract.head_seed),
        )
        for encoder in (self.model.stem, self.model.blocks):
            for parameter in encoder.parameters():
                parameter.requires_grad_(contract.adaptation == "full")
        self.train(self.training)

    def _validate_backbone_policy(self):
        architecture = self.experimental_contract.to_config()["architecture"]
        expected_grids = [tuple(grid) for grid in architecture["block_grid_sizes"]]
        blocks = getattr(self.model, "blocks", ())
        if (getattr(self.model, "partition_size", None) != architecture["partition_size"]
                or [getattr(block, "grid_size", None) for block in blocks] != expected_grids):
            raise RuntimeError("unsupported torchvision MaxViT-T partition geometry")
        partitions = [module for module in self.model.modules() if isinstance(module, PartitionAttentionLayer)]
        if len(partitions) != 22:
            raise RuntimeError("unsupported torchvision MaxViT-T attention topology")
        for layer in partitions:
            partition = layer.p if layer.partition_type == "window" else layer.g
            if partition != 7 or layer.grid_size not in expected_grids:
                raise RuntimeError("unsupported torchvision MaxViT-T partition geometry")
        normalization = architecture["normalization"]
        batch_norms = [module for module in self.model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
        if not batch_norms or any(
                module.eps != normalization["batch_norm_eps"]
                or module.momentum != normalization["batch_norm_momentum"]
                or not module.affine or not module.track_running_stats for module in batch_norms):
            raise RuntimeError("unsupported torchvision MaxViT-T BatchNorm policy")

    @staticmethod
    def _validate_input_shape(input_shape):
        if type(input_shape) is int:
            shape = (input_shape, input_shape)
        elif isinstance(input_shape, (tuple, list)) and len(input_shape) == 2:
            shape = tuple(input_shape)
        else:
            raise ValueError("MaxViTTExperiment requires the declared 224x224 input")
        if any(type(side) is not int for side in shape) or shape != (224, 224):
            raise ValueError("MaxViTTExperiment requires the declared 224x224 input")
        return shape

    @property
    def adaptation(self):
        return self.experimental_contract.adaptation

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "model") and self.adaptation == "frozen_encoder":
            self.model.stem.eval()
            self.model.blocks.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if (not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] <= 0
                or tuple(image.shape[1:]) != (3, 224, 224)):
            raise ValueError("experimental MaxViT input must have shape [B,3,224,224], B>0")
        # Preserve input gradients through a frozen encoder for Grad-CAM.
        return self.model(image)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # These are deterministic geometry, not learned state. Standard strict
        # loading checks keys/shapes but would accept changed offsets or cast a
        # floating index tensor silently. Run here for parent Lightning loads too.
        for name, expected in self.named_buffers():
            if not name.endswith(".relative_position_index"):
                continue
            key = prefix + name
            if key not in state_dict:
                continue  # Let the ordinary strict loader report a missing key.
            value = state_dict[key]
            if (not isinstance(value, torch.Tensor) or value.dtype != expected.dtype
                    or value.shape != expected.shape
                    or not torch.equal(value.detach().cpu(), expected.detach().cpu())):
                raise RuntimeError(f"MaxViT relative-position geometry differs at {key}")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @property
    def conv_target_layer(self):
        return self.model.blocks[-1]

    @property
    def classifier_target_layer(self):
        # Factorization supplies already-pooled [concept,512] vectors; expose
        # the complete linear scoring operation rather than the pooling container.
        return self.model.classifier[-1]

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
        if (contract.model_id != "maxvit_t" or type(params["num_classes"]) is not int
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
