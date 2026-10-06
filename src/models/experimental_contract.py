"""Shared 4A protocol identity and construction primitives; no model adapters."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from collections.abc import Mapping

import torch

from src.data_processing.experimental_transforms import ExperimentalFitPad224

CHECKPOINT_CONTRACT_KEY = "experimental_model_contract"
_WEIGHTS = {
    "efficientnet_v2_s": "EfficientNet_V2_S_Weights.IMAGENET1K_V1",
    "maxvit_t": "MaxVit_T_Weights.IMAGENET1K_V1",
    "p2_s": "EfficientNet_V2_S_Weights.IMAGENET1K_V1",
}


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ExperimentalModelContract:
    model_id: str
    num_classes: int
    adaptation: str = "full"
    head_seed: int = 42
    horizontal_flip_probability: float = 0.0

    def __post_init__(self):
        if self.model_id not in _WEIGHTS:
            raise ValueError("unsupported experimental model identity")
        _positive_int(self.num_classes, "num_classes")
        if self.adaptation not in ("full", "frozen_encoder"):
            raise ValueError("unsupported experimental adaptation")
        if type(self.head_seed) is not int or not 0 <= self.head_seed < 2**63:
            raise ValueError("head_seed must be an integer in [0,2**63)")
        transform = ExperimentalFitPad224(self.horizontal_flip_probability)
        object.__setattr__(self, "horizontal_flip_probability", transform.horizontal_flip_probability)

    def to_config(self) -> dict:
        architecture = {"readout": "gap_biased_linear", "added_dropout": 0.0}
        initialization = {"policy": "xavier_uniform_gain1_zero_bias_v1", "seed": self.head_seed,
                          "order": "pretrained_encoder_then_new_modules"}
        if self.model_id == "p2_s":
            architecture = {
                "readout": "dual_scale_query_plus_pooled_context", "scale": "S",
                "feature_taps": [5, 7], "memory_tokens": [196, 49],
                "token_order": "f16_row_major_then_f32_row_major", "width": 128,
                "heads": 4, "query_blocks": 1, "ffn_ratio": 4,
                "added_dropout": 0.0, "query_self_attention": False,
                "context_coefficient": 1, "query_coefficient": 1,
                "layer_norm_eps": 1e-5, "ffn_activation": "GELU",
                "projection_bias": False, "attention_projection_bias": True,
                "memory_block_normalization": False, "positional_encoding": False,
            }
            initialization |= {
                "layer_norm_weight": 1.0, "layer_norm_bias": 0.0,
                "query_and_scale_distribution": "independent_normal", "embedding_std": 0.02,
                "qkv_initialization": "separate_xavier_uniform_gain1",
                "new_module_order": ["f16_projection", "f32_projection", "f16_token_norm",
                                     "f32_token_norm", "f16_scale_embedding", "f32_scale_embedding",
                                     "ingredient_queries", "blocks_in_forward_order",
                                     "final_query_norm", "classwise_readout", "pooled_context_readout"],
                "block_module_order": ["query_norm", "q_projection", "k_projection", "v_projection",
                                       "attention_output", "ffn_norm", "ffn_input", "ffn_output"],
            }
        return {
            "schema_version": 1, "role": "4a_experiment", "model_id": self.model_id,
            "architecture_version": 1, "architecture": architecture,
            "num_classes": self.num_classes, "output": "ordered_raw_logits",
            "weights_enum": _WEIGHTS[self.model_id], "adaptation": self.adaptation,
            "transform": ExperimentalFitPad224(self.horizontal_flip_probability).to_config(),
            "head_initialization": initialization,
        }

    @classmethod
    def from_config(cls, config: dict) -> ExperimentalModelContract:
        try:
            transform = ExperimentalFitPad224.from_config(config["transform"])
            contract = cls(config["model_id"], config["num_classes"], config["adaptation"],
                           config["head_initialization"]["seed"], transform.horizontal_flip_probability)
            if json.dumps(config, sort_keys=True, allow_nan=False) != json.dumps(contract.to_config(), sort_keys=True):
                raise ValueError("unsupported or inconsistent experimental model contract")
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("invalid experimental model contract") from error
        return contract


def make_experimental_linear(in_features: int, out_features: int, seed: int) -> torch.nn.Linear:
    """Initialize a CPU head with a private generator, leaving global RNG unchanged."""
    _positive_int(in_features, "in_features")
    _positive_int(out_features, "out_features")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2**63)")
    with torch.random.fork_rng(devices=[]):
        layer = torch.nn.Linear(in_features, out_features, bias=True, device="cpu", dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.nn.init.xavier_uniform_(layer.weight, gain=1.0, generator=generator)
    torch.nn.init.zeros_(layer.bias)
    return layer


def construct_experimental_model(constructor, contract: ExperimentalModelContract, *, state_dict=None):
    """Fresh ImageNet initialization or offline strict state restoration via an adapter factory.

    The factory takes ``contract`` and operational ``initialize_pretrained``;
    that flag never replaces the original weight identity in the contract.
    """
    if not isinstance(contract, ExperimentalModelContract):
        raise ValueError("a validated experimental model contract is required")
    if state_dict is not None and (not isinstance(state_dict, Mapping) or not state_dict):
        raise ValueError("restore requires a nonempty complete model state")
    model = constructor(contract=contract, initialize_pretrained=state_dict is None)
    if getattr(model, "experimental_contract", None) != contract:
        raise ValueError("constructed model does not preserve its experimental contract")
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def validate_checkpoint_contract(checkpoint: dict, expected: ExperimentalModelContract):
    """Called only for new experimental models; legacy checkpoint paths remain unchanged."""
    actual = ExperimentalModelContract.from_config(checkpoint.get(CHECKPOINT_CONTRACT_KEY))
    if actual != expected:
        raise ValueError("checkpoint experimental protocol differs from the requested model")


def experimental_smoke_policy() -> dict:
    """Predeclared engineering settings, not a benchmark/HPO configuration."""
    return copy.deepcopy({
        "policy_id": "phase5-engineering-smoke-v1", "seed": 42,
        "requested_batch_size": 128, "physical_probe_candidates": [128, 64, 32, 16, 8],
        "minimum_useful_physical_batch_size": 8, "precision": "32-true",
        "float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False, "memory_allowance_fraction": 0.85,
        "probe_optimizer_updates": 2, "smoke_optimizer_updates": 4,
        "validation_batches": 2, "max_epochs": 1,
        "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 0.0,
        "loss": "BCEWithLogitsLoss", "loss_reduction": "mean", "pos_weight": None,
        "augmentation": "none", "drop_last": False, "num_workers": 2,
        "pin_memory": None, "single_device": True,
    })
