"""Build explicit comparability signatures and pairwise checks."""
from __future__ import annotations

from typing import Any

from .normalization import config_section, label_contract
from .readers.config import class_name


def trial_signature(config: dict[str, Any]) -> dict[str, Any]:
    hp = config_section(config, "hyper_parameters")
    dm = config_section(config, "datamodule_hyper_parameters")
    trainer = config_section(config, "trainer_hyper_parameters")
    torch_model = hp.get("torch_model", {}) if isinstance(hp.get("torch_model"), dict) else {}
    labels = label_contract(config)
    return {
        "metadata_filename": dm.get("metadata_filename"),
        "feature_label": dm.get("feature_label"),
        "category": dm.get("category"),
        "label_count": labels["count"],
        "label_order_sha256": labels["sha256"],
        "loss_fn": class_name(hp.get("loss_fn")),
        "weighted_loss": hp.get("weighted_loss", hp.get("weight_loss")),
        "model_type": class_name(torch_model.get("type")),
        "max_epochs": trainer.get("max_epochs"),
        "metric_contract": "legacy_batch-derived aggregate scalars",
    }


def comparison_cohort(signature: dict[str, Any]) -> str:
    return "|".join(str(signature.get(key)) for key in (
        "metadata_filename", "feature_label", "label_order_sha256", "loss_fn", "weighted_loss"
    ))


def compare_signatures(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    required = ("metadata_filename", "feature_label", "label_count", "label_order_sha256")
    objective = ("loss_fn", "weighted_loss", "metric_contract")
    required_mismatches = [key for key in required if left.get(key) != right.get(key)]
    objective_mismatches = [key for key in objective if left.get(key) != right.get(key)]
    context_differences = [key for key in ("model_type", "max_epochs", "category") if left.get(key) != right.get(key)]
    status = "compatible"
    if required_mismatches:
        status = "incompatible"
    elif objective_mismatches:
        status = "separate_cohort"
    return {
        "status": status,
        "required_mismatches": required_mismatches,
        "objective_mismatches": objective_mismatches,
        "context_differences": context_differences,
    }
