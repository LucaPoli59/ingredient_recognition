"""Reusable, blind-gated analysis of a completed Phase 3 selector campaign."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.ingredient_selection.artifacts import read_json, write_json
from src.ingredient_selection.data import SelectorDataBundle
from src.ingredient_selection.metrics import (
    ProfileThresholds,
    classify_profile,
    sigmoid,
    trajectory_evidence,
)
from src.ingredient_selection.protocol import (
    SelectorProtocol,
    ordered_values_hash,
    sha256_json,
    validate_pilot_cohort,
)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_rank = pd.Series(left, dtype=float).rank(method="average").to_numpy()
    right_rank = pd.Series(right, dtype=float).rank(method="average").to_numpy()
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return np.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _control_metrics(
        bundle: SelectorDataBundle,
        validation_logits: np.ndarray,
) -> pd.DataFrame:
    train_targets = bundle.train.targets
    val_targets = bundle.val.targets
    train_prevalence = train_targets.mean(axis=0)
    train_cuisines = np.asarray(bundle.train.cuisines)
    val_cuisines = np.asarray(bundle.val.cuisines)
    cuisine_counts: dict[str, tuple[int, np.ndarray]] = {}
    for cuisine in sorted(set(train_cuisines)):
        mask = train_cuisines == cuisine
        cuisine_counts[cuisine] = (int(mask.sum()), train_targets[mask].sum(axis=0))
    cuisine_scores = np.empty_like(val_targets, dtype=np.float64)
    for row, cuisine in enumerate(val_cuisines):
        if cuisine in cuisine_counts:
            total, positives = cuisine_counts[cuisine]
            cuisine_scores[row] = (positives + 1.0) / (total + 2.0)
        else:
            cuisine_scores[row] = train_prevalence

    image_scores = sigmoid(validation_logits)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(bundle.class_names):
        target = val_targets[:, index]
        valid = 0 < target.sum() < len(target)
        rows.append({
            "class_index": index,
            "class_name": name,
            "prevalence_baseline_ap": (
                float(average_precision_score(target, np.full(len(target), train_prevalence[index])))
                if valid else np.nan
            ),
            "cuisine_prior_ap": (
                float(average_precision_score(target, cuisine_scores[:, index])) if valid else np.nan
            ),
            "final_image_ap": (
                float(average_precision_score(target, image_scores[:, index])) if valid else np.nan
            ),
        })
    return pd.DataFrame(rows)


def _validate_profile_rule(
        rule: dict[str, Any], manifest: dict[str, Any], pilot: dict[str, Any]) -> ProfileThresholds:
    payload = dict(rule)
    artifact_hash = payload.pop("artifact_hash", None)
    if artifact_hash != sha256_json(payload):
        raise ValueError("profile rule artifact hash is invalid")
    if rule.get("protocol_id") != manifest["campaign_identity"]["protocol_id"]:
        raise ValueError("profile rule protocol does not match the campaign")
    if rule.get("campaign_identity_hash") != manifest["campaign_identity_hash"]:
        raise ValueError("profile rule campaign identity hash does not match")
    if rule.get("pilot_artifact_hash") != pilot["artifact_hash"]:
        raise ValueError("profile rule was not frozen from this pilot cohort")
    return ProfileThresholds(**rule["gates"])


def analyze_campaign(
        output_dir: str | Path,
        bundle: SelectorDataBundle,
        protocol: SelectorProtocol = SelectorProtocol(),
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    manifest = read_json(output / "campaign_manifest.json")
    pilot = read_json(output / "pilot_cohort.json")
    identity = manifest["campaign_identity"]
    if identity["protocol_id"] != protocol.protocol_id:
        raise ValueError("campaign protocol ID is not Phase 3-D1")
    if manifest["campaign_identity_hash"] != sha256_json(identity):
        raise ValueError("campaign identity hash is invalid")
    if identity.get("seed") != protocol.seed:
        raise ValueError("campaign seed is missing or differs from Phase 3-D1")
    expected_contract = {
        "model": {"type": "EfficientNetV2SSelector", "weights": "EfficientNet_V2_S_Weights.IMAGENET1K_V1", "input_size": [384, 384], "full_backbone_trainable": True},
        "loss": {"type": "BCEWithLogitsLoss", "reduction": "mean", "pos_weight_formula": "(N_train-P_c)/P_c"},
        "optimizer": {"type": "AdamW", "lr": protocol.learning_rate, "betas": list(protocol.adam_betas), "eps": protocol.adam_eps, "weight_decay": protocol.weight_decay, "amsgrad": False, "foreach": False, "fused": False, "parameter_groups": 1, "gradient_clipping": None},
        "scheduler": {"type": "SequentialLR", "milestones": [2], "interval": "epoch"},
        "execution": {"max_epochs": protocol.max_epochs, "batch_size": protocol.batch_size, "drop_last": False, "gradient_accumulation": 1, "precision": "32-true", "early_stopping": False, "swa": False, "audit_epochs": list(protocol.audit_epochs)},
    }
    for section, expected_fields in expected_contract.items():
        actual = identity.get(section)
        if not isinstance(actual, dict) or any(actual.get(key) != value for key, value in expected_fields.items()):
            raise ValueError(f"campaign {section} fields contradict Phase 3-D1")
    if identity["class_order_hash"] != ordered_values_hash(bundle.class_names):
        raise ValueError("campaign class order differs from current train metadata")
    if identity["metadata_sha256"] != {
        "train": bundle.train.metadata_sha256,
        "val": bundle.val.metadata_sha256,
    }:
        raise ValueError("campaign metadata hashes differ from the supplied train/validation data")
    validate_pilot_cohort(pilot, bundle.class_names, bundle.train.supports)

    metrics = pd.read_csv(output / "metrics_per_label_epoch.csv")
    if set(metrics["run_id"]) != {protocol.protocol_id}:
        raise ValueError("campaign metrics contain a missing or foreign run identity")
    observed_classes = metrics[["class_index", "class_name"]].drop_duplicates().sort_values("class_index")
    if (observed_classes["class_index"].tolist() != list(range(protocol.num_classes))
            or observed_classes["class_name"].tolist() != list(bundle.class_names)):
        raise ValueError("campaign metric class order differs from the frozen encoder")
    if "test" in set(metrics["split"]):
        raise ValueError("selector metrics must not contain a test split")
    if set(metrics["audit_epoch"]) != set(protocol.audit_epochs):
        raise ValueError("campaign metrics do not contain the complete frozen audit cadence")
    expected_pairs = {(split, epoch) for split in ("train", "val") for epoch in protocol.audit_epochs}
    observed_pairs = set(zip(metrics["split"], metrics["audit_epoch"]))
    if observed_pairs != expected_pairs:
        raise ValueError("campaign metrics have missing or unexpected split/epoch combinations")

    score_path = output / "audit_scores" / f"validation_epoch_{protocol.max_epochs:02d}.npz"
    with np.load(score_path) as scores:
        record_ids = scores["record_ids"].astype(str).tolist()
        targets = scores["targets"]
        logits = scores["logits"]
    if record_ids != list(bundle.val.record_ids):
        raise ValueError("final validation score archive has a different record order")
    if not np.array_equal(targets, bundle.val.targets):
        raise ValueError("final validation score archive targets differ from metadata")

    evidence = trajectory_evidence(metrics, protocol)
    controls = _control_metrics(bundle, logits)
    evidence = evidence.merge(controls, on=["class_index", "class_name"], validate="one_to_one")
    evidence["image_vs_cuisine_ap_advantage"] = (
        evidence["val_late_median_ap"] - evidence["cuisine_prior_ap"]
    )
    bootstrap_path = output / "final_validation_bootstrap.csv"
    bootstrap = pd.read_csv(bootstrap_path).rename(columns={
        "valid": "bootstrap_valid",
        "valid_draws": "bootstrap_valid_draws",
        "attempted_draws": "bootstrap_attempted_draws",
        "lower": "bootstrap_ap_lower",
        "upper": "bootstrap_ap_upper",
    })
    evidence = evidence.merge(bootstrap, on=["class_index", "class_name"], validate="one_to_one")

    rule_path = output / "profile_rule.json"
    rule_applied = rule_path.is_file()
    if rule_applied:
        gates = _validate_profile_rule(read_json(rule_path), manifest, pilot)
        outcomes = [classify_profile(row, gates) for row in evidence.to_dict("records")]
        evidence["provisional_outcome"] = [item[0] for item in outcomes]
        evidence["profile_reasons"] = [";".join(item[1]) for item in outcomes]
        visible = evidence
        scope = "full"
    else:
        pilot_names = {item["class_name"] for item in pilot["labels"]}
        visible = evidence[evidence["class_name"].isin(pilot_names)].copy()
        visible["provisional_outcome"] = "pilot_unclassified"
        visible["profile_reasons"] = "P3_profile_rule_not_frozen"
        scope = "pilot_only"

    visible = visible.sort_values("class_index").reset_index(drop=True)
    encoded = visible.to_csv(index=False, lineterminator="\n").encode("utf-8")
    (output / "profile_evidence.csv").write_bytes(encoded)
    correlations = {
        "support_vs_train_gain": _spearman(evidence["train_support"], evidence["train_initial_to_late_gain"]),
        "support_vs_val_late_ap": _spearman(evidence["train_support"], evidence["val_late_median_ap"]),
        "prevalence_vs_val_late_ap": _spearman(evidence["train_prevalence"], evidence["val_late_median_ap"]),
    }
    correlations = {name: (value if np.isfinite(value) else None) for name, value in correlations.items()}
    summary = {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "campaign_identity_hash": manifest["campaign_identity_hash"],
        "analysis_scope": scope,
        "profile_rule_applied": rule_applied,
        "visible_label_count": len(visible),
        "total_label_count": len(evidence),
        "class_order_valid": True,
        "metadata_hashes_valid": True,
        "pilot_rule_valid": True,
        "test_split_accessed": False,
        "audit_cadence_valid": True,
        "score_record_order_valid": True,
        "resource_gate_passed": bool(manifest.get("resource_gate", {}).get("measurement", {}).get("passed")),
        "configuration_sensitivity_available": False,
        "seed_stability_available": False,
        "support_correlations": correlations,
        "profile_evidence_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    write_json(output / "validation_summary.json", summary)
    return summary
