"""Deterministic per-label metrics and trajectory summaries for Phase 3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from src.ingredient_selection.protocol import SelectorProtocol


PROFILE_CLASSIFIER_VERSION = "phase3-pilot-bootstrap-band-v1"


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    positive = values >= 0
    output = np.empty_like(values)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def per_label_metrics(
        logits: np.ndarray,
        targets: np.ndarray,
        class_names: Sequence[str],
        *,
        run_id: str,
        split: str,
        audit_epoch: int,
        learning_rate: float,
        threshold: float = 0.5,
) -> list[dict[str, Any]]:
    logits = np.asarray(logits)
    targets = np.asarray(targets, dtype=np.uint8)
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("logits and targets must have the same [records, labels] shape")
    if logits.shape[1] != len(class_names):
        raise ValueError("class order length does not match score columns")
    if not np.isfinite(logits).all():
        raise ValueError("audit logits contain non-finite values")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("audit targets must be binary")

    probabilities = sigmoid(logits)
    predictions = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average=None,
        zero_division=0,
    )
    _, _, micro_f1, _ = precision_recall_fscore_support(
        targets.ravel(),
        predictions.ravel(),
        average="binary",
        zero_division=0,
    )
    rows: list[dict[str, Any]] = []
    total = targets.shape[0]
    for index, class_name in enumerate(class_names):
        support = int(targets[:, index].sum())
        ap_valid = 0 < support < total
        ap = float(average_precision_score(targets[:, index], probabilities[:, index])) if ap_valid else np.nan
        rows.append({
            "run_id": run_id,
            "audit_epoch": int(audit_epoch),
            "split": split,
            "class_index": index,
            "class_name": class_name,
            "support": support,
            "records": total,
            "prevalence": support / total,
            "average_precision": ap,
            "ap_valid": ap_valid,
            "precision_at_0_5": float(precision[index]),
            "recall_at_0_5": float(recall[index]),
            "f1_at_0_5": float(f1[index]),
            "micro_f1_at_0_5": float(micro_f1),
            "learning_rate": float(learning_rate),
        })
    return rows


def bootstrap_average_precision(
        scores: np.ndarray,
        targets: np.ndarray,
        *,
        seed: int,
        samples: int = 1000,
        max_draws: int = 10_000,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.uint8)
    if scores.ndim != 1 or targets.ndim != 1 or scores.shape != targets.shape:
        raise ValueError("bootstrap scores and targets must be equally sized vectors")
    if len(np.unique(targets)) < 2:
        return {"valid": False, "valid_draws": 0, "attempted_draws": 0, "lower": np.nan, "upper": np.nan}
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    attempts = 0
    while len(estimates) < samples and attempts < max_draws:
        attempts += 1
        indices = generator.integers(0, len(targets), size=len(targets))
        sampled_targets = targets[indices]
        if sampled_targets.min() == sampled_targets.max():
            continue
        estimates.append(float(average_precision_score(sampled_targets, scores[indices])))
    valid = len(estimates) == samples
    if not valid:
        return {
            "valid": False,
            "valid_draws": len(estimates),
            "attempted_draws": attempts,
            "lower": np.nan,
            "upper": np.nan,
        }
    lower, upper = np.quantile(np.asarray(estimates), [0.025, 0.975])
    return {
        "valid": True,
        "valid_draws": len(estimates),
        "attempted_draws": attempts,
        "lower": float(lower),
        "upper": float(upper),
    }


def final_validation_bootstrap(
        logits: np.ndarray,
        targets: np.ndarray,
        class_names: Sequence[str],
        protocol: SelectorProtocol = SelectorProtocol(),
) -> list[dict[str, Any]]:
    probabilities = sigmoid(logits)
    return [
        {
            "class_index": index,
            "class_name": name,
            **bootstrap_average_precision(
                probabilities[:, index],
                np.asarray(targets)[:, index],
                seed=42_000 + index,
                samples=protocol.bootstrap_samples,
                max_draws=protocol.bootstrap_max_draws,
            ),
        }
        for index, name in enumerate(class_names)
    ]


def _window_values(group: pd.DataFrame, split: str, epochs: Iterable[int]) -> np.ndarray:
    expected = tuple(epochs)
    selected = group[(group["split"] == split) & group["audit_epoch"].isin(expected)]
    if set(selected["audit_epoch"].tolist()) != set(expected):
        return np.asarray([], dtype=np.float64)
    selected = selected.set_index("audit_epoch").loc[list(expected)]
    return selected["average_precision"].to_numpy(dtype=np.float64)


def trajectory_evidence(
        metrics: pd.DataFrame,
        protocol: SelectorProtocol = SelectorProtocol(),
) -> pd.DataFrame:
    required = {
        "run_id", "audit_epoch", "split", "class_index", "class_name",
        "support", "prevalence", "average_precision",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"metrics table is missing columns: {sorted(missing)}")
    if metrics.duplicated(["run_id", "audit_epoch", "split", "class_index"]).any():
        raise ValueError("metrics table contains duplicated run/epoch/split/class rows")

    evidence: list[dict[str, Any]] = []
    for (run_id, class_index), group in metrics.groupby(["run_id", "class_index"], sort=True):
        class_names = group["class_name"].unique()
        if len(class_names) != 1:
            raise ValueError("one class index maps to multiple class names")
        train_initial = group[(group["split"] == "train") & (group["audit_epoch"] == 0)]
        early = _window_values(group, "train", protocol.early_window)
        near = _window_values(group, "train", protocol.near_window)
        late_train = _window_values(group, "train", protocol.late_window)
        late_val = _window_values(group, "val", protocol.late_window)
        complete = (
            len(train_initial) == 1 and early.size and near.size and late_train.size and late_val.size
            and np.isfinite(np.concatenate([early, near, late_train, late_val])).all()
        )
        row: dict[str, Any] = {
            "run_id": run_id,
            "class_index": int(class_index),
            "class_name": class_names[0],
            "trajectory_complete": bool(complete),
            "configuration_sensitivity_available": False,
            "seed_stability_available": False,
        }
        if complete:
            initial = float(train_initial.iloc[0]["average_precision"])
            row.update({
                "train_support": int(group[group["split"] == "train"]["support"].iloc[0]),
                "train_prevalence": float(group[group["split"] == "train"]["prevalence"].iloc[0]),
                "val_support": int(group[group["split"] == "val"]["support"].iloc[0]),
                "val_prevalence": float(group[group["split"] == "val"]["prevalence"].iloc[0]),
                "train_initial_ap": initial,
                "train_initial_to_late_gain": float(np.median(late_train) - initial),
                "train_early_to_late_gain": float(np.median(late_train) - np.median(early)),
                "train_late_median_ap": float(np.median(late_train)),
                "val_late_median_ap": float(np.median(late_val)),
                "train_late_iqr": float(np.quantile(late_train, 0.75) - np.quantile(late_train, 0.25)),
                "val_late_iqr": float(np.quantile(late_val, 0.75) - np.quantile(late_val, 0.25)),
                "train_near_to_late_shift": float(np.median(late_train) - np.median(near)),
                "train_minus_val_late_gap": float(np.median(late_train) - np.median(late_val)),
            })
        evidence.append(row)
    return pd.DataFrame(evidence).sort_values(["run_id", "class_index"]).reset_index(drop=True)


@dataclass(frozen=True)
class ProfileThresholds:
    min_train_support: int
    min_train_gain: float
    min_train_late_ap: float
    min_val_late_ap: float
    max_train_late_iqr: float
    max_val_late_iqr: float
    max_train_val_gap: float
    min_image_advantage: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_profile(row: Mapping[str, Any], gates: ProfileThresholds) -> tuple[str, list[str]]:
    """Apply absolute gates with a conservative final-checkpoint bootstrap band."""
    required = [
        "trajectory_complete", "train_support", "train_initial_to_late_gain",
        "train_late_median_ap", "val_late_median_ap", "train_late_iqr",
        "val_late_iqr", "train_minus_val_late_gap", "image_vs_cuisine_ap_advantage",
        "cuisine_prior_ap", "bootstrap_valid", "bootstrap_ap_lower", "bootstrap_ap_upper",
    ]
    if any(key not in row or pd.isna(row[key]) for key in required) or not row["trajectory_complete"]:
        return "uncertain", ["incomplete_evidence"]
    if not row["bootstrap_valid"]:
        return "uncertain", ["invalid_validation_bootstrap"]
    lower = float(row["bootstrap_ap_lower"])
    upper = float(row["bootstrap_ap_upper"])
    if not 0 <= lower <= upper <= 1:
        return "uncertain", ["invalid_validation_bootstrap"]
    if int(row["train_support"]) < gates.min_train_support:
        return "uncertain", ["low_support"]
    if (
            float(row["train_initial_to_late_gain"]) < gates.min_train_gain
            or float(row["train_late_median_ap"]) < gates.min_train_late_ap
    ):
        return "no_sustained_optimization", ["train_signal_below_gate"]
    val_ap = float(row["val_late_median_ap"])
    if val_ap < gates.min_val_late_ap:
        if upper < gates.min_val_late_ap:
            return "optimization_only", ["validation_signal_below_gate"]
        return "uncertain", ["validation_gate_overlaps_bootstrap_interval"]
    if lower < gates.min_val_late_ap:
        return "uncertain", ["validation_gate_overlaps_bootstrap_interval"]
    if (
            float(row["train_late_iqr"]) > gates.max_train_late_iqr
            or float(row["val_late_iqr"]) > gates.max_val_late_iqr
            or float(row["train_minus_val_late_gap"]) > gates.max_train_val_gap
    ):
        return "uncertain", ["late_window_or_generalization_instability"]
    advantage = float(row["image_vs_cuisine_ap_advantage"])
    cuisine_ap = float(row["cuisine_prior_ap"])
    if advantage < gates.min_image_advantage:
        if upper - cuisine_ap < gates.min_image_advantage:
            return "context_predictable", ["insufficient_image_advantage"]
        return "uncertain", ["image_advantage_gate_overlaps_bootstrap_interval"]
    if lower - cuisine_ap < gates.min_image_advantage:
        return "uncertain", ["image_advantage_gate_overlaps_bootstrap_interval"]
    return "generalizable_candidate", ["all_numeric_gates_passed"]
