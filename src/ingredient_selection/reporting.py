"""Deterministic, read-only scientific figures for a validated full selector profile."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ingredient_selection.analysis import _validate_profile_rule
from src.ingredient_selection.artifacts import read_json, write_json
from src.ingredient_selection.metrics import ProfileThresholds, classify_profile


OUTCOMES = (
    "generalizable_candidate",
    "optimization_only",
    "context_predictable",
    "no_sustained_optimization",
    "uncertain",
)
COLORS = {
    "generalizable_candidate": "#18794e",
    "optimization_only": "#b45f06",
    "context_predictable": "#6f42a1",
    "no_sustained_optimization": "#b42318",
    "uncertain": "#667085",
}
SVG_SETTINGS = {"svg.hashsalt": "ingredient-selection-profile-v1", "svg.fonttype": "none"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_profile(output: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], ProfileThresholds]:
    manifest = read_json(output / "campaign_manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError("full profile report requires a completed campaign")
    rule = read_json(output / "profile_rule.json")
    gates = _validate_profile_rule(rule, manifest, read_json(output / "pilot_cohort.json"), output)
    validation = read_json(output / "validation_summary.json")
    if (validation.get("analysis_scope") != "full"
            or validation.get("profile_rule_applied") is not True
            or validation.get("test_split_accessed") is not False
            or validation.get("campaign_identity_hash") != manifest["campaign_identity_hash"]):
        raise ValueError("a validated full, test-isolated analysis is required")

    evidence_path = output / "profile_evidence.csv"
    if validation.get("profile_evidence_sha256") != _sha256(evidence_path):
        raise ValueError("full profile evidence changed after validation")
    evidence = pd.read_csv(evidence_path)
    names = manifest["campaign_identity"]["class_order"]
    ordered = evidence.sort_values("class_index")
    if (len(ordered) != len(names)
            or ordered["class_index"].tolist() != list(range(len(names)))
            or ordered["class_name"].tolist() != names
            or set(ordered["provisional_outcome"]) - set(OUTCOMES)):
        raise ValueError("full profile class order or outcome is invalid")
    expected = [classify_profile(row, gates) for row in ordered.to_dict("records")]
    if (ordered["provisional_outcome"].tolist() != [item[0] for item in expected]
            or ordered["profile_reasons"].tolist() != [";".join(item[1]) for item in expected]):
        raise ValueError("full profile decisions differ from the frozen rule")

    pilot_summary = read_json(output / "pilot_profile_summary.json")
    if (pilot_summary.get("analysis_scope") != "pilot_only"
            or pilot_summary.get("profile_rule_hash") != rule["artifact_hash"]
            or pilot_summary.get("pilot_decisions_sha256") != _sha256(output / "pilot_profile_decisions.csv")):
        raise ValueError("frozen pilot decisions are missing or changed")
    pilot_decisions = pd.read_csv(output / "pilot_profile_decisions.csv")
    pilot_rows = ordered.merge(
        pilot_decisions[["class_index", "class_name", "provisional_outcome"]],
        on=["class_index", "class_name"], suffixes=("_full", "_pilot"), validate="one_to_one",
    )
    if len(pilot_rows) != 24 or not pilot_rows["provisional_outcome_full"].equals(
            pilot_rows["provisional_outcome_pilot"]):
        raise ValueError("full profile changed a frozen pilot decision")

    metrics_path = output / "metrics_per_label_epoch.csv"
    metrics = pd.read_csv(metrics_path)
    epochs = manifest["campaign_identity"]["execution"]["audit_epochs"]
    expected_rows = len(names) * len(epochs) * 2
    keys = metrics[["split", "audit_epoch", "class_index"]]
    if (len(metrics) != expected_rows or keys.duplicated().any()
            or set(metrics["split"]) != {"train", "val"}
            or set(metrics["audit_epoch"]) != set(epochs)
            or set(metrics["class_index"]) != set(range(len(names)))
            or not metrics["ap_valid"].all()
            or not metrics.apply(lambda row: row["class_name"] == names[int(row["class_index"])], axis=1).all()):
        raise ValueError("trajectory metrics do not match the completed campaign")
    return ordered.reset_index(drop=True), metrics, rule, gates


def _examples(evidence: pd.DataFrame) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        group = evidence[evidence["provisional_outcome"] == outcome].copy()
        if group.empty:
            continue
        median = float(group["val_late_median_ap"].median())
        group["distance_to_group_median"] = (group["val_late_median_ap"] - median).abs()
        for row in group.sort_values(["distance_to_group_median", "class_index"]).head(2).itertuples():
            examples.append({
                "class_index": int(row.class_index),
                "class_name": row.class_name,
                "provisional_outcome": outcome,
            })
    return examples


def _save_figure(fig: Any, path: Path) -> None:
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(path.with_suffix(".png"), format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _decision_map(evidence: pd.DataFrame, gates: ProfileThresholds, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for outcome in OUTCOMES:
        group = evidence[evidence["provisional_outcome"] == outcome]
        if group.empty:
            continue
        ax.scatter(group["val_late_median_ap"], group["image_vs_cuisine_ap_advantage"],
                   s=25 + 11 * np.log10(group["train_support"].clip(lower=1)),
                   color=COLORS[outcome], alpha=0.78, edgecolor="white", linewidth=0.3,
                   label=f"{outcome} ({len(group)})")
    ax.axvline(gates.min_val_late_ap, color="#344054", linestyle="--", linewidth=1)
    ax.axhline(gates.min_image_advantage, color="#344054", linestyle=":", linewidth=1)
    ax.set(xlabel="Late-window validation AP (median, epochs 32–40)",
           ylabel="Validation AP minus train-only cuisine-prior AP",
           title="Frozen numerical profile: held-out AP and non-visual control")
    ax.grid(alpha=0.15)
    ax.legend(loc="best", fontsize=8)
    fig.text(0.02, 0.01, "v5 • seed 42 • single run • colors are provisional profile outcomes; size reflects train support", fontsize=8)
    _save_figure(fig, path)


def _support_map(evidence: pd.DataFrame, gates: ProfileThresholds, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for outcome in OUTCOMES:
        group = evidence[evidence["provisional_outcome"] == outcome]
        if not group.empty:
            ax.scatter(group["train_support"], group["val_late_median_ap"],
                       s=35, color=COLORS[outcome], alpha=0.8, label=f"{outcome} ({len(group)})")
    ax.set_xscale("log")
    ax.axvline(gates.min_train_support, color="#344054", linestyle=":", linewidth=1)
    ax.axhline(gates.min_val_late_ap, color="#344054", linestyle="--", linewidth=1)
    ax.set(xlabel="Final train positive support (log scale)",
           ylabel="Late-window validation AP (median, epochs 32–40)",
           title="Support dependence of held-out ranking quality")
    ax.grid(alpha=0.15)
    ax.legend(loc="best", fontsize=8)
    fig.text(0.02, 0.01, "v5 • seed 42 • one selector configuration; support association is descriptive, not causal", fontsize=8)
    _save_figure(fig, path)


def _trajectories(metrics: pd.DataFrame, examples: list[dict[str, Any]], path: Path) -> None:
    columns = 2
    rows = math.ceil(len(examples) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.1 * rows), squeeze=False, sharex=True, sharey=True)
    for ax, item in zip(axes.flat, examples):
        subset = metrics[metrics["class_index"] == item["class_index"]]
        for split, color in (("train", "#1d4ed8"), ("val", "#e67e22")):
            line = subset[subset["split"] == split].sort_values("audit_epoch")
            ax.plot(line["audit_epoch"], line["average_precision"], marker=".",
                    linewidth=1.4, markersize=3, color=color, label=f"{split} AP")
        ax.axvspan(32, 40, color="#98a2b3", alpha=0.12)
        ax.set_title(f"{item['class_name']} — {item['provisional_outcome']}", fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.13)
        ax.legend(fontsize=7, loc="best")
    for ax in list(axes.flat)[len(examples):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Completed training epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Average precision")
    fig.suptitle("Fixed-state train/validation AP trajectories: median-by-AP examples", fontsize=13)
    fig.text(0.02, 0.005, "Examples are selected deterministically for visualization only; shaded area is the late window. v5 • seed 42.", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    _save_figure(fig, path)


def generate_campaign_report(output_dir: str | Path) -> dict[str, Any]:
    """Report the unchanged P3 rule on all labels; never select a final vocabulary."""
    output = Path(output_dir).resolve()
    evidence, metrics, rule, gates = _validated_profile(output)
    examples = _examples(evidence)
    figures_dir = output / "p4_figures"
    figures_dir.mkdir(exist_ok=True)
    with matplotlib.rc_context(SVG_SETTINGS):
        _decision_map(evidence, gates, figures_dir / "decision_map.svg")
        _support_map(evidence, gates, figures_dir / "support_vs_validation_ap.svg")
        _trajectories(metrics, examples, figures_dir / "ap_trajectory_examples.svg")
    figure_names = ("decision_map", "support_vs_validation_ap", "ap_trajectory_examples")
    figure_hashes = {
        f"{name}.{extension}": _sha256(figures_dir / f"{name}.{extension}")
        for name in figure_names for extension in ("svg", "png")
    }
    groups = {
        name: evidence.loc[evidence["provisional_outcome"] == name, "class_name"].tolist()
        for name in OUTCOMES
    }
    report = {
        "schema_version": 1,
        "protocol_id": rule["protocol_id"],
        "analysis_scope": "full_profile_only",
        "campaign_identity_hash": rule["campaign_identity_hash"],
        "profile_rule_hash": rule["artifact_hash"],
        "reporting_source_sha256": _sha256(Path(__file__)),
        "profile_evidence_sha256": _sha256(output / "profile_evidence.csv"),
        "trajectory_metrics_sha256": _sha256(output / "metrics_per_label_epoch.csv"),
        "pilot_decisions_sha256": _sha256(output / "pilot_profile_decisions.csv"),
        "visible_label_count": len(evidence),
        "outcome_counts": {name: len(groups[name]) for name in OUTCOMES},
        "provisional_groups": groups,
        "reason_counts": {
            name: int(count) for name, count in evidence["profile_reasons"].value_counts().sort_index().items()
        },
        "trajectory_examples": examples,
        "figure_sha256": figure_hashes,
        "test_split_accessed": False,
        "configuration_sensitivity_available": False,
        "seed_stability_available": False,
        "final_vocabulary_selected": False,
    }
    write_json(output / "p4_profile_report.json", report)
    return report
