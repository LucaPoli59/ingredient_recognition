"""Read-only sensitivity audit of the frozen Phase 3 v3 profile (stdlib only)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


TEXT_FIELDS = {"run_id", "class_name", "provisional_outcome", "profile_reasons"}
BOOL_FIELDS = {"trajectory_complete", "configuration_sensitivity_available",
               "seed_stability_available", "bootstrap_valid"}
EXPECTED_COUNTS = {"generalizable_candidate": 25, "optimization_only": 40,
                   "context_predictable": 13, "no_sustained_optimization": 4,
                   "uncertain": 83}
EXPECTED_REASONS = {"all_numeric_gates_passed": 25, "validation_signal_below_gate": 40,
                    "insufficient_image_advantage": 13, "train_signal_below_gate": 4,
                    "low_support": 13, "validation_gate_overlaps_bootstrap_interval": 50,
                    "image_advantage_gate_overlaps_bootstrap_interval": 17,
                    "late_window_or_generalization_instability": 3}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if key in BOOL_FIELDS:
                if value not in {"True", "False"}:
                    raise ValueError(f"Invalid boolean {key}: {value}")
                row[key] = value == "True"
            elif key not in TEXT_FIELDS:
                row[key] = float(value)
    return rows


def gate_failures(row, gates):
    return {
        "support": row["train_support"] < gates["min_train_support"],
        "train_gain": row["train_initial_to_late_gain"] < gates["min_train_gain"],
        "train_ap": row["train_late_median_ap"] < gates["min_train_late_ap"],
        "val_ap": row["val_late_median_ap"] < gates["min_val_late_ap"],
        "val_ci": row["bootstrap_ap_lower"] < gates["min_val_late_ap"],
        "train_iqr": row["train_late_iqr"] > gates["max_train_late_iqr"],
        "val_iqr": row["val_late_iqr"] > gates["max_val_late_iqr"],
        "gap": row["train_minus_val_late_gap"] > gates["max_train_val_gap"],
        "advantage": row["image_vs_cuisine_ap_advantage"] < gates["min_image_advantage"],
        "advantage_ci": row["bootstrap_ap_lower"] - row["cuisine_prior_ap"] < gates["min_image_advantage"],
    }


def classify(row, gates):
    """Independently replay the valid-evidence D4 branch ordering."""
    flags = gate_failures(row, gates)
    if flags["support"]:
        return "uncertain", "low_support"
    if flags["train_gain"] or flags["train_ap"]:
        return "no_sustained_optimization", "train_signal_below_gate"
    if flags["val_ap"] and row["bootstrap_ap_upper"] < gates["min_val_late_ap"]:
        return "optimization_only", "validation_signal_below_gate"
    if flags["val_ap"] or flags["val_ci"]:
        return "uncertain", "validation_gate_overlaps_bootstrap_interval"
    if flags["train_iqr"] or flags["val_iqr"] or flags["gap"]:
        return "uncertain", "late_window_or_generalization_instability"
    if (flags["advantage"] and
            row["bootstrap_ap_upper"] - row["cuisine_prior_ap"] < gates["min_image_advantage"]):
        return "context_predictable", "insufficient_image_advantage"
    if flags["advantage"] or flags["advantage_ci"]:
        return "uncertain", "image_advantage_gate_overlaps_bootstrap_interval"
    return "generalizable_candidate", "all_numeric_gates_passed"


def audit(campaign):
    rule = json.loads((campaign / "profile_rule.json").read_text())
    manifest = json.loads((campaign / "campaign_manifest.json").read_text())
    payload = {key: value for key, value in rule.items() if key != "artifact_hash"}
    assert canonical_hash(payload) == rule["artifact_hash"]
    identity = manifest["campaign_identity"]
    assert canonical_hash(identity) == manifest["campaign_identity_hash"] == rule["campaign_identity_hash"]
    assert manifest["status"] == "completed" and rule["protocol_id"] == "phase3-d1-v3"
    rows = load_rows(campaign / "profile_evidence.csv")
    assert len(rows) == 165
    assert [row["class_index"] for row in rows] == list(range(165))
    assert [row["class_name"] for row in rows] == identity["class_order"]
    assert all(row["trajectory_complete"] and row["bootstrap_valid"] and
               0 <= row["bootstrap_ap_lower"] <= row["bootstrap_ap_upper"] <= 1 for row in rows)
    gates = rule["gates"]
    assert all(classify(row, gates) == (row["provisional_outcome"], row["profile_reasons"]) for row in rows)
    assert Counter(row["provisional_outcome"] for row in rows) == EXPECTED_COUNTS
    assert Counter(row["profile_reasons"] for row in rows) == EXPECTED_REASONS

    metric_path = campaign / "metrics_per_label_epoch.csv"
    with metric_path.open(newline="") as stream:
        metric_rows = list(csv.DictReader(stream))
    keys = [(row["split"], int(row["audit_epoch"]), int(row["class_index"])) for row in metric_rows]
    expected = {(split, epoch, index) for split in ("train", "val")
                for epoch in range(0, 41, 2) for index in range(165)}
    assert len(keys) == len(set(keys)) == len(expected) and set(keys) == expected
    assert all(row["class_name"] == identity["class_order"][int(row["class_index"])] for row in metric_rows)

    def accepts(row, omit=()):
        return all(not failed for key, failed in gate_failures(row, gates).items() if key not in omit)

    def heldout(row, threshold=.20, support=500):
        return (row["train_support"] >= support and row["val_late_median_ap"] >= threshold
                and row["bootstrap_ap_lower"] >= threshold)

    def summarize(group):
        return {"count": len(group), "names": [row["class_name"] for row in group]}

    policies = [
        ("unchanged_d4", lambda row: accepts(row)),
        ("remove_cuisine_veto", lambda row: accepts(row, ("advantage", "advantage_ci"))),
        ("also_remove_gap", lambda row: accepts(row, ("advantage", "advantage_ci", "gap"))),
        ("heldout_only_support500", lambda row: heldout(row)),
        ("heldout_only_no_support_veto", lambda row: heldout(row, support=0)),
    ]
    nested = {}
    previous = set()
    for name, predicate in policies:
        group = [row for row in rows if predicate(row)]
        selected = {row["class_name"] for row in group}
        assert previous <= selected
        nested[name] = {**summarize(group), "added_from_previous": sorted(selected - previous)}
        previous = selected

    measures = ["train_support", "val_support", "val_prevalence", "train_initial_to_late_gain",
                "train_late_median_ap", "val_late_median_ap", "train_late_iqr", "val_late_iqr",
                "train_minus_val_late_gap", "cuisine_prior_ap", "image_vs_cuisine_ap_advantage"]
    details = lambda group: [{key: row[key] for key in ["class_name", "provisional_outcome"] + measures}
                             for row in group]
    output = {
        "scope": "post-P4 diagnostic sensitivity; no final projection, no source-artifact writes, no test access",
        "campaign_identity_hash": manifest["campaign_identity_hash"],
        "profile_rule_artifact_hash": rule["artifact_hash"],
        "input_sha256": {name: file_hash(campaign / name) for name in ["profile_rule.json", "profile_evidence.csv",
                          "p4_profile_report.json", "campaign_manifest.json", "metrics_per_label_epoch.csv"]},
        "original_outcome_counts": EXPECTED_COUNTS, "original_primary_reasons": EXPECTED_REASONS,
        "independent_failure_counts": {key: sum(gate_failures(row, gates)[key] for row in rows)
                                        for key in gate_failures(rows[0], gates)},
        "nested_counterfactuals": nested,
        "heldout_floor_sensitivity_support500": {str(t): summarize([r for r in rows if heldout(r, t)])
                                                  for t in [.15, .20, .25]},
        "heldout_floor_sensitivity_no_support_veto": {str(t): summarize([r for r in rows if heldout(r, t, support=0)])
                                                      for t in [.15, .20, .25]},
        "point_only_heldout_support500": summarize([r for r in rows if r["train_support"] >= 500
                                                    and r["val_late_median_ap"] >= .20]),
        "uncertain_overlap_point_passes": {
            "validation": sum(r["profile_reasons"] == "validation_gate_overlaps_bootstrap_interval"
                              and r["val_late_median_ap"] >= .20 for r in rows),
            "advantage": sum(r["profile_reasons"] == "image_advantage_gate_overlaps_bootstrap_interval"
                             and r["image_vs_cuisine_ap_advantage"] >= .10 for r in rows)},
        "ranges": {key: {"min": min(r[key] for r in rows), "median": statistics.median(r[key] for r in rows),
                           "max": max(r[key] for r in rows)} for key in measures},
        "final_minus_late_ap": {
            "median_absolute": statistics.median(abs(r["final_image_ap"] - r["val_late_median_ap"]) for r in rows),
            "max_absolute": max(abs(r["final_image_ap"] - r["val_late_median_ap"]) for r in rows),
            "point_020_disagreements": [r["class_name"] for r in rows
                if (r["final_image_ap"] >= .20) != (r["val_late_median_ap"] >= .20)]},
        "baseline_diagnostics": {
            "late_point_above_val_prevalence": sum(r["val_late_median_ap"] > r["val_prevalence"] for r in rows),
            "final_ap_lower_above_val_prevalence_point": sum(r["bootstrap_ap_lower"] > r["val_prevalence"] for r in rows),
            "minimum_final_ap_lower_minus_val_prevalence_point": min(r["bootstrap_ap_lower"] - r["val_prevalence"] for r in rows),
            "normalized_excess_ap_warning": "Illustrative only: (late_AP - val_prevalence)/(1-val_prevalence); no uncertainty adjustment or adopted threshold",
            "normalized_excess_ap_counts_support500": {str(t): sum(r["train_support"] >= 500 and
                (r["val_late_median_ap"] - r["val_prevalence"]) / (1 - r["val_prevalence"]) >= t for r in rows)
                for t in [.10, .15, .20]},
            "within_heldout020_normalized_excess_ap_counts": {str(t): sum(heldout(r) and
                (r["val_late_median_ap"] - r["val_prevalence"]) / (1 - r["val_prevalence"]) >= t for r in rows)
                for t in [.10, .15, .20]}},
        "gap_only_primary_exclusions": details([r for r in rows if r["profile_reasons"] == "late_window_or_generalization_instability"]),
        "low_support_labels": details([r for r in rows if r["train_support"] < 500]),
        "nearest_support_above500": details(sorted([r for r in rows if r["train_support"] >= 500],
                                                    key=lambda r: r["train_support"])[:5]),
    }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.campaign_dir.resolve()), indent=2, sort_keys=True, allow_nan=False))
