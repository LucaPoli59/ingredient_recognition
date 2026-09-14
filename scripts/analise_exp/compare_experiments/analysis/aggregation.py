"""Aggregate trials within experiments and comparable cohorts across experiments."""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from ..comparability import compare_signatures, comparison_cohort


def _aggregate_curve(trials: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = defaultdict(list)
    for trial in trials:
        if trial.get("curve", {}).get("status") != "available":
            continue
        for point in trial["curve"]["points"]:
            buckets[point["coordinate"]].append((point["value"], trial.get("state") or "UNKNOWN"))
    return {
        "method": "Exact-coordinate aggregation without interpolation.",
        "points": [{
            "coordinate": coordinate,
            "contributors": len(observations),
            "state_counts": {
                state: sum(item_state == state for _, item_state in observations)
                for state in sorted({item_state for _, item_state in observations})
            },
            "mean": statistics.fmean(value for value, _ in observations),
            "median": statistics.median(value for value, _ in observations),
            "min": min(value for value, _ in observations),
            "max": max(value for value, _ in observations),
        } for coordinate, observations in sorted(buckets.items())],
    }


def aggregate_experiment(trials: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    eligible = [
        trial for trial in trials
        if trial.get("objective") is not None and trial.get("state") in (None, "COMPLETE")
    ]
    choose = min if direction == "min" else max
    study_selected = choose(eligible, key=lambda trial: trial["objective"]) if eligible else None
    cohorts = defaultdict(list)
    for trial in eligible:
        cohorts[comparison_cohort(trial["comparability_signature"])].append(trial)
    cohort_summaries = {}
    curve_cohorts = defaultdict(list)
    for trial in trials:
        curve_cohorts[comparison_cohort(trial["comparability_signature"])].append(trial)
    for key, items in cohorts.items():
        cohort_best = choose(items, key=lambda trial: trial["objective"])
        cohort_summaries[key] = {
            "trials": len(items),
            "numbers": [item["number"] for item in items],
            "best_trial": {
                "number": cohort_best["number"],
                "objective": cohort_best["objective"],
                "selected_by": cohort_best.get("objective_source"),
            },
            "objective_mean": statistics.fmean(item["objective"] for item in items),
            "objective_median": statistics.median(item["objective"] for item in items),
            "objective_min": min(item["objective"] for item in items),
            "objective_max": max(item["objective"] for item in items),
        }
    selected_summary = ({
        "number": study_selected["number"],
        "objective": study_selected["objective"],
        "selected_by": study_selected.get("objective_source"),
    } if study_selected else None)
    return {
        "trials": len(trials),
        "states": {
            (state or "UNKNOWN"): sum(trial.get("state") == state for trial in trials)
            for state in sorted({trial.get("state") for trial in trials}, key=str)
        },
        "eligible_trials": len(eligible),
        "study_selected_trial": selected_summary,
        "best_trial": selected_summary if len(cohort_summaries) == 1 else None,
        "best_trial_reason": (
            None if len(cohort_summaries) <= 1
            else "No single comparable best: objective policies form multiple cohorts."
        ),
        "cohorts": cohort_summaries,
        "aggregate_curves": {
            key: _aggregate_curve(items) for key, items in curve_cohorts.items()
        },
    }


def compare_experiments(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    for index, left in enumerate(experiments):
        for right in experiments[index + 1:]:
            shared = sorted(set(left["summary"]["cohorts"]) & set(right["summary"]["cohorts"]))
            cohort_differences = []
            for cohort in shared:
                left_summary = left["summary"]["cohorts"].get(cohort)
                right_summary = right["summary"]["cohorts"].get(cohort)
                if left_summary and right_summary:
                    cohort_differences.append({
                        "cohort": cohort,
                        "left_trials": left_summary["trials"],
                        "right_trials": right_summary["trials"],
                        "median_objective_difference_left_minus_right": (
                            left_summary["objective_median"] - right_summary["objective_median"]
                        ),
                    })
            example_check = None
            if left["trials"] and right["trials"]:
                example_check = compare_signatures(
                    left["trials"][0]["comparability_signature"],
                    right["trials"][0]["comparability_signature"],
                )
            pairs.append({
                "left": left["name"],
                "right": right["name"],
                "shared_objective_cohorts": len(shared),
                "cohort_comparisons": cohort_differences,
                "representative_signature_check": example_check,
            })
    return {"pairs": pairs}
