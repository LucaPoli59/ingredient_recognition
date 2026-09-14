"""Aggregate approximate W&B tensor-distribution diagnostics."""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


def tensor_role(key: str) -> str:
    if key.startswith("gradients/"):
        return "gradient"
    if key.endswith(".weight"):
        return "weight"
    if key.endswith(".bias"):
        return "bias"
    return "parameter"


def aggregate_distributions(wandb_data: dict[str, Any]) -> dict[str, Any]:
    values = defaultdict(list)
    trajectories = defaultdict(lambda: defaultdict(list))
    sessions = 0
    for trial_number, trial in wandb_data.get("trials", {}).items():
        for session in trial.get("sessions", []):
            if session.get("status") != "available":
                continue
            sessions += 1
            for key, summary in session.get("series", {}).items():
                values[key].append({
                    "trial": int(trial_number),
                    "source": session["source"],
                    "samples": summary["samples"],
                    "rms_relative_change": summary["rms_relative_change"],
                    "mean_change": summary["mean_change"],
                    "normalized_cdf_distance_estimate": summary["normalized_cdf_distance_estimate"],
                })
                for sample_index, sample in enumerate(summary.get("trajectory", [])):
                    clocks = sample["history_record_clocks"]
                    coordinate = clocks.get("_step")
                    if coordinate is None:
                        coordinate = clocks.get("trainer/global_step")
                    if coordinate is None:
                        coordinate = clocks.get("epoch", sample_index)
                    trajectories[key][float(coordinate)].append(sample["rms_midpoint_estimate"])
    tensors = []
    for key, observations in values.items():
        rms_changes = [item["rms_relative_change"] for item in observations if item["rms_relative_change"] is not None]
        distances = [item["normalized_cdf_distance_estimate"] for item in observations]
        tensors.append({
            "key": key,
            "role": tensor_role(key),
            "observations": len(observations),
            "rms_relative_change_median": statistics.median(rms_changes) if rms_changes else None,
            "normalized_cdf_distance_median": statistics.median(distances) if distances else None,
            "sources": observations,
            "rms_trajectory": [{
                "coordinate": coordinate,
                "contributors": len(samples),
                "mean": statistics.fmean(samples),
                "median": statistics.median(samples),
                "min": min(samples),
                "max": max(samples),
            } for coordinate, samples in sorted(trajectories[key].items())],
        })
    tensors.sort(key=lambda item: item["normalized_cdf_distance_median"] or 0.0, reverse=True)
    return {
        "status": "available" if tensors else "unavailable",
        "sessions": sessions,
        "tensors": tensors,
        "largest_distribution_drifts": tensors[:20],
        "weighting": "Each session contributes one first-to-last observation per tensor; duplicate sessions are not averaged silently.",
    }
