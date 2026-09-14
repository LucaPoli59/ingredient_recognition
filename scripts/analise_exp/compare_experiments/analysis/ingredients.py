"""Summarize optional indexed per-ingredient metric streams."""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


def logging_status(config: dict[str, Any], ingredient_series: dict[str, Any]) -> str:
    hp = config.get("hyper_parameters", {})
    if "log_per_ingredient_metrics" in hp:
        if not hp["log_per_ingredient_metrics"]:
            return "disabled"
        return "available" if ingredient_series else "enabled_but_missing"
    return "available_legacy" if ingredient_series else "unknown_legacy"


def analyze_ingredients(trials: list[dict[str, Any]], labels: list[str] | None) -> dict[str, Any]:
    values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    trajectories: dict[tuple[str, str, int, float], list[float]] = defaultdict(list)
    coverage = defaultdict(int)
    for trial in trials:
        seen = set()
        for key, observations in trial.get("ingredient_series", {}).items():
            split, metric, index = key.split("/")
            if observations:
                label_index = int(index)
                values[(split, metric, label_index)].append(observations[-1]["value"])
                for observation_index, observation in enumerate(observations):
                    coordinate = observation.get("epoch")
                    if coordinate is None:
                        coordinate = observation.get("step", observation_index)
                    trajectories[(split, metric, label_index, float(coordinate))].append(
                        observation["value"]
                    )
                seen.add((split, metric))
        for item in seen:
            coverage[item] += 1
    summaries = []
    for (split, metric, index), samples in sorted(values.items()):
        summaries.append({
            "split": split,
            "metric": metric,
            "label_index": index,
            "label": labels[index] if labels is not None and index < len(labels) else None,
            "trials": len(samples),
            "last_value_mean": statistics.fmean(samples),
            "last_value_median": statistics.median(samples),
            "last_value_min": min(samples),
            "last_value_max": max(samples),
            "trajectory": [{
                "coordinate": coordinate,
                "contributors": len(trajectory_values),
                "mean": statistics.fmean(trajectory_values),
                "median": statistics.median(trajectory_values),
            } for (trajectory_split, trajectory_metric, trajectory_index, coordinate), trajectory_values
              in sorted(trajectories.items())
              if (trajectory_split, trajectory_metric, trajectory_index) == (split, metric, index)],
        })
    return {
        "status": "available" if summaries else "unavailable",
        "label_mapping": "available" if labels is not None else "unavailable",
        "series": summaries,
        "coverage_by_split_metric": {
            f"{split}/{metric}": count for (split, metric), count in sorted(coverage.items())
        },
    }
