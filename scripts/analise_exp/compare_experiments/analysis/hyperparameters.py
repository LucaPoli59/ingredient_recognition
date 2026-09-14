"""Descriptive conditional hyperparameter associations."""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    mean_left, mean_right = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    denominator = (
        sum((x - mean_left) ** 2 for x in left) * sum((y - mean_right) ** 2 for y in right)
    ) ** 0.5
    return numerator / denominator if denominator else None


def analyze_hyperparameters(trials: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        trial for trial in trials
        if trial.get("objective") is not None and trial.get("state") in (None, "COMPLETE")
    ]
    keys = sorted({key for row in rows for key in row.get("hyperparameters", {})})
    result = {}
    for key in keys:
        pairs = [
            (row["hyperparameters"][key], float(row["objective"]))
            for row in rows if key in row.get("hyperparameters", {})
            and row["hyperparameters"][key] is not None
        ]
        if len(pairs) < 2 or len({repr(value) for value, _ in pairs}) < 2:
            continue
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value, _ in pairs):
            values = [float(value) for value, _ in pairs]
            objectives = [objective for _, objective in pairs]
            result[key] = {
                "kind": "numeric",
                "observations": len(pairs),
                "spearman": _pearson(_ranks(values), _ranks(objectives)),
                "interpretation": "Descriptive association; HPO samples are not causal replicates.",
            }
        else:
            groups = defaultdict(list)
            for value, objective in pairs:
                groups[str(value)].append(objective)
            result[key] = {
                "kind": "categorical",
                "groups": {
                    value: {
                        "count": len(objectives),
                        "mean_objective": statistics.fmean(objectives),
                        "median_objective": statistics.median(objectives),
                    }
                    for value, objectives in sorted(groups.items())
                },
                "interpretation": "Conditional descriptive groups; HPO samples are not causal replicates.",
            }
    return result
