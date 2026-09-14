"""Training-curve summaries with explicit observed-window semantics."""
from __future__ import annotations

import math
import statistics
from typing import Any


def _coordinate(point: dict[str, Any], fallback: int) -> float:
    for key in ("analysis_coordinate", "epoch", "step"):
        value = point.get(key)
        if value is not None:
            return float(value)
    return float(fallback)


def summarize_curve(points: list[dict[str, Any]], direction: str,
                    target: float | None = None) -> dict[str, Any]:
    usable = [point for point in points if point.get("value") is not None]
    if not usable:
        return {"status": "unavailable", "reason": "no finite observations"}
    coordinates = [_coordinate(point, index) for index, point in enumerate(usable)]
    values = [float(point["value"]) for point in usable]
    choose = min if direction == "min" else max
    best_value = choose(values)
    best_index = values.index(best_value)
    span = coordinates[-1] - coordinates[0]
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2
        for left_x, right_x, left_y, right_y in zip(
            coordinates, coordinates[1:], values, values[1:]
        )
    )
    tail_start = max(0, math.floor(len(values) * 0.75))
    tail_x, tail_y = coordinates[tail_start:], values[tail_start:]
    if len(tail_x) >= 2 and len(set(tail_x)) > 1:
        mean_x, mean_y = statistics.fmean(tail_x), statistics.fmean(tail_y)
        denominator = sum((value - mean_x) ** 2 for value in tail_x)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(tail_x, tail_y)) / denominator
    else:
        slope = None
    early = values[:max(1, math.ceil(len(values) * 0.25))]
    target_crossing = None
    if target is not None:
        reached = [
            (coordinate, value) for coordinate, value in zip(coordinates, values)
            if (value <= target if direction == "min" else value >= target)
        ]
        target_crossing = ({
            "status": "reached",
            "target": target,
            "coordinate": reached[0][0],
            "value": reached[0][1],
        } if reached else {
            "status": "censored",
            "target": target,
            "last_coordinate": coordinates[-1],
        })
    return {
        "status": "available",
        "observations": len(values),
        "window": [coordinates[0], coordinates[-1]],
        "best": {
            "value": best_value,
            "coordinate": coordinates[best_index],
            "observation_index": best_index,
        },
        "last": {"value": values[-1], "coordinate": coordinates[-1]},
        "normalized_auc": area / span if span > 0 else None,
        "target_crossing": target_crossing,
        "early_mean": statistics.fmean(early),
        "late_mean": statistics.fmean(tail_y),
        "late_median": statistics.median(tail_y),
        "early_to_late_mean_change": statistics.fmean(tail_y) - statistics.fmean(early),
        "late_slope": slope,
        "late_std": statistics.pstdev(tail_y) if len(tail_y) > 1 else 0.0,
        "points": [
            {"coordinate": coordinate, "value": value}
            for coordinate, value in zip(coordinates, values)
        ],
    }
