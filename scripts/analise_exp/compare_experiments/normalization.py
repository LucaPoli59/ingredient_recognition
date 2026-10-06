"""Normalize configuration and scalar observations for analysis."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .readers.config import class_name
from src.ingredient_selection.runtime import resolve_projection


def flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    output = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else key
            output.update(flatten_config(item, child))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        output[prefix] = class_name(value)
    return output


def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    return config.get(name, {}) if isinstance(config.get(name, {}), dict) else {}


def label_contract(config: dict[str, Any]) -> dict[str, Any]:
    datamodule = config_section(config, "datamodule_hyper_parameters")
    encoder = datamodule.get("label_encoder", {})
    classes = encoder.get("classes") if isinstance(encoder, dict) else None
    if not isinstance(classes, list):
        classes = None
    projection = resolve_projection(datamodule.get("ingredient_projection"))
    model_projection = resolve_projection(config_section(config, "hyper_parameters").get("ingredient_projection"))
    if model_projection != projection:
        raise ValueError("analysis configuration has conflicting ingredient projections")
    if projection is not None:
        projection.validate_data_config(datamodule.get("metadata_filename"), datamodule.get("feature_label"),
                                        datamodule.get("category"))
        if classes is not None and classes != list(projection.class_order):
            raise ValueError("analysis encoder order conflicts with the frozen projection")
        classes = list(projection.class_order)
    encoded = json.dumps(classes, ensure_ascii=False, separators=(",", ":")) if classes is not None else None
    return {
        "classes": classes,
        "count": len(classes) if classes is not None else None,
        "sha256": hashlib.sha256(encoded.encode()).hexdigest() if encoded is not None else None,
        "mapping_status": "available" if classes is not None else "unavailable",
        "projection_id": None if projection is None else projection.projection_id,
        "projection_artifact_hash": None if projection is None else projection.artifact_hash,
    }


def select_metric_series(csv_data: dict[str, Any], tensorboard: dict[str, Any],
                         metric: str) -> dict[str, Any]:
    csv_points = csv_data.get("series", {}).get(metric, [])
    tensorboard_points = tensorboard.get(metric, [])
    if not csv_points and not tensorboard_points:
        return {
            "status": "unavailable",
            "reason": f"metric {metric!r} absent from CSV and TensorBoard",
            "points": [],
            "sources": {"csv": [], "tensorboard": []},
        }

    conflicts = []
    csv_by_step = {
        int(point["step"]): point for point in csv_points if point.get("step") is not None
    }
    tensorboard_by_step = {}
    for point in tensorboard_points:
        step = int(point["step"])
        previous = tensorboard_by_step.get(step)
        if previous is not None and abs(previous["value"] - point["value"]) > 1e-6:
            conflicts.append({
                "step": step,
                "kind": "tensorboard_restart_conflict",
                "values": [previous["value"], point["value"]],
                "sources": [previous["source"], point["source"]],
            })
        if previous is None or point.get("wall_time", 0) >= previous.get("wall_time", 0):
            tensorboard_by_step[step] = point

    if csv_points and tensorboard_points and csv_by_step:
        points = []
        for step in sorted(set(csv_by_step) | set(tensorboard_by_step)):
            csv_point = csv_by_step.get(step)
            tensorboard_point = tensorboard_by_step.get(step)
            if csv_point is not None:
                selected = dict(csv_point)
                if tensorboard_point and abs(csv_point["value"] - tensorboard_point["value"]) > 1e-6:
                    conflicts.append({
                        "step": step,
                        "kind": "csv_tensorboard_conflict",
                        "values": [csv_point["value"], tensorboard_point["value"]],
                        "sources": [csv_point["source"], tensorboard_point["source"]],
                    })
            else:
                selected = dict(tensorboard_point)
            selected["analysis_coordinate"] = float(step)
            points.append(selected)
        axis = "optimizer_step"
        policy = "CSV wins at shared optimizer steps; latest TensorBoard event fills missing steps."
    elif csv_points:
        points = [dict(point) for point in csv_points]
        axis = "epoch" if any(point.get("epoch") is not None for point in points) else "optimizer_step"
        for index, point in enumerate(points):
            point["analysis_coordinate"] = (
                point.get("epoch") if axis == "epoch" else point.get("step", index)
            )
        policy = "CSV only."
    else:
        points = []
        for step, point in sorted(tensorboard_by_step.items()):
            points.append(dict(point) | {"analysis_coordinate": float(step)})
        axis = "optimizer_step"
        policy = "Latest TensorBoard event at each step; restart conflicts are retained."

    return {
        "status": "available",
        "primary_source": "reconciled" if csv_points and tensorboard_points else ("csv" if csv_points else "tensorboard"),
        "analysis_axis": axis,
        "points": points,
        "sources": {"csv": csv_points, "tensorboard": tensorboard_points},
        "conflicts": conflicts,
        "source_conflict_policy": policy,
    }
