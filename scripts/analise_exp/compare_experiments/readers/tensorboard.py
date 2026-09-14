"""Read scalar series from one or more TensorBoard event files."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_tensorboard_scalars(paths: Iterable[Path]) -> dict[str, list[dict[str, Any]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                result[tag].append({
                    "step": event.step,
                    "wall_time": event.wall_time,
                    "value": event.value,
                    "source": str(path),
                })
    for values in result.values():
        values.sort(key=lambda item: (item["step"], item["wall_time"]))
    return dict(result)
