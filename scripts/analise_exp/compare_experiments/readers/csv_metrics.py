"""Read Lightning CSV metrics as traceable observations."""
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

NEW_INGREDIENT_RE = re.compile(
    r"^(train|val|test)_per_ingredient/(precision|recall|f1)/(\d+)$"
)
LEGACY_INGREDIENT_RE = re.compile(
    r"^(train|val|test)_(f1_label|precision_label|recall_label)_(\d+)$"
)


def parse_ingredient_key(key: str) -> tuple[str, str, str, bool] | None:
    match = NEW_INGREDIENT_RE.fullmatch(key)
    legacy = False
    if match is None:
        match = LEGACY_INGREDIENT_RE.fullmatch(key)
        legacy = match is not None
    if match is None:
        return None
    split, metric, label_index = match.groups()
    return split, metric.removesuffix("_label"), label_index, legacy


def _number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def read_csv_metrics(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"series": {}, "ingredient_series": {}, "columns": [], "rows": 0}
    series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ingredients: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        row_count = 0
        for row_index, row in enumerate(reader):
            row_count += 1
            epoch, step = _number(row.get("epoch")), _number(row.get("step"))
            for key, raw in row.items():
                if key in {"epoch", "step"}:
                    continue
                value = _number(raw)
                if value is None:
                    continue
                observation = {
                    "epoch": epoch,
                    "step": step,
                    "value": value,
                    "source": str(path),
                    "row": row_index,
                }
                series[key].append(observation)
                parsed = parse_ingredient_key(key)
                if parsed:
                    split, metric, label_index, legacy = parsed
                    ingredients[f"{split}/{metric}/{label_index}"].append(
                        observation | {"legacy_key": legacy}
                    )
    return {
        "series": dict(series),
        "ingredient_series": dict(ingredients),
        "columns": columns,
        "rows": row_count,
    }
