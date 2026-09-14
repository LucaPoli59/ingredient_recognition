"""Decode persisted experiment configuration without importing recorded callables."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def decode_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: decode_data(item) for key, item in value.items()}
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
        kind, item = value
        if kind == "config":
            return decode_data(item)
        if kind == "None":
            return None
        if kind in {"list", "tuple", "ndarray"} and isinstance(item, str):
            try:
                return ast.literal_eval(item)
            except (SyntaxError, ValueError):
                return item
        if kind in {"int", "float", "bool", "str", "class", "function"}:
            return item
    if isinstance(value, list):
        return [decode_data(item) for item in value]
    return value


def read_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as stream:
        return decode_data(json.load(stream))


def class_name(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value.startswith("<class '") and value.endswith("'>"):
        return value[8:-2]
    if value.startswith("<function '") and value.endswith("'>"):
        return value[11:-2]
    return value
