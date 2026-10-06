"""Opt-in consumption of frozen vocabularies; never select labels at runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


PROJECTION_ID = "ingredients_selected_v5_d6_v1"
_APPROVED = {PROJECTION_ID: "c8e9c88fc240dd3a27cfe527289e691ecdd579595776b71c5837c67a003e9767"}
_RESOURCES = Path(__file__).with_name("resources")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class IngredientProjection:
    projection_id: str
    artifact_hash: str
    class_order: tuple[str, ...]
    base_class_order: tuple[str, ...]
    base_class_indices: tuple[int, ...]
    metadata_filename: str
    target_field: str
    metadata_hashes: tuple[tuple[str, str], ...]

    def to_config(self):
        return {
            "projection_id": self.projection_id,
            "artifact_hash": self.artifact_hash,
            "class_order": list(self.class_order),
            "class_order_hash": _hash(self.class_order),
            "base_class_order_hash": _hash(self.base_class_order),
            "base_class_indices": list(self.base_class_indices),
        }

    def validate_data_config(self, metadata_filename, target_field, category=None):
        if metadata_filename != self.metadata_filename or target_field != self.target_field:
            raise ValueError("ingredient projection requires its original metadata generation and target field")
        if category not in (None, "all"):
            raise ValueError("ingredient projection must preserve the full record population; category filtering is forbidden")

    def verify_metadata(self, data_dir):
        for split, expected in self.metadata_hashes:
            path = Path(data_dir) / split / self.metadata_filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"ingredient projection {split} metadata differs from the frozen input")

    def project_targets(self, rows):
        base, selected = set(self.base_class_order), set(self.class_order)
        result = []
        for index, row in enumerate(rows):
            if isinstance(row, (str, bytes)) or not all(isinstance(x, str) for x in row):
                raise ValueError(f"invalid ingredient target row {index}")
            unknown = set(row) - base
            if unknown:
                raise ValueError(f"ingredient target row {index} has labels outside the base vocabulary: {sorted(unknown)}")
            result.append([name for name in row if name in selected])
        return result

    def project_columns(self, values, source_class_order):
        """Slice NumPy/Torch full-task matrices without guessing their column meaning."""
        if tuple(source_class_order) != self.base_class_order:
            raise ValueError("full-model source class order differs from the frozen base vocabulary")
        if len(values.shape) != 2 or values.shape[1] != len(self.base_class_order):
            raise ValueError("expected a record-by-base-label matrix")
        return values[:, list(self.base_class_indices)]


def resolve_projection(value=None):
    """Resolve a registered name or validate an exact saved contract; None is full-task."""
    if value is None:
        return None
    if not isinstance(value, (str, dict)):
        raise ValueError("ingredient_projection must be None, a registered ID or its saved config")
    name = value if isinstance(value, str) else value.get("projection_id")
    if not isinstance(name, str) or name not in _APPROVED:
        raise ValueError(f"unregistered ingredient projection: {name!r}")
    payload = json.loads((_RESOURCES / f"{name}.json").read_text(encoding="utf-8"))
    artifact_hash = payload.pop("artifact_hash", None)
    if artifact_hash != _APPROVED[name] or _hash(payload) != artifact_hash:
        raise ValueError("ingredient projection artifact hash differs from the approved resource")
    base = payload["base_vocabulary"]
    names, indices = payload["class_order"], payload["base_class_indices"]
    if (payload["schema_version"] != 1 or payload["projection_id"] != name
            or payload["label_count"] != len(names)
            or not names or len(set(names)) != len(names)
            or any(type(i) is not int or i < 0 or i >= len(base["class_order"]) for i in indices)
            or indices != sorted(set(indices))
            or [base["class_order"][i] for i in indices] != names
            or _hash(names) != payload["class_order_hash"]
            or _hash(base["class_order"]) != base["class_order_hash"]):
        raise ValueError("invalid frozen projection class mapping")
    result = IngredientProjection(name, artifact_hash, tuple(names), tuple(base["class_order"]),
                                  tuple(indices), base["metadata_filename"], base["target_field"],
                                  tuple(sorted(base["metadata_sha256"].items())))
    if isinstance(value, dict) and value != result.to_config():
        raise ValueError("saved ingredient projection contract differs from the approved resource")
    return result


def projection_config(value=None):
    projection = resolve_projection(value)
    return None if projection is None else projection.to_config()


def project_output_columns(values, source_class_order, projection=PROJECTION_ID):
    resolved = resolve_projection(projection)
    if resolved is None:
        raise ValueError("output projection requires an explicit selected vocabulary")
    return resolved.project_columns(values, source_class_order)
