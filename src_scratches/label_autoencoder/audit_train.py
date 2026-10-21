"""Read only v5 TRAIN metadata; print aggregate diagnostics without ML imports."""
import collections
import hashlib
import json
import math
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / "data/input/yummly/train/ingredients_target_v5_metadata.json"
    raw = source.read_bytes()
    rows = json.loads(raw)
    dimensions = ("bitter", "meaty", "piquant", "salty", "sour", "sweet")
    labels = collections.Counter()
    cardinality = collections.Counter()
    cuisines = collections.Counter()
    complete_by_cuisine = collections.Counter()
    flavor_status = collections.Counter()
    observed = {key: [] for key in dimensions}
    raw_key_schemas = collections.Counter()
    collisions = 0
    for row in rows:
        target = set(row["ingredients_target"])
        labels.update(target)
        cardinality[len(target)] += 1
        cuisine = str(row.get("cuisine", "")).strip().lower()
        cuisines[cuisine] += 1
        flavor = row.get("flavors")
        if not isinstance(flavor, dict) or not flavor:
            flavor_status["absent_or_empty"] += 1
            continue
        raw_key_schemas[tuple(sorted(flavor))] += 1
        normalized = {str(k).strip().lower(): v for k, v in flavor.items()}
        if len(normalized) != len(flavor):
            collisions += 1
        valid = {}
        for key in dimensions:
            value = normalized.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                valid[key] = value
                observed[key].append(value)
        if len(valid) == len(dimensions):
            flavor_status["six_numeric_dimensions"] += 1
            complete_by_cuisine[cuisine] += 1
        else:
            flavor_status["nonempty_incomplete_or_nonnumeric"] += 1
    n = len(rows)
    positives = sum(labels.values())
    report = {
        "scope": "Train metadata only; no images, validation, test, model or GPU accessed",
        "source": str(source.relative_to(root)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": n,
        "label_count": len(labels),
        "positive_assignments": positives,
        "mean_labels_per_recipe": positives / n,
        "label_density": positives / (n * len(labels)),
        "all_negative_label_accuracy": 1 - positives / (n * len(labels)),
        "label_cardinality_histogram": dict(sorted(cardinality.items())),
        "label_support_min_max": [min(labels.values()), max(labels.values())],
        "flavor_status": dict(flavor_status),
        "flavor_complete_fraction": flavor_status["six_numeric_dimensions"] / n,
        "flavor_normalized_key_collision_rows": collisions,
        "flavor_raw_key_schemas": [
            {"keys": list(keys), "rows": count}
            for keys, count in raw_key_schemas.most_common()
        ],
        "flavor_dimensions": {
            key: {
                "numeric_rows": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "outside_zero_one": sum(not 0 <= v <= 1 for v in values),
            }
            for key, values in observed.items()
        },
        "cuisine": {
            key: {"rows": count, "complete_flavor_rows": complete_by_cuisine[key],
                  "complete_flavor_fraction": complete_by_cuisine[key] / count}
            for key, count in sorted(cuisines.items())
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
