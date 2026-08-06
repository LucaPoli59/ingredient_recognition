"""Audit one ingredient field from one Yummly metadata generation.

The script is intentionally smaller than ``yummly_audit.py`` and
``ingredient_vocabulary_audit.py``.  It is meant for quick, repeatable
questions such as:

    python src_scratches/data_anlysis/metadata_field_audit.py \
        --metadata ingredients_target_v5_metadata.json \
        --field ingredients_target

    python src_scratches/data_anlysis/metadata_field_audit.py \
        --metadata metadata.json --field ingredients

The selected field must contain a list per recipe.  Values are reported in
two complementary ways:

* ``occurrence_count`` counts every list item, including repeated raw lines;
* ``recipe_support`` counts the number of distinct recipes containing the
  value, which is the useful support measure for multi-label targets.

The source metadata and records are never modified.  For ``ingredients`` the
raw line is intentionally preserved; use ``--normalize-ingredients`` when an
additional summary based on the current deterministic normalizer is useful.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.ingredient_standardization import normalize_ingredient_line


SPLITS = ("train", "val", "test")
FIELDS = ("ingredients", "ingredients_target")
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "input" / "yummly"
DEFAULT_METADATA = "ingredients_target_v5_metadata.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "src_scratches" / "data_anlysis" / "outputs" / "metadata_field_audit"


def percentile(values: Iterable[int], probability: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))


def distribution(values: Iterable[int]) -> dict[str, float | int | None]:
    values = list(values)
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def canonical_value(value: str) -> str:
    """Return a stable key while retaining the original value in the CSV."""
    return " ".join(value.casefold().split())


def load_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as metadata_file:
        records = json.load(metadata_file)
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{path} must contain a JSON list of objects")
    return records


def resolve_metadata_path(dataset_root: Path, split: str, metadata: str) -> Path:
    """Resolve either a split-relative filename or an explicit metadata path."""
    candidate = Path(metadata)
    if candidate.is_file():
        return candidate.resolve()
    path = dataset_root / split / metadata
    if not path.is_file():
        raise FileNotFoundError(f"metadata file not found: {path}")
    return path.resolve()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def value_rows(
    occurrence_counts: Counter[str],
    recipe_support: Counter[str],
    examples: dict[str, list[str]],
    record_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for value, occurrence_count in occurrence_counts.most_common():
        support = recipe_support[value]
        rows.append(
            {
                "value": value,
                "occurrence_count": occurrence_count,
                "recipe_support": support,
                "recipe_prevalence": round(support / record_count, 8) if record_count else 0.0,
                "occurrences_per_supporting_recipe": round(occurrence_count / support, 8) if support else 0.0,
                "examples": " | ".join(examples[value]),
            }
        )
    return rows


def audit_records(
    records: list[dict[str, Any]],
    *,
    field: str,
    split: str,
    metadata_path: Path,
    top_k: int,
    normalize_ingredients: bool,
    include_pairs: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if field not in FIELDS:
        raise ValueError(f"field must be one of: {', '.join(FIELDS)}")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    occurrence_counts: Counter[str] = Counter()
    recipe_support: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    cardinalities: list[int] = []
    unique_cardinalities: list[int] = []
    duplicate_items = 0
    missing_field_records = 0
    invalid_field_records = 0
    invalid_item_count = 0
    empty_item_count = 0
    malformed_examples: list[dict[str, Any]] = []
    problematic_record_ids: set[object] = set()
    record_rows: list[dict[str, Any]] = []
    cuisine_records: Counter[str] = Counter()
    cuisine_support: dict[str, Counter[str]] = defaultdict(Counter)
    pair_counts: Counter[tuple[str, str]] = Counter()

    for record in records:
        record_id = record.get("id")
        raw_values = record.get(field)
        if raw_values is None:
            missing_field_records += 1
            problematic_record_ids.add(record_id)
            raw_values = []
            if len(malformed_examples) < 20:
                malformed_examples.append({"id": record_id, "problem": "missing_or_null_field"})
        elif not isinstance(raw_values, list):
            invalid_field_records += 1
            problematic_record_ids.add(record_id)
            raw_values = []
            if len(malformed_examples) < 20:
                malformed_examples.append({"id": record_id, "problem": "field_is_not_a_list"})

        values: list[str] = []
        for raw_value in raw_values:
            if not isinstance(raw_value, str):
                invalid_item_count += 1
                problematic_record_ids.add(record_id)
                continue
            stripped = raw_value.strip()
            if not stripped:
                empty_item_count += 1
                problematic_record_ids.add(record_id)
                continue
            values.append(stripped)

        keys = [canonical_value(value) for value in values]
        unique_keys = set(keys)
        duplicate_items += len(keys) - len(unique_keys)
        cardinalities.append(len(values))
        unique_cardinalities.append(len(unique_keys))
        occurrence_counts.update(keys)
        recipe_support.update(unique_keys)
        for key, original in zip(keys, values):
            if len(examples[key]) < 5 and original not in examples[key]:
                examples[key].append(original)

        cuisine = str(record.get("cuisine", "<missing>")).casefold()
        cuisine_records[cuisine] += 1
        cuisine_support[cuisine].update(unique_keys)
        if include_pairs:
            pair_counts.update(combinations(sorted(unique_keys), 2))

        record_rows.append(
            {
                "split": record.get("_audit_split", split),
                "id": record_id,
                "old_id": record.get("old_id", ""),
                "name": record.get("name", ""),
                "cuisine": record.get("cuisine", ""),
                "value_count": len(values),
                "unique_value_count": len(unique_keys),
                "duplicate_value_count": len(keys) - len(unique_keys),
                "values": " | ".join(values),
            }
        )

    rows = value_rows(occurrence_counts, recipe_support, examples, len(records))
    top_values = rows[:top_k]
    support_counts = Counter(recipe_support.values())
    cardinality_histogram = Counter(cardinalities)
    unique_cardinality_histogram = Counter(unique_cardinalities)

    cuisine_rows = []
    for cuisine, count in cuisine_records.most_common():
        top_cuisine_values = cuisine_support[cuisine].most_common(top_k)
        cuisine_rows.append(
            {
                "cuisine": cuisine,
                "record_count": count,
                "unique_value_count": len(cuisine_support[cuisine]),
                "top_values": " | ".join(f"{value}:{support}" for value, support in top_cuisine_values),
            }
        )

    pair_rows = [
        {"left": left, "right": right, "cooccurrence": count}
        for (left, right), count in pair_counts.most_common()
    ]

    normalized_rows: list[dict[str, Any]] = []
    normalized_summary: dict[str, Any] | None = None
    if normalize_ingredients:
        if field != "ingredients":
            raise ValueError("--normalize-ingredients is only valid with --field ingredients")
        normalized_occurrences: Counter[str] = Counter()
        normalized_support: Counter[str] = Counter()
        normalized_examples: dict[str, list[str]] = defaultdict(list)
        for record in records:
            normalized_values = {
                target
                for raw_value in (record.get(field) or [])
                if isinstance(raw_value, str)
                if (target := normalize_ingredient_line(raw_value)) is not None
            }
            normalized_occurrences.update(normalized_values)
            normalized_support.update(normalized_values)
            for target in normalized_values:
                if len(normalized_examples[target]) < 5:
                    normalized_examples[target].append(target)
        normalized_rows = value_rows(
            normalized_occurrences,
            normalized_support,
            normalized_examples,
            len(records),
        )
        normalized_cardinalities = [
            len(
                {
                    target
                    for raw_value in (record.get(field) or [])
                    if isinstance(raw_value, str)
                    if (target := normalize_ingredient_line(raw_value)) is not None
                }
            )
            for record in records
        ]
        normalized_summary = {
            "normalizer": "src.data_processing.ingredient_standardization.normalize_ingredient_line",
            "unique_value_count": len(normalized_occurrences),
            "total_recipe_assignments": sum(normalized_occurrences.values()),
            "cardinality": distribution(normalized_cardinalities),
        }

    report = {
        "metadata": {
            "path": str(metadata_path),
            "filename": metadata_path.name,
            "split": split,
            "field": field,
            "records": len(records),
        },
        "field_quality": {
            "missing_or_null_field_records": missing_field_records,
            "non_list_field_records": invalid_field_records,
            "invalid_items": invalid_item_count,
            "empty_string_items": empty_item_count,
            "records_with_any_problem": len(problematic_record_ids),
            "duplicate_items_within_records": duplicate_items,
            "examples": malformed_examples,
        },
        "value_summary": {
            "unique_value_count": len(occurrence_counts),
            "total_value_occurrences": sum(occurrence_counts.values()),
            "total_recipe_assignments": sum(recipe_support.values()),
            "top_k": top_k,
            "top_values": top_values,
            "support_distribution": distribution(recipe_support.values()),
            "support_histogram": {str(key): support_counts[key] for key in sorted(support_counts)},
        },
        "cardinality": {
            "valid_values_per_record": distribution(cardinalities),
            "unique_values_per_record": distribution(unique_cardinalities),
            "valid_value_histogram": {str(key): cardinality_histogram[key] for key in sorted(cardinality_histogram)},
            "unique_value_histogram": {
                str(key): unique_cardinality_histogram[key] for key in sorted(unique_cardinality_histogram)
            },
        },
        "cuisine_summary": cuisine_rows,
        "normalized_ingredients": normalized_summary,
        "pair_summary": {
            "enabled": include_pairs,
            "pair_count": len(pair_counts),
            "top_pairs": pair_rows[:top_k],
        },
    }
    return report, rows, record_rows, cuisine_rows, normalized_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA,
        help="metadata filename under each split, or an explicit path to one JSON file",
    )
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="train")
    parser.add_argument("--field", choices=FIELDS, default="ingredients_target")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--normalize-ingredients",
        action="store_true",
        help="also write a summary after the current deterministic normalizer (ingredients only)",
    )
    parser.add_argument(
        "--include-pairs",
        action="store_true",
        help="compute co-occurrence pairs; disabled by default because raw ingredients can be numerous",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory; defaults to outputs/metadata_field_audit/<metadata>/<split>/<field>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    dataset_root = args.dataset_root.resolve()
    selected_splits = SPLITS if args.split == "all" else (args.split,)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / Path(args.metadata).stem / args.split / args.field
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    paths: list[Path] = []
    for split in selected_splits:
        path = resolve_metadata_path(dataset_root, split, args.metadata)
        paths.append(path)
        records = load_metadata(path)
        for record in records:
            record = dict(record)
            record.setdefault("_audit_split", split)
            all_records.append(record)

    metadata_path = paths[0] if len(paths) == 1 else dataset_root
    report, rows, record_rows, cuisine_rows, normalized_rows = audit_records(
        all_records,
        field=args.field,
        split=args.split,
        metadata_path=metadata_path,
        top_k=args.top_k,
        normalize_ingredients=args.normalize_ingredients,
        include_pairs=args.include_pairs,
    )
    report["metadata"]["paths"] = [str(path) for path in paths]

    (output_dir / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "value_counts.csv", rows, list(rows[0]) if rows else ["value"])
    write_csv(output_dir / "record_cardinality.csv", record_rows, list(record_rows[0]) if record_rows else ["id"])
    write_csv(output_dir / "cuisine_summary.csv", cuisine_rows, list(cuisine_rows[0]) if cuisine_rows else ["cuisine"])
    if args.include_pairs:
        pair_rows = report["pair_summary"]["top_pairs"]
        write_csv(output_dir / "top_pairs.csv", pair_rows, list(pair_rows[0]) if pair_rows else ["left", "right", "cooccurrence"])
    if normalized_rows:
        write_csv(output_dir / "normalized_value_counts.csv", normalized_rows, list(normalized_rows[0]))

    print(f"records: {len(all_records)}")
    print(f"field: {args.field}")
    print(f"unique values: {report['value_summary']['unique_value_count']}")
    print(f"cardinality: {report['cardinality']['valid_values_per_record']}")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
