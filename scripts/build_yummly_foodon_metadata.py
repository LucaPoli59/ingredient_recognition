"""Build the FoodOn-first Yummly ``ingredients_target`` generation.

This command is intentionally separate from the legacy/v1-v4 builder.  It
keeps those generations unchanged while implementing Work package 2.2d:
exact FoodOn association, bounded local fallback, exact retry, local concepts,
train-only support filtering, and deterministic exact-image-aware splitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.foodon_lexicon import DEFAULT_INDEX_PATH, load_packaged_foodon
from src.data_processing.ingredient_target_generation import derive_controlled_targets
from src.data_processing.ingredient_standardization import (
    DEFAULT_MIN_RECIPE_SUPPORT,
    DEFAULT_MIN_TARGETS_PER_RECIPE,
)
from scripts.build_yummly_metadata import (
    SPLITS,
    allocate_groups,
    load_source_records,
    resolve_image,
    sha256,
    validate_split,
)


def load_split_records(dataset_root: Path, split: str, metadata_filename: str) -> list[dict[str, Any]]:
    metadata_path = dataset_root / split / metadata_filename
    with metadata_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{metadata_path} must contain a list")
    return records


def threshold_summary(
    all_targets: list[list[str]],
    train_support: Counter[str],
    thresholds: tuple[int, ...],
    min_targets_per_recipe: int,
) -> list[dict[str, object]]:
    rows = []
    total_assignments = sum(train_support.values())
    for threshold in thresholds:
        retained = {target for target, support in train_support.items() if support >= threshold}
        cardinalities = [len(set(targets) & retained) for targets in all_targets]
        assignments = sum(train_support[target] for target in retained)
        rows.append(
            {
                "minimum_train_recipe_support": threshold,
                "vocabulary_size": len(retained),
                "retained_train_assignments": assignments,
                "retained_train_assignment_fraction": assignments / total_assignments if total_assignments else 0.0,
                "recipes_with_zero_targets": sum(value == 0 for value in cardinalities),
                "recipes_with_one_target": sum(value == 1 for value in cardinalities),
                "recipes_with_two_targets": sum(value == 2 for value in cardinalities),
                "recipes_with_at_least_three_targets": sum(value >= min_targets_per_recipe for value in cardinalities),
            }
        )
    return rows


def load_generation_by_id(dataset_root: Path, metadata_filename: str) -> dict[object, tuple[str, dict[str, Any]]]:
    result: dict[object, tuple[str, dict[str, Any]]] = {}
    for split in SPLITS:
        for record in load_split_records(dataset_root, split, metadata_filename):
            record_id = record.get("id")
            if record_id in result:
                raise ValueError(f"duplicate record id in generation {metadata_filename}: {record_id!r}")
            result[record_id] = (split, record)
    return result


def comparison_summary(
    dataset_root: Path,
    comparison_metadata_filename: str,
    records_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, object]:
    """Compare retained IDs and target lists with the v4 baseline."""
    baseline = load_generation_by_id(dataset_root, comparison_metadata_filename)
    current = {
        record.get("id"): (split, record)
        for split in SPLITS
        for record in records_by_split[split]
    }
    shared = set(baseline) & set(current)
    changed_targets = sum(
        baseline[record_id][1].get("ingredients_target")
        != current[record_id][1].get("ingredients_target")
        for record_id in shared
    )
    changed_splits = sum(baseline[record_id][0] != current[record_id][0] for record_id in shared)
    baseline_vocab = {
        target for _, record in baseline.values() for target in record.get("ingredients_target", [])
    }
    current_vocab = {
        target for _, record in current.values() for target in record.get("ingredients_target", [])
    }
    return {
        "baseline_metadata": comparison_metadata_filename,
        "baseline_records": len(baseline),
        "current_records": len(current),
        "shared_record_ids": len(shared),
        "removed_record_ids": len(set(baseline) - set(current)),
        "added_record_ids": len(set(current) - set(baseline)),
        "shared_records_with_changed_targets": changed_targets,
        "shared_records_with_unchanged_targets": len(shared) - changed_targets,
        "shared_records_with_changed_splits": changed_splits,
        "baseline_vocabulary_size": len(baseline_vocab),
        "current_vocabulary_size": len(current_vocab),
        "new_target_strings": sorted(current_vocab - baseline_vocab),
        "removed_target_strings": sorted(baseline_vocab - current_vocab),
    }


def build_generation(
    dataset_root: Path,
    source_metadata_filename: str,
    foodon_index: Path,
    min_recipe_support: int,
    min_targets_per_recipe: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, object]]:
    images_root = (dataset_root / "imgs" / "standard").resolve()
    if not images_root.is_dir():
        raise FileNotFoundError(f"common image directory not found: {images_root}")
    source_records = load_source_records(dataset_root, source_metadata_filename)
    source_train = load_split_records(dataset_root, "train", source_metadata_filename)
    lexicon = load_packaged_foodon(foodon_index)
    generation = derive_controlled_targets(
        source_records,
        source_train,
        lexicon,
        min_recipe_support=min_recipe_support,
        min_targets_per_recipe=min_targets_per_recipe,
    )

    retained_records: list[dict[str, Any]] = []
    for index in generation.retained_records:
        record = dict(source_records[index])
        record["ingredients_target"] = generation.targets_by_record[index]
        image_path = resolve_image(images_root, record.get("image"))
        record["_sha256"] = sha256(image_path)
        retained_records.append(record)

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in retained_records:
        groups.setdefault(record.pop("_sha256"), []).append(record)
    records_by_split = allocate_groups(groups)
    for records in records_by_split.values():
        records.sort(key=lambda record: str(record["id"]))
    validate_split(records_by_split, images_root)

    train_support = Counter(generation.train_support)
    report = {
        "method": {
            "association": "exact preferred label or exact synonym; then bounded local fallback and exact retry",
            "fuzzy_matching": "disabled",
            "unresolved": "retained as canonical local concepts",
            "support": "distinct source-train recipes per canonical target",
            "minimum_targets_per_recipe": min_targets_per_recipe,
        },
        "source": {
            "metadata_filename": source_metadata_filename,
            "records": len(source_records),
            "train_records_for_support": len(source_train),
        },
        "foodon": {
            "index": str(foodon_index),
            "concepts": lexicon.concept_count,
            "surfaces": len(lexicon.surface_to_ids),
        },
        "association_statuses": generation.association_statuses,
        "support": {
            "canonical_targets_before_filter": len(generation.train_support),
            "canonical_targets_at_selected_threshold": len(
                {target for target, support in train_support.items() if support >= min_recipe_support}
            ),
            "selected_minimum_train_recipe_support": min_recipe_support,
            "threshold_sweep": threshold_summary(
                generation.unfiltered_targets_by_record,
                train_support,
                (10, 25, 50, 100, 250, 500, 750, 1000),
                min_targets_per_recipe,
            ),
        },
        "retention": {
            "records_by_split": {split: len(records_by_split[split]) for split in SPLITS},
            "records_retained_before_split": len(retained_records),
        },
    }
    return records_by_split, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/input/yummly"))
    parser.add_argument("--source-metadata", default="metadata.json")
    parser.add_argument("--foodon-index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output-metadata", default="ingredients_target_v5_metadata.json")
    parser.add_argument("--compare-metadata", default="ingredients_target_v4_metadata.json")
    parser.add_argument("--min-recipe-support", type=int, default=DEFAULT_MIN_RECIPE_SUPPORT)
    parser.add_argument("--min-targets-per-recipe", type=int, default=DEFAULT_MIN_TARGETS_PER_RECIPE)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("src_scratches/data_anlysis/outputs/controlled_target_generation/report.json"),
    )
    parser.add_argument("--apply", action="store_true", help="write the validated generation")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_paths = [dataset_root / split / args.output_metadata for split in SPLITS]
    if any(path.exists() for path in output_paths):
        raise FileExistsError(f"refusing to overwrite an existing generation: {args.output_metadata}")
    records_by_split, report = build_generation(
        dataset_root,
        args.source_metadata,
        args.foodon_index.resolve(),
        args.min_recipe_support,
        args.min_targets_per_recipe,
    )
    comparison_path = dataset_root / "train" / args.compare_metadata
    if comparison_path.is_file():
        report["comparison_with_baseline"] = comparison_summary(
            dataset_root, args.compare_metadata, records_by_split
        )
    print("records by split:", {split: len(records_by_split[split]) for split in SPLITS})
    print("vocabulary size:", len({target for record in records_by_split["train"] for target in record["ingredients_target"]}))
    print("association statuses:", report["association_statuses"])
    if not args.apply:
        print("dry run complete; no metadata was written")
        return

    for split, output_path in zip(SPLITS, output_paths):
        with output_path.open("x", encoding="utf-8") as output_file:
            json.dump(records_by_split[split], output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2, sort_keys=True)
        report_file.write("\n")
    print(f"wrote {args.output_metadata} to train, val, and test")
    print(f"wrote report to {args.report_output}")


if __name__ == "__main__":
    main()
