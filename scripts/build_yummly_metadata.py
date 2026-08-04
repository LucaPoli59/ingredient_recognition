"""Build one deterministic Yummly ``ingredients_target`` metadata generation.

The source metadata and every legacy artifact remain read-only.  The command
only writes the requested new filename after all validations have passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePath
from typing import Any, Iterable

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.ingredient_standardization import (
    DEFAULT_MIN_RECIPE_SUPPORT,
    DEFAULT_MIN_TARGETS_PER_RECIPE,
    derive_ingredients_target,
)


SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_records(dataset_root: Path, metadata_filename: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ids = set()
    for split in SPLITS:
        metadata_path = dataset_root / split / metadata_filename
        with metadata_path.open(encoding="utf-8") as metadata_file:
            split_records = json.load(metadata_file)
        if not isinstance(split_records, list):
            raise ValueError(f"{metadata_path} must contain a list")
        for record in split_records:
            record_id = record.get("id")
            if record_id in ids:
                raise ValueError(f"duplicate record id in source metadata: {record_id!r}")
            ids.add(record_id)
            records.append(record)
    return records


def resolve_image(images_root: Path, image_ref: object) -> Path:
    if not isinstance(image_ref, str) or not image_ref:
        raise ValueError(f"invalid image reference: {image_ref!r}")
    relative = PurePath(image_ref)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe image reference: {image_ref!r}")
    image_path = (images_root / relative).resolve()
    if images_root not in image_path.parents or not image_path.is_file():
        raise FileNotFoundError(f"missing image below common root: {image_ref!r}")
    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as error:
        raise ValueError(f"image does not decode: {image_path}") from error
    return image_path


def record_features(record: dict[str, Any]) -> Counter[str]:
    features = Counter({f"cuisine:{str(record.get('cuisine', '')).casefold()}": 1})
    features.update(f"target:{target}" for target in record["ingredients_target"])
    return features


def allocate_groups(groups: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Allocate exact-image groups with a deterministic weighted deficit score."""
    total_records = sum(len(group) for group in groups.values())
    total_features = Counter()
    group_features = {}
    for checksum, group in groups.items():
        features = sum((record_features(record) for record in group), Counter())
        group_features[checksum] = features
        total_features.update(features)

    split_records = {split: [] for split in SPLITS}
    split_sizes = Counter()
    split_features = {split: Counter() for split in SPLITS}
    ordered_groups = sorted(
        groups,
        key=lambda checksum: (
            min(total_features[feature] for feature in group_features[checksum]),
            -len(groups[checksum]),
            checksum,
        ),
    )

    for checksum in ordered_groups:
        group = groups[checksum]
        features = group_features[checksum]
        group_size = len(group)
        candidate_scores = {}
        for split in SPLITS:
            target_size = total_records * RATIOS[split]
            score = (target_size - split_sizes[split]) / max(target_size, 1)
            for feature, count in features.items():
                target_count = total_features[feature] * RATIOS[split]
                # Rarer features need more balancing pressure than ubiquitous labels.
                weight = 1 / math.sqrt(total_features[feature])
                score += weight * (target_count - split_features[split][feature]) / max(target_count, 1)
            candidate_scores[split] = score
        split = max(SPLITS, key=lambda candidate: (candidate_scores[candidate], -SPLITS.index(candidate)))
        split_records[split].extend(group)
        split_sizes[split] += group_size
        split_features[split].update(features)
    return split_records


def validate_split(records_by_split: dict[str, list[dict[str, Any]]], images_root: Path) -> None:
    seen_ids = set()
    checksums_by_split = {split: set() for split in SPLITS}
    all_records = [record for split in SPLITS for record in records_by_split[split]]
    total = len(all_records)
    all_features = Counter(feature for record in all_records for feature in record_features(record))

    for split in SPLITS:
        for record in records_by_split[split]:
            record_id = record.get("id")
            if record_id in seen_ids:
                raise AssertionError(f"record id appears in more than one split: {record_id!r}")
            seen_ids.add(record_id)
            targets = record.get("ingredients_target")
            if not isinstance(targets, list) or not targets or targets != sorted(set(targets)):
                raise AssertionError(f"invalid ingredients_target for record {record_id!r}")
            checksums_by_split[split].add(sha256(resolve_image(images_root, record.get("image"))))

        expected_size = total * RATIOS[split]
        if abs(len(records_by_split[split]) - expected_size) > max(5, math.ceil(total * 0.02)):
            raise AssertionError(
                f"{split} ratio is outside the 2% tolerance: "
                f"{len(records_by_split[split])} instead of {expected_size:.1f}"
            )

    if checksums_by_split["train"] & checksums_by_split["val"] or checksums_by_split["train"] & checksums_by_split["test"] or checksums_by_split["val"] & checksums_by_split["test"]:
        raise AssertionError("an exact-image SHA-256 group crosses splits")

    for feature, support in all_features.items():
        if support < 20:
            continue
        for split in SPLITS:
            actual = sum(record_features(record)[feature] for record in records_by_split[split])
            expected = support * RATIOS[split]
            if abs(actual - expected) > max(5, math.ceil(expected * 0.25)):
                raise AssertionError(f"{feature} is outside the 25% distribution tolerance in {split}")

    train_vocabulary = {target for record in records_by_split["train"] for target in record["ingredients_target"]}
    held_out_targets = {
        target
        for split in ("val", "test")
        for record in records_by_split[split]
        for target in record["ingredients_target"]
        if target not in train_vocabulary
    }
    if held_out_targets:
        raise AssertionError(f"validation/test targets missing from training vocabulary: {sorted(held_out_targets)}")


def build_generation(
    dataset_root: Path,
    source_metadata_filename: str,
    min_recipe_support: int,
    min_targets_per_recipe: int,
) -> dict[str, list[dict[str, Any]]]:
    images_root = (dataset_root / "imgs" / "standard").resolve()
    if not images_root.is_dir():
        raise FileNotFoundError(f"common image directory not found: {images_root}")
    source_records = load_source_records(dataset_root, source_metadata_filename)
    standardization = derive_ingredients_target(source_records, min_recipe_support, min_targets_per_recipe)
    retained_records = []
    for index in standardization.retained_records:
        record = dict(source_records[index])
        record["ingredients_target"] = standardization.targets_by_record[index]
        image_path = resolve_image(images_root, record.get("image"))
        record["_sha256"] = sha256(image_path)
        retained_records.append(record)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in retained_records:
        groups[record.pop("_sha256")].append(record)
    records_by_split = allocate_groups(groups)
    for records in records_by_split.values():
        records.sort(key=lambda record: str(record["id"]))
    validate_split(records_by_split, images_root)
    return records_by_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/input/yummly"))
    parser.add_argument("--source-metadata", default="metadata.json")
    parser.add_argument("--output-metadata", default="ingredients_target_v1_metadata.json")
    parser.add_argument("--min-recipe-support", type=int, default=DEFAULT_MIN_RECIPE_SUPPORT)
    parser.add_argument("--min-targets-per-recipe", type=int, default=DEFAULT_MIN_TARGETS_PER_RECIPE)
    parser.add_argument("--apply", action="store_true", help="write the validated metadata generation")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_paths = [dataset_root / split / args.output_metadata for split in SPLITS]
    if any(path.exists() for path in output_paths):
        raise FileExistsError(f"refusing to overwrite an existing generation: {args.output_metadata}")
    records_by_split = build_generation(
        dataset_root, args.source_metadata, args.min_recipe_support, args.min_targets_per_recipe
    )
    print("records by split:", {split: len(records_by_split[split]) for split in SPLITS})
    print("vocabulary size:", len({target for record in records_by_split["train"] for target in record["ingredients_target"]}))
    if not args.apply:
        print("dry run complete; no metadata was written")
        return
    for split, output_path in zip(SPLITS, output_paths):
        with output_path.open("x", encoding="utf-8") as output_file:
            json.dump(records_by_split[split], output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
    print(f"wrote {args.output_metadata} to train, val, and test")


if __name__ == "__main__":
    main()
