"""Compare train-only support thresholds for provisional ingredient targets.

The script normalizes the original ``ingredients`` field with the current
deterministic rules, then measures candidate support thresholds.  It is a
decision aid only: it never writes metadata, changes the split, or chooses a
threshold.  The later controlled-vocabulary pipeline must rerun the same
analysis after its final association rules are implemented.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_processing.ingredient_standardization import normalized_recipe_targets


DEFAULT_THRESHOLDS = (10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000)


def parse_thresholds(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, strictly positive support-threshold list."""
    try:
        thresholds = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated positive integers") from error
    if not thresholds or thresholds[0] < 1:
        raise argparse.ArgumentTypeError("thresholds must contain at least one positive integer")
    return thresholds


def load_train_records(dataset_root: Path, metadata_filename: str) -> list[dict[str, object]]:
    metadata_path = dataset_root / "train" / metadata_filename
    with metadata_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"expected a JSON list in {metadata_path}")
    return records


def ordered_targets(targets: Iterable[str], support: Counter[str]) -> list[dict[str, object]]:
    return [
        {"ingredient": target, "train_recipe_support": support[target]}
        for target in sorted(targets, key=lambda item: (-support[item], item))
    ]


def cardinality_buckets(recipe_targets: list[list[str]], retained: set[str]) -> dict[str, int]:
    cardinalities = [len(set(targets) & retained) for targets in recipe_targets]
    return {
        "zero": sum(value == 0 for value in cardinalities),
        "one": sum(value == 1 for value in cardinalities),
        "two": sum(value == 2 for value in cardinalities),
        "three_or_more": sum(value >= 3 for value in cardinalities),
    }


def threshold_row(
    threshold: int,
    recipe_targets: list[list[str]],
    support: Counter[str],
    total_assignments: int,
) -> tuple[dict[str, object], set[str]]:
    retained = {target for target, count in support.items() if count >= threshold}
    retained_assignments = sum(support[target] for target in retained)
    buckets = cardinality_buckets(recipe_targets, retained)
    return (
        {
            "minimum_train_recipe_support": threshold,
            "vocabulary_size": len(retained),
            "retained_recipe_target_assignments": retained_assignments,
            "retained_assignment_fraction": retained_assignments / total_assignments,
            "recipes": buckets,
            "recipes_retained_if_minimum_targets_is_1": len(recipe_targets) - buckets["zero"],
            "recipes_retained_if_minimum_targets_is_2": buckets["two"] + buckets["three_or_more"],
            "recipes_retained_if_minimum_targets_is_3": buckets["three_or_more"],
        },
        retained,
    )


def build_report(records: list[dict[str, object]], thresholds: tuple[int, ...]) -> dict[str, object]:
    recipe_targets = [normalized_recipe_targets(record.get("ingredients")) for record in records]
    support = Counter(target for targets in recipe_targets for target in targets)
    total_assignments = sum(support.values())

    rows: list[dict[str, object]] = []
    retained_by_threshold: dict[int, set[str]] = {}
    for threshold in thresholds:
        row, retained = threshold_row(threshold, recipe_targets, support, total_assignments)
        rows.append(row)
        retained_by_threshold[threshold] = retained

    transitions = []
    for lower, higher in zip(thresholds, thresholds[1:]):
        removed = retained_by_threshold[lower] - retained_by_threshold[higher]
        transitions.append(
            {
                "from_minimum_support": lower,
                "to_minimum_support": higher,
                "ingredients_removed": ordered_targets(removed, support),
                "removed_ingredient_count": len(removed),
                "removed_assignment_count": sum(support[target] for target in removed),
            }
        )

    return {
        "purpose": "Train-only support-threshold comparison for provisional normalized ingredients_target values.",
        "method": {
            "source": "data/input/yummly/train/metadata.json original ingredients field",
            "normalization": "src.data_processing.ingredient_standardization.normalized_recipe_targets",
            "support": "Number of distinct train recipes containing a normalized target.",
            "non_goal": "This output does not select a threshold or generate metadata. Repeat it after the controlled-vocabulary association rules are finalized.",
        },
        "corpus": {
            "train_recipes": len(records),
            "normalized_ingredients_before_threshold": len(support),
            "recipe_target_assignments_before_threshold": total_assignments,
        },
        "thresholds": rows,
        "transitions": transitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/input/yummly"))
    parser.add_argument("--metadata-filename", default="metadata.json")
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated minimum train-recipe supports (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src_scratches/data_anlysis/outputs/ingredient_threshold_sweep.json"),
    )
    args = parser.parse_args()

    report = build_report(load_train_records(args.dataset_root, args.metadata_filename), args.thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
