"""Audit a Yummly ``ingredients_target`` metadata generation without changing it.

The outputs are aggregate research artifacts.  In particular, this script does
not persist a raw-line-to-target mapping and does not propose automatic merges.
"""

from __future__ import annotations

import argparse
import csv
import difflib
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
DEFAULT_METADATA = "ingredients_target_v1_metadata.json"

_SINGULAR_WORDS = {
    "berries": "berry",
    "chilies": "chili",
    "chillies": "chili",
    "chiles": "chile",
    "leaves": "leaf",
    "loaves": "loaf",
    "potatoes": "potato",
    "tomatoes": "tomato",
    "yolks": "yolk",
}

_KNOWN_COLLISIONS = (
    {"raw": "1 pineapple", "expected": "pineapple", "forbidden": "apple"},
    {"raw": "1 butternut squash", "expected": "butternut squash", "forbidden": "butter"},
    {"raw": "1 pepperoni", "expected": "pepperoni", "forbidden": "pepper"},
    {"raw": "1 cup watercress", "expected": "watercress", "forbidden": "water"},
)

# Discussion aids only.  These counterfactuals quantify provisional review
# packages; they are never imported by the production standardizer.
_CONSERVATIVE_REVIEW_MAPPING = {
    "basil leaves": "basil",
    "bay leaves": "bay leaf",
    "cayenne": "cayenne pepper",
    "coarse salt": "salt",
    "cold water": "water",
    "cooking oil": "oil",
    "cider vinegar": "apple cider vinegar",
    "flat leaf parsley": "parsley",
    "garlic paste": "garlic",
    "ginger root": "ginger",
    "granulated sugar": "sugar",
    "light brown sugar": "brown sugar",
    "low sodium chicken broth": "chicken broth",
    "low sodium soy sauce": "soy sauce",
    "mint leaves": "mint",
    "plain greek yogurt": "greek yogurt",
    "plain yogurt": "yogurt",
    "purple onion": "red onion",
    "spring onions": "green onion",
    "thyme leaves": "thyme",
    "warm water": "water",
    "white sugar": "sugar",
    "whole milk": "milk",
    "yoghurt": "yogurt",
}

_FORM_GRANULARITY_MAPPING = {
    "chicken stock": "chicken broth",
    "coriander powder": "coriander",
    "greek yogurt": "yogurt",
    "ground cinnamon": "cinnamon",
    "ground coriander": "coriander",
    "ground cumin": "cumin",
    "ground ginger": "ginger",
    "ground nutmeg": "nutmeg",
    "ground turmeric": "turmeric",
    "light soy sauce": "soy sauce",
    "toasted sesame oil": "sesame oil",
}

_TAXONOMY_MAPPING = {
    "baby spinach": "spinach",
    "french bread": "bread",
    "red onion": "onion",
    "romaine lettuce": "lettuce",
    "spanish onion": "onion",
    "sweet onion": "onion",
    "white onion": "onion",
    "yellow onion": "onion",
}


def load_generation(
    dataset_root: Path,
    metadata_filename: str,
    *,
    require_targets: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[object] = set()
    for split in SPLITS:
        path = dataset_root / split / metadata_filename
        with path.open(encoding="utf-8") as metadata_file:
            records = json.load(metadata_file)
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON list")
        for record in records:
            record_id = record.get("id")
            if record_id in seen_ids:
                raise ValueError(f"duplicate record id across splits: {record_id!r}")
            seen_ids.add(record_id)
            if require_targets:
                targets = record.get("ingredients_target")
                if not isinstance(targets, list) or targets != sorted(set(targets)):
                    raise ValueError(f"invalid ingredients_target for record {record_id!r}")
        records_by_split[split] = records
    return records_by_split


def percentile(values: list[int], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def tokenise(value: str) -> tuple[str, ...]:
    return tuple(value.casefold().replace("-", " ").split())


def singularise_token(token: str) -> str:
    if token in _SINGULAR_WORDS:
        return _SINGULAR_WORDS[token]
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "ous")) and len(token) > 3:
        return token[:-1]
    return token


def singular_signature(value: str) -> tuple[str, ...]:
    return tuple(singularise_token(token) for token in tokenise(value))


def contains_phrase(container: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    if len(phrase) >= len(container):
        return False
    return any(container[index : index + len(phrase)] == phrase for index in range(len(container) - len(phrase) + 1))


def lexical_relationships(vocabulary: Iterable[str]) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    for left, right in combinations(sorted(vocabulary), 2):
        left_tokens = tokenise(left)
        right_tokens = tokenise(right)
        relation = None
        base = None
        specific = None
        similarity = difflib.SequenceMatcher(None, left, right).ratio()
        if singular_signature(left) == singular_signature(right):
            relation = "singular_plural_variant"
        elif contains_phrase(right_tokens, left_tokens):
            relation, base, specific = "token_phrase_containment", left, right
        elif contains_phrase(left_tokens, right_tokens):
            relation, base, specific = "token_phrase_containment", right, left
        elif similarity >= 0.84 and abs(len(left) - len(right)) <= 3:
            relation = "near_spelling"
        if relation:
            relationships.append(
                {
                    "left": left,
                    "right": right,
                    "relation": relation,
                    "base": base or "",
                    "specific": specific or "",
                    "string_similarity": round(similarity, 6),
                }
            )
    return relationships


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_pair_views(pair_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    supported = [row for row in pair_rows if row["cooccurrence"] >= 50]
    return {
        "jaccard_min_50": sorted(
            supported,
            key=lambda row: (-row["jaccard"], -row["cooccurrence"], row["left"], row["right"]),
        )[:40],
        "conditional_min_50": sorted(
            supported,
            key=lambda row: (-max(row["p_right_given_left"], row["p_left_given_right"]), -row["cooccurrence"]),
        )[:40],
        "lift_min_50": sorted(
            supported,
            key=lambda row: (-row["lift"], -row["cooccurrence"], row["left"], row["right"]),
        )[:40],
    }


def simulate_review_scenario(
    records: list[dict[str, Any]],
    mapping: dict[str, str],
    *,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    """Measure a provisional review package without modifying any record."""
    exclude = exclude or set()
    before_lists = [set(record["ingredients_target"]) for record in records]
    after_lists = []
    affected_records = 0
    for before in before_lists:
        after: set[str] = set()
        for target in before:
            if target in exclude:
                continue
            if target == "salt and pepper":
                after.update(("salt", "pepper"))
            else:
                after.add(mapping.get(target, target))
        after_lists.append(after)
        affected_records += after != before
    before_vocabulary = set().union(*before_lists)
    after_vocabulary = set().union(*after_lists)
    return {
        "vocabulary_before": len(before_vocabulary),
        "vocabulary_after": len(after_vocabulary),
        "vocabulary_reduction": len(before_vocabulary) - len(after_vocabulary),
        "affected_records": affected_records,
        "target_assignments_before": sum(map(len, before_lists)),
        "target_assignments_after": sum(map(len, after_lists)),
        "records_below_three_targets": sum(len(targets) < 3 for targets in after_lists),
        "excluded_targets": sorted(exclude),
    }


def audit(
    records_by_split: dict[str, list[dict[str, Any]]],
    source_records_by_split: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_records = [record for split in SPLITS for record in records_by_split[split]]
    support_by_split = {
        split: Counter(target for record in records for target in record["ingredients_target"])
        for split, records in records_by_split.items()
    }
    total_support = sum(support_by_split.values(), Counter())
    vocabulary = sorted(total_support)
    cardinalities = [len(record["ingredients_target"]) for record in all_records]
    cardinality_histogram = Counter(cardinalities)

    pair_counts: Counter[tuple[str, str]] = Counter()
    source_variants: dict[str, Counter[str]] = defaultdict(Counter)
    attributed_records = Counter()
    unattributed_records = Counter()
    output_line_counts = Counter()
    mixed_target_categories: dict[str, Counter[str]] = {
        "coriander": Counter(),
        "red pepper": Counter(),
    }
    mixed_target_multi_category_records = Counter()
    raw_line_count = 0

    source_recipe_support = Counter()
    source_record_count = 0
    if source_records_by_split is not None:
        source_records = [record for split in SPLITS for record in source_records_by_split[split]]
        source_record_count = len(source_records)
        for record in source_records:
            ingredients = record.get("ingredients")
            normalized_targets = set()
            if isinstance(ingredients, list):
                normalized_targets = {
                    normalized
                    for raw_line in ingredients
                    if (normalized := normalize_ingredient_line(raw_line)) is not None
                }
            source_recipe_support.update(normalized_targets)

    for record in all_records:
        targets = record["ingredients_target"]
        pair_counts.update(combinations(targets, 2))
        normalized_lines: dict[str, list[str]] = defaultdict(list)
        ingredients = record.get("ingredients")
        if isinstance(ingredients, list):
            for raw_line in ingredients:
                raw_line_count += 1
                normalized = normalize_ingredient_line(raw_line)
                if normalized is not None:
                    output_line_counts[normalized] += 1
                    if isinstance(raw_line, str):
                        normalized_lines[normalized].append(raw_line.strip())
        for mixed_target in mixed_target_categories:
            categories = set()
            for raw_line in normalized_lines[mixed_target]:
                lowered = raw_line.casefold()
                if mixed_target == "coriander":
                    if "fresh" in lowered or "leav" in lowered:
                        categories.add("fresh_or_leaf")
                    elif "ground" in lowered or "powder" in lowered:
                        categories.add("ground_or_powder")
                    else:
                        categories.add("generic")
                elif mixed_target == "red pepper":
                    if "bell" in lowered or "capsicum" in lowered:
                        categories.add("bell_or_capsicum")
                    elif any(word in lowered for word in ("ground", "flake", "crush")):
                        categories.add("ground_crushed_or_flaked")
                    else:
                        categories.add("generic")
            mixed_target_categories[mixed_target].update(categories)
            if len(categories) > 1:
                mixed_target_multi_category_records[mixed_target] += 1
        for target in targets:
            if normalized_lines[target]:
                attributed_records[target] += 1
                source_variants[target].update(sorted(set(normalized_lines[target])))
            else:
                unattributed_records[target] += 1

    target_rows = []
    for target in vocabulary:
        support = total_support[target]
        target_rows.append(
            {
                "target": target,
                "support_total": support,
                "prevalence": round(support / len(all_records), 8),
                "support_train": support_by_split["train"][target],
                "support_val": support_by_split["val"][target],
                "support_test": support_by_split["test"][target],
                "source_recipe_support_before_retention": source_recipe_support[target] if source_records_by_split else "",
                "retained_support_fraction": (
                    round(support / source_recipe_support[target], 8)
                    if source_records_by_split and source_recipe_support[target]
                    else ""
                ),
                "attributed_records": attributed_records[target],
                "unattributed_records": unattributed_records[target],
                "distinct_raw_variants": len(source_variants[target]),
                "top_raw_examples": " | ".join(raw for raw, _ in source_variants[target].most_common(5)),
            }
        )
    target_rows.sort(key=lambda row: (-row["support_total"], row["target"]))

    pair_rows = []
    for (left, right), count in pair_counts.items():
        left_support = total_support[left]
        right_support = total_support[right]
        union = left_support + right_support - count
        pair_rows.append(
            {
                "left": left,
                "right": right,
                "support_left": left_support,
                "support_right": right_support,
                "cooccurrence": count,
                "p_right_given_left": round(count / left_support, 8),
                "p_left_given_right": round(count / right_support, 8),
                "jaccard": round(count / union, 8),
                "lift": round((count * len(all_records)) / (left_support * right_support), 8),
            }
        )
    pair_rows.sort(key=lambda row: (-row["cooccurrence"], row["left"], row["right"]))

    lexical_rows = lexical_relationships(vocabulary)
    pair_lookup = {(row["left"], row["right"]): row for row in pair_rows}
    for row in lexical_rows:
        pair = pair_lookup.get((row["left"], row["right"]))
        row.update(
            {
                "support_left": total_support[row["left"]],
                "support_right": total_support[row["right"]],
                "cooccurrence": pair["cooccurrence"] if pair else 0,
                "jaccard": pair["jaccard"] if pair else 0.0,
                "p_right_given_left": pair["p_right_given_left"] if pair else 0.0,
                "p_left_given_right": pair["p_left_given_right"] if pair else 0.0,
            }
        )
    lexical_rows.sort(key=lambda row: (row["relation"], row["left"], row["right"]))

    known_collision_checks = []
    for case in _KNOWN_COLLISIONS:
        observed = normalize_ingredient_line(case["raw"])
        known_collision_checks.append(
            {
                **case,
                "observed": observed,
                "passed": observed == case["expected"] and observed != case["forbidden"],
                "observed_line_count_in_retained_records": output_line_counts[case["expected"]],
            }
        )

    conservative_mapping = dict(_CONSERVATIVE_REVIEW_MAPPING)
    form_mapping = {**conservative_mapping, **_FORM_GRANULARITY_MAPPING}
    taxonomy_mapping = {**form_mapping, **_TAXONOMY_MAPPING}

    split_vocabularies = {
        split: set(counter) for split, counter in support_by_split.items()
    }
    report = {
        "metadata": {
            "records_by_split": {split: len(records_by_split[split]) for split in SPLITS},
            "records_total": len(all_records),
            "source_records_total": source_record_count or None,
            "records_removed_by_target_retention": source_record_count - len(all_records) if source_record_count else None,
            "raw_ingredient_lines_scanned": raw_line_count,
            "vocabulary_size": len(vocabulary),
            "vocabulary_missing_from_train": sorted((split_vocabularies["val"] | split_vocabularies["test"]) - split_vocabularies["train"]),
            "vocabulary_missing_from_val": sorted(split_vocabularies["train"] - split_vocabularies["val"]),
            "vocabulary_missing_from_test": sorted(split_vocabularies["train"] - split_vocabularies["test"]),
        },
        "target_cardinality": {
            "minimum": min(cardinalities),
            "maximum": max(cardinalities),
            "mean": round(statistics.fmean(cardinalities), 6),
            "median": statistics.median(cardinalities),
            "p10": round(percentile(cardinalities, 0.10), 6),
            "p90": round(percentile(cardinalities, 0.90), 6),
            "histogram": {str(key): cardinality_histogram[key] for key in sorted(cardinality_histogram)},
        },
        "support_profile": {
            "minimum_post_retention_support": min(total_support.values()),
            "maximum_support": max(total_support.values()),
            "median_support": statistics.median(total_support.values()),
            "targets_below_500_after_record_retention": sorted(
                ({"target": target, "support": count} for target, count in total_support.items() if count < 500),
                key=lambda item: (item["support"], item["target"]),
            ),
            "targets_below_1000": sum(count < 1000 for count in total_support.values()),
            "targets_at_least_5000": sum(count >= 5000 for count in total_support.values()),
        },
        "source_traceability": {
            "target_assignments": sum(total_support.values()),
            "assignments_with_matching_normalized_raw_line": sum(attributed_records.values()),
            "assignments_without_matching_normalized_raw_line": sum(unattributed_records.values()),
            "top_raw_examples_by_target": {
                target: [
                    {"raw": raw, "record_support": count}
                    for raw, count in source_variants[target].most_common(5)
                ]
                for target in vocabulary
            },
        },
        "pair_relationship_views": top_pair_views(pair_rows),
        "lexical_relationship_counts": dict(Counter(row["relation"] for row in lexical_rows)),
        "known_legacy_collision_checks": known_collision_checks,
        "mixed_target_diagnostics": {
            target: {
                "retained_support": total_support[target],
                "category_record_counts": dict(sorted(category_counts.items())),
                "records_in_multiple_categories": mixed_target_multi_category_records[target],
            }
            for target, category_counts in mixed_target_categories.items()
        },
        "provisional_review_scenarios": {
            "conservative_normalization": simulate_review_scenario(all_records, conservative_mapping),
            "conservative_plus_generic_sauce_exclusion": simulate_review_scenario(
                all_records, conservative_mapping, exclude={"sauce"}
            ),
            "plus_form_granularity": simulate_review_scenario(all_records, form_mapping),
            "plus_taxonomy_collapse": simulate_review_scenario(all_records, taxonomy_mapping),
            "plus_generic_sauce_exclusion": simulate_review_scenario(
                all_records, taxonomy_mapping, exclude={"sauce"}
            ),
            "warning": (
                "These are counterfactual discussion aids, not approved rules. Mixed raw-line targets such as "
                "coriander and red pepper require line-level rule replacement and are not represented here."
            ),
        },
    }
    return report, target_rows, pair_rows, lexical_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/input/yummly"))
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--source-metadata", default="metadata.json")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("src_scratches/data_anlysis/outputs/ingredient_vocabulary_audit")
    )
    args = parser.parse_args()

    records_by_split = load_generation(args.dataset_root.resolve(), args.metadata)
    source_records_by_split = load_generation(
        args.dataset_root.resolve(), args.source_metadata, require_targets=False
    )
    report, target_rows, pair_rows, lexical_rows = audit(records_by_split, source_records_by_split)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "target_support.csv", target_rows, list(target_rows[0]))
    write_csv(output_dir / "pair_relationships.csv", pair_rows, list(pair_rows[0]))
    write_csv(output_dir / "lexical_relationships.csv", lexical_rows, list(lexical_rows[0]))

    print("records:", report["metadata"]["records_by_split"])
    print("vocabulary size:", report["metadata"]["vocabulary_size"])
    print("cardinality:", report["target_cardinality"])
    print("post-retention support:", report["support_profile"])
    print("lexical relationships:", report["lexical_relationship_counts"])
    print("outputs:", output_dir)


if __name__ == "__main__":
    main()
