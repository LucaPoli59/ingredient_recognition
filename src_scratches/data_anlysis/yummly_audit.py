"""Reproducible audit of the Yummly data used by this repository.

The script reads the source and processed Yummly trees without modifying them.
It writes compact machine-readable results and a deterministic image contact
sheet under ``src_scratches/data_anlysis/outputs``.

Run from the repository root with::

    python src_scratches/data_anlysis/yummly_audit.py

Use ``--skip-images`` for a metadata-only audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
YUMMLY_INPUT = REPOSITORY_ROOT / "data" / "input" / "yummly"
YUMMLY_RAW = REPOSITORY_ROOT / "data" / "raw_input" / "yummly"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SPLITS = ("train", "val", "test")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def percentile(values: list[float] | list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))


def distribution(values: Iterable[float | int]) -> dict[str, float | int | None]:
    values = list(values)
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None, "mean": None}
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


def normalized_text(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(value.casefold()))


def canonical_name(value: str) -> str:
    return "".join(TOKEN_PATTERN.findall(value.casefold()))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def counter_rows(counter: Counter, limit: int | None = None) -> list[dict[str, Any]]:
    rows = counter.most_common(limit)
    return [{"value": key, "count": value} for key, value in rows]


def field_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = sorted({key for record in records for key in record})
    output = {}
    for field in fields:
        present = sum(field in record for record in records)
        null = sum(record.get(field) is None for record in records if field in record)
        types = Counter(type(record[field]).__name__ for record in records if field in record)
        output[field] = {
            "present": present,
            "missing": len(records) - present,
            "null": null,
            "types": dict(sorted(types.items())),
        }
    return output


def intersection_examples(groups: dict[Any, list[tuple[str, dict[str, Any]]]], limit: int = 20) -> list[dict[str, Any]]:
    examples = []
    for key, entries in groups.items():
        splits = sorted({split for split, _ in entries})
        if len(splits) <= 1:
            continue
        examples.append(
            {
                "key": str(key),
                "splits": splits,
                "records": [
                    {
                        "split": split,
                        "id": record.get("id"),
                        "old_id": record.get("old_id"),
                        "name": record.get("name"),
                        "image": record.get("image"),
                    }
                    for split, record in entries[:5]
                ],
            }
        )
        if len(examples) >= limit:
            break
    return examples


def cross_split_group_count(groups: dict[Any, list[tuple[str, dict[str, Any]]]]) -> int:
    return sum(len({split for split, _ in entries}) > 1 for entries in groups.values())


def metadata_audit(split_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    combined = [(split, record) for split in SPLITS for record in split_records[split]]
    records = [record for _, record in combined]

    ingredient_frequency: Counter[str] = Counter()
    raw_ingredient_frequency: Counter[str] = Counter()
    cuisine_frequency: Counter[str] = Counter()
    course_frequency: Counter[str] = Counter()
    label_set_frequency: Counter[tuple[str, ...]] = Counter()
    label_text_support: Counter[str] = Counter()
    label_token_support: Counter[str] = Counter()
    label_occurrences: Counter[str] = Counter()
    cuisine_labels: dict[str, Counter[str]] = defaultdict(Counter)
    ingredient_pairs: Counter[tuple[str, str]] = Counter()
    flavor_values: dict[str, list[float]] = defaultdict(list)
    flavor_missing = 0
    empty_flavors = 0
    flavor_complete_numeric = 0
    flavor_keysets: Counter[tuple[str, ...]] = Counter()
    flavor_value_types: Counter[str] = Counter()

    label_counts_per_recipe = []
    raw_counts_per_recipe = []
    course_counts_per_recipe = []
    duplicate_normalized_labels = 0
    duplicate_raw_lines = 0
    empty_label_records = []
    malformed_records = []

    old_id_groups: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    name_groups: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    label_set_groups: dict[tuple[str, ...], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    raw_ingredient_groups: dict[tuple[str, ...], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    name_and_ingredient_groups: dict[tuple[str, tuple[str, ...]], list[tuple[str, dict[str, Any]]]] = defaultdict(list)

    collision_rules = {
        "liquor_from_ginger": ("liquor", re.compile(r"\bginger\b", re.IGNORECASE)),
        "liquor_from_portobello": ("liquor", re.compile(r"\bportobellos?\b", re.IGNORECASE)),
        "liquor_from_crumb": ("liquor", re.compile(r"\b\w*crumb\w*\b", re.IGNORECASE)),
        "cheese_from_cantaloupe": ("cheese", re.compile(r"\bcantaloupes?\b", re.IGNORECASE)),
        "cheese_from_jackfruit": ("cheese", re.compile(r"\bjackfruits?\b", re.IGNORECASE)),
        "egg_from_eggplant": ("egg", re.compile(r"\beggplants?\b", re.IGNORECASE)),
        "sage_from_sausage": ("sage", re.compile(r"\bsausages?\b", re.IGNORECASE)),
        "tea_from_steak": ("tea", re.compile(r"\bsteaks?\b", re.IGNORECASE)),
        "oil_from_boiling": ("oil", re.compile(r"\bboil\w*\b", re.IGNORECASE)),
        "rice_from_licorice": ("rice", re.compile(r"\blicorice\b", re.IGNORECASE)),
        "oat_from_coating": ("oat", re.compile(r"\bcoat\w*\b", re.IGNORECASE)),
    }
    collision_counts = Counter()
    collision_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for split, record in combined:
        required = ("id", "old_id", "image", "name", "cuisine", "ingredients", "ingredients_ok")
        missing = [field for field in required if field not in record]
        if missing:
            malformed_records.append({"split": split, "id": record.get("id"), "missing": missing})

        cuisine = str(record.get("cuisine", "<missing>")).casefold()
        cuisine_frequency[cuisine] += 1
        courses = record.get("course") or []
        for course in courses:
            course_frequency[str(course)] += 1
        course_counts_per_recipe.append(len(courses))

        labels = [str(label) for label in (record.get("ingredients_ok") or [])]
        raw_lines = [str(line) for line in (record.get("ingredients") or [])]
        label_counts_per_recipe.append(len(labels))
        raw_counts_per_recipe.append(len(raw_lines))
        duplicate_normalized_labels += len(labels) - len(set(labels))
        duplicate_raw_lines += len(raw_lines) - len(set(raw_lines))
        if not labels:
            empty_label_records.append({"split": split, "id": record.get("id"), "name": record.get("name")})

        ingredient_frequency.update(labels)
        raw_ingredient_frequency.update(raw_lines)
        cuisine_labels[cuisine].update(set(labels))
        label_set = tuple(sorted(set(labels)))
        label_set_frequency[label_set] += 1
        label_set_groups[label_set].append((split, record))
        canonical_raw = tuple(sorted(normalized_text(line) for line in raw_lines))
        raw_ingredient_groups[canonical_raw].append((split, record))
        name_and_ingredient_groups[(canonical_name(str(record.get("name", ""))), canonical_raw)].append((split, record))
        for index, left in enumerate(label_set):
            for right in label_set[index + 1 :]:
                ingredient_pairs[(left, right)] += 1

        raw_text = " ".join(normalized_text(line) for line in raw_lines)
        padded_raw_text = f" {raw_text} "
        for label in labels:
            label_occurrences[label] += 1
            normalized_label = normalized_text(label)
            if normalized_label and normalized_label in raw_text:
                label_text_support[label] += 1
            if normalized_label and f" {normalized_label} " in padded_raw_text:
                label_token_support[label] += 1

        original_raw_text = " | ".join(raw_lines)
        for rule_name, (target, pattern) in collision_rules.items():
            if target in labels and pattern.search(original_raw_text):
                collision_counts[rule_name] += 1
                if len(collision_examples[rule_name]) < 10:
                    collision_examples[rule_name].append(
                        {"split": split, "id": record.get("id"), "name": record.get("name"), "ingredients": raw_lines, "labels": labels}
                    )

        flavors = record.get("flavors")
        if flavors is None:
            flavor_missing += 1
            flavor_keysets[("<null>",)] += 1
        elif not flavors:
            empty_flavors += 1
            flavor_keysets[("<empty>",)] += 1
        elif isinstance(flavors, dict):
            flavor_keysets[tuple(flavors)] += 1
            normalized_values = {str(name).casefold(): value for name, value in flavors.items()}
            if set(normalized_values) == {"bitter", "meaty", "piquant", "salty", "sour", "sweet"} and all(
                isinstance(value, (int, float)) for value in normalized_values.values()
            ):
                flavor_complete_numeric += 1
            for name, value in flavors.items():
                flavor_value_types[type(value).__name__] += 1
                if isinstance(value, (int, float)):
                    flavor_values[str(name).casefold()].append(float(value))

        old_id_groups[str(record.get("old_id"))].append((split, record))
        name_groups[canonical_name(str(record.get("name", "")))].append((split, record))

    split_label_frequencies: dict[str, Counter[str]] = {}
    split_cuisine_frequencies: dict[str, Counter[str]] = {}
    split_summaries = {}
    for split, current in split_records.items():
        labels = Counter(label for record in current for label in (record.get("ingredients_ok") or []))
        cuisines = Counter(str(record.get("cuisine", "<missing>")).casefold() for record in current)
        split_label_frequencies[split] = labels
        split_cuisine_frequencies[split] = cuisines
        split_summaries[split] = {
            "records": len(current),
            "fields": field_audit(current),
            "cuisines": dict(sorted(cuisines.items())),
            "label_vocabulary_size": len(labels),
            "label_occurrences": sum(labels.values()),
            "label_cardinality": distribution([len(record.get("ingredients_ok") or []) for record in current]),
            "raw_ingredient_cardinality": distribution([len(record.get("ingredients") or []) for record in current]),
            "course_cardinality": distribution([len(record.get("course") or []) for record in current]),
        }

    train_vocabulary = set(split_label_frequencies["train"])
    oov = {}
    for split in ("val", "test"):
        labels = split_label_frequencies[split]
        unseen = sorted(set(labels) - train_vocabulary)
        oov[split] = {
            "labels": unseen,
            "label_count": len(unseen),
            "occurrences": sum(labels[label] for label in unseen),
            "affected_records": sum(any(label not in train_vocabulary for label in record.get("ingredients_ok", [])) for record in split_records[split]),
        }

    total_records = len(records)
    label_support_rows = []
    for label, count in ingredient_frequency.most_common():
        label_support_rows.append(
            {
                "label": label,
                "count": count,
                "recipe_prevalence": count / total_records,
                "literal_text_support_count": label_text_support[label],
                "literal_text_support_rate": label_text_support[label] / count,
                "token_phrase_support_count": label_token_support[label],
                "token_phrase_support_rate": label_token_support[label] / count,
                "train": split_label_frequencies["train"][label],
                "val": split_label_frequencies["val"][label],
                "test": split_label_frequencies["test"][label],
            }
        )

    top_k_coverage = {}
    train_ranked = [label for label, _ in split_label_frequencies["train"].most_common()]
    for k in (20, 50, 100, 200):
        vocabulary = set(train_ranked[:k])
        remaining = [len(set(record.get("ingredients_ok") or []) & vocabulary) for record in records]
        retained_occurrences = sum(remaining)
        top_k_coverage[str(k)] = {
            "label_occurrence_coverage": retained_occurrences / max(1, sum(label_counts_per_recipe)),
            "records_with_at_least_one_label": sum(value >= 1 for value in remaining),
            "records_with_at_least_three_labels": sum(value >= 3 for value in remaining),
            "remaining_cardinality": distribution(remaining),
        }

    distinctive_by_cuisine = {}
    global_prevalence = {label: count / total_records for label, count in ingredient_frequency.items()}
    for cuisine, cuisine_count in cuisine_frequency.items():
        candidates = []
        for label, count in cuisine_labels[cuisine].items():
            if count < 20:
                continue
            within = count / cuisine_count
            lift = within / global_prevalence[label]
            candidates.append({"label": label, "count": count, "within_cuisine_prevalence": within, "lift": lift})
        distinctive_by_cuisine[cuisine] = sorted(candidates, key=lambda item: (-item["lift"], -item["count"]))[:15]

    split_drift = {}
    all_cuisines = set(cuisine_frequency)
    all_labels = set(ingredient_frequency)
    for split in ("val", "test"):
        cuisine_diffs = []
        for cuisine in all_cuisines:
            train_p = split_cuisine_frequencies["train"][cuisine] / len(split_records["train"])
            split_p = split_cuisine_frequencies[split][cuisine] / len(split_records[split])
            cuisine_diffs.append((abs(split_p - train_p), cuisine, train_p, split_p))
        label_diffs = []
        train_n = len(split_records["train"])
        split_n = len(split_records[split])
        for label in all_labels:
            train_p = split_label_frequencies["train"][label] / train_n
            split_p = split_label_frequencies[split][label] / split_n
            label_diffs.append((abs(split_p - train_p), label, train_p, split_p))
        split_drift[split] = {
            "cuisine_total_variation_distance": sum(diff for diff, *_ in cuisine_diffs) / 2,
            "largest_cuisine_prevalence_differences": [
                {"cuisine": name, "absolute_difference": diff, "train": train_p, split: split_p}
                for diff, name, train_p, split_p in sorted(cuisine_diffs, reverse=True)[:10]
            ],
            "largest_label_prevalence_differences": [
                {"label": name, "absolute_difference": diff, "train": train_p, split: split_p}
                for diff, name, train_p, split_p in sorted(label_diffs, reverse=True)[:20]
            ],
        }

    def evaluate_constant_predictions(prediction_builder) -> dict[str, Any]:
        true_positive = false_positive = false_negative = true_negative = 0
        sample_precision = []
        sample_recall = []
        sample_f1 = []
        exact_matches = 0
        for record in split_records["test"]:
            actual = set(record.get("ingredients_ok") or [])
            predicted = set(prediction_builder(record))
            tp = len(actual & predicted)
            fp = len(predicted - actual)
            fn = len(actual - predicted)
            tn = len(ingredient_frequency) - tp - fp - fn
            true_positive += tp
            false_positive += fp
            false_negative += fn
            true_negative += tn
            precision = tp / len(predicted) if predicted else 0.0
            recall = tp / len(actual) if actual else 0.0
            sample_precision.append(precision)
            sample_recall.append(recall)
            sample_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
            exact_matches += actual == predicted
        return {
            "micro_precision": true_positive / max(1, true_positive + false_positive),
            "micro_recall": true_positive / max(1, true_positive + false_negative),
            "micro_f1": 2 * true_positive / max(1, 2 * true_positive + false_positive + false_negative),
            "mean_sample_precision": statistics.fmean(sample_precision),
            "mean_sample_recall": statistics.fmean(sample_recall),
            "mean_sample_f1": statistics.fmean(sample_f1),
            "label_accuracy": (true_positive + true_negative) / max(1, true_positive + true_negative + false_positive + false_negative),
            "hamming_loss": (false_positive + false_negative) / max(1, true_positive + true_negative + false_positive + false_negative),
            "exact_match_ratio": exact_matches / len(split_records["test"]),
        }

    prior_baselines = {"all_negative": evaluate_constant_predictions(lambda _: set())}
    train_cuisine_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for record in split_records["train"]:
        train_cuisine_labels[str(record.get("cuisine", "<missing>")).casefold()].update(set(record.get("ingredients_ok") or []))
    for k in (3, 5, 9, 10):
        global_top = set(train_ranked[:k])
        cuisine_top = {
            cuisine: set(counter.most_common(k)[index][0] for index in range(min(k, len(counter))))
            for cuisine, counter in train_cuisine_labels.items()
        }
        prior_baselines[f"global_top_{k}"] = evaluate_constant_predictions(lambda _, labels=global_top: labels)
        prior_baselines[f"cuisine_top_{k}"] = evaluate_constant_predictions(
            lambda record, labels=cuisine_top: labels[str(record.get("cuisine", "<missing>")).casefold()]
        )

    return {
        "total_records": total_records,
        "split_summaries": split_summaries,
        "combined_fields": field_audit(records),
        "cuisines": dict(sorted(cuisine_frequency.items())),
        "courses": counter_rows(course_frequency),
        "course_cardinality": distribution(course_counts_per_recipe),
        "normalized_ingredient_vocabulary_size": len(ingredient_frequency),
        "normalized_ingredient_occurrences": sum(ingredient_frequency.values()),
        "normalized_ingredient_cardinality": distribution(label_counts_per_recipe),
        "raw_ingredient_line_vocabulary_size": len(raw_ingredient_frequency),
        "raw_ingredient_line_occurrences": sum(raw_ingredient_frequency.values()),
        "raw_ingredient_cardinality": distribution(raw_counts_per_recipe),
        "duplicate_normalized_labels_within_records": duplicate_normalized_labels,
        "duplicate_raw_lines_within_records": duplicate_raw_lines,
        "empty_label_records": empty_label_records[:100],
        "malformed_records": malformed_records[:100],
        "ingredient_frequency": label_support_rows,
        "frequency_tail": {
            "singletons": sum(count == 1 for count in ingredient_frequency.values()),
            "frequency_at_most_5": sum(count <= 5 for count in ingredient_frequency.values()),
            "frequency_at_most_10": sum(count <= 10 for count in ingredient_frequency.values()),
            "frequency_at_least_100": sum(count >= 100 for count in ingredient_frequency.values()),
            "frequency_at_least_1000": sum(count >= 1000 for count in ingredient_frequency.values()),
        },
        "top_ingredient_pairs": [
            {"left": pair[0], "right": pair[1], "count": count}
            for pair, count in ingredient_pairs.most_common(100)
        ],
        "top_k_coverage": top_k_coverage,
        "oov_against_train_vocabulary": oov,
        "distinctive_ingredients_by_cuisine": distinctive_by_cuisine,
        "split_drift": split_drift,
        "test_set_prior_baselines": prior_baselines,
        "duplicate_keys": {
            "old_id_duplicate_groups": sum(len(entries) > 1 for entries in old_id_groups.values()),
            "old_id_cross_split_group_count": cross_split_group_count(old_id_groups),
            "old_id_cross_split_examples": intersection_examples(old_id_groups),
            "canonical_name_duplicate_groups": sum(len(entries) > 1 for entries in name_groups.values()),
            "canonical_name_cross_split_group_count": cross_split_group_count(name_groups),
            "canonical_name_cross_split_examples": intersection_examples(name_groups),
            "label_set_duplicate_groups": sum(len(entries) > 1 for entries in label_set_groups.values()),
            "label_set_cross_split_group_count": cross_split_group_count(label_set_groups),
            "label_set_cross_split_examples": intersection_examples(label_set_groups),
            "raw_ingredient_list_duplicate_groups": sum(len(entries) > 1 for entries in raw_ingredient_groups.values()),
            "raw_ingredient_list_cross_split_group_count": cross_split_group_count(raw_ingredient_groups),
            "raw_ingredient_list_cross_split_examples": intersection_examples(raw_ingredient_groups),
            "name_and_raw_ingredient_duplicate_groups": sum(len(entries) > 1 for entries in name_and_ingredient_groups.values()),
            "name_and_raw_ingredient_cross_split_group_count": cross_split_group_count(name_and_ingredient_groups),
            "name_and_raw_ingredient_cross_split_examples": intersection_examples(name_and_ingredient_groups),
        },
        "known_substring_collision_indicators": {
            name: {"count": collision_counts[name], "examples": collision_examples[name]}
            for name in collision_rules
        },
        "flavors": {
            "missing_records": flavor_missing,
            "empty_records": empty_flavors,
            "complete_six_dimension_numeric_records": flavor_complete_numeric,
            "keysets": [{"keys": list(keys), "count": count} for keys, count in flavor_keysets.most_common()],
            "value_types": dict(sorted(flavor_value_types.items())),
            "dimensions": {name: distribution(values) for name, values in sorted(flavor_values.items())},
        },
    }


def raw_alignment_audit(split_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    raw_recipe_files = sorted((YUMMLY_RAW / "recipes").glob("*.json"))
    raw_recipe_counts = {}
    raw_recipes = []
    for path in raw_recipe_files:
        records = load_json(path)
        raw_recipe_counts[path.name] = len(records)
        raw_recipes.extend(records)

    all_recipes_path = YUMMLY_INPUT / "recipes_general" / "all_recipes.json"
    sorted_recipes_path = YUMMLY_INPUT / "recipes_general" / "recipes_sorted.json"
    all_recipes = load_json(all_recipes_path)
    sorted_recipes = load_json(sorted_recipes_path)
    processed = [record for split in SPLITS for record in split_records[split]]

    raw_ids = {str(record.get("id")) for record in all_recipes}
    processed_old_ids = {str(record.get("old_id")) for record in processed}
    sorted_ids = [str(record.get("id")) for record in sorted_recipes]
    all_ids = [str(record.get("id")) for record in all_recipes]

    raw_images_by_cuisine = {}
    raw_image_ids = set()
    for cuisine_dir in sorted((YUMMLY_RAW / "images").iterdir()):
        if not cuisine_dir.is_dir():
            continue
        images = list(cuisine_dir.glob("*.jpg"))
        raw_images_by_cuisine[cuisine_dir.name.casefold()] = len(images)
        raw_image_ids.update(image.stem for image in images)

    raw_metadata_counts = {}
    raw_metadata_ids = set()
    for path in sorted((YUMMLY_RAW / "metadata").glob("*.json")):
        records = load_json(path)
        raw_metadata_counts[path.name] = len(records)
        raw_metadata_ids.update(str(record.get("id")) for record in records)

    raw_not_processed = raw_ids - processed_old_ids
    raw_without_image = raw_ids - raw_image_ids
    raw_records_by_id = {str(record.get("id")): record for record in all_recipes}
    return {
        "raw_recipe_files": raw_recipe_counts,
        "raw_recipe_records_total": len(raw_recipes),
        "all_recipes_records": len(all_recipes),
        "sorted_recipes_records": len(sorted_recipes),
        "all_recipes_same_order_as_sorted": all_ids == sorted_ids,
        "unique_raw_recipe_ids": len(raw_ids),
        "duplicate_raw_recipe_ids": len(all_ids) - len(set(all_ids)),
        "processed_records": len(processed),
        "unique_processed_old_ids": len(processed_old_ids),
        "raw_recipe_ids_not_processed": len(raw_ids - processed_old_ids),
        "raw_recipe_ids_not_processed_examples": [raw_records_by_id[value] for value in sorted(raw_not_processed)[:30]],
        "processed_old_ids_not_in_raw_recipes": len(processed_old_ids - raw_ids),
        "raw_images_by_cuisine": raw_images_by_cuisine,
        "raw_images_total": sum(raw_images_by_cuisine.values()),
        "unique_raw_image_ids": len(raw_image_ids),
        "raw_recipe_ids_without_raw_image": len(raw_ids - raw_image_ids),
        "raw_recipe_ids_without_raw_image_examples": [raw_records_by_id[value] for value in sorted(raw_without_image)[:30]],
        "raw_image_ids_without_raw_recipe": len(raw_image_ids - raw_ids),
        "raw_metadata_files": raw_metadata_counts,
        "raw_metadata_records_total": sum(raw_metadata_counts.values()),
        "unique_raw_metadata_ids": len(raw_metadata_ids),
        "raw_recipe_ids_without_metadata": len(raw_ids - raw_metadata_ids),
        "metadata_ids_without_raw_recipe": len(raw_metadata_ids - raw_ids),
    }


def difference_hash(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    bits = []
    for row in range(8):
        offset = row * 9
        bits.extend(pixels[offset + column] > pixels[offset + column + 1] for column in range(8))
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:016x}"


def image_audit(split_records: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any], Path]]]:
    widths = []
    heights = []
    aspect_ratios = []
    byte_sizes = []
    dimensions = Counter()
    modes = Counter()
    formats = Counter()
    orientations = Counter()
    missing = []
    decode_errors = []
    exact_hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    perceptual_hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    available_samples = []
    metadata_filenames_by_split = {}
    extra_files_by_split = {}
    extra_file_counts_by_split = {}

    for split in SPLITS:
        split_dir = YUMMLY_INPUT / split
        metadata_filenames = {str(record.get("image")) for record in split_records[split]}
        disk_filenames = {path.name for path in split_dir.glob("*.jpg")}
        metadata_filenames_by_split[split] = len(metadata_filenames)
        extra_files_by_split[split] = sorted(disk_filenames - metadata_filenames)[:100]
        extra_file_counts_by_split[split] = len(disk_filenames - metadata_filenames)

        for record in split_records[split]:
            path = split_dir / str(record.get("image"))
            if not path.is_file():
                missing.append({"split": split, "id": record.get("id"), "image": record.get("image")})
                continue
            byte_sizes.append(path.stat().st_size)
            try:
                content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    widths.append(width)
                    heights.append(height)
                    aspect_ratios.append(width / height)
                    dimensions[(width, height)] += 1
                    modes[image.mode] += 1
                    formats[image.format or "<unknown>"] += 1
                    orientations[str(image.getexif().get(274, "<missing>"))] += 1
                    perceptual_hash = difference_hash(image)
                entry = {
                    "split": split,
                    "id": record.get("id"),
                    "old_id": record.get("old_id"),
                    "name": record.get("name"),
                    "image": record.get("image"),
                    "cuisine": record.get("cuisine"),
                    "labels": sorted(set(record.get("ingredients_ok") or [])),
                }
                exact_hash_groups[content_hash].append(entry)
                perceptual_hash_groups[perceptual_hash].append(entry)
                available_samples.append((split, record, path))
            except Exception as error:  # audit must record corrupt data rather than abort
                decode_errors.append({"split": split, "id": record.get("id"), "image": record.get("image"), "error": repr(error)})

    def duplicate_summary(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        duplicates = [(key, entries) for key, entries in groups.items() if len(entries) > 1]
        cross_split = [(key, entries) for key, entries in duplicates if len({entry["split"] for entry in entries}) > 1]
        ordered = sorted(duplicates, key=lambda item: len(item[1]), reverse=True)
        conflicting = [(key, entries) for key, entries in duplicates if len({tuple(entry["labels"]) for entry in entries}) > 1]
        majority_label_disagreements = 0
        for _, entries in conflicting:
            vocabulary = {label for entry in entries for label in entry["labels"]}
            for label in vocabulary:
                positives = sum(label in entry["labels"] for entry in entries)
                majority_label_disagreements += min(positives, len(entries) - positives)
        evaluation_leakage = {}
        train_hashes = {key for key, entries in groups.items() if any(entry["split"] == "train" for entry in entries)}
        for split in ("val", "test"):
            leaked_records = []
            same_label_set = 0
            same_name = 0
            for key in train_hashes:
                entries = groups[key]
                train_entries = [entry for entry in entries if entry["split"] == "train"]
                for entry in (candidate for candidate in entries if candidate["split"] == split):
                    leaked_records.append(entry)
                    same_label_set += any(train["labels"] == entry["labels"] for train in train_entries)
                    same_name += any(canonical_name(str(train["name"])) == canonical_name(str(entry["name"])) for train in train_entries)
            evaluation_leakage[split] = {
                "records_with_hash_seen_in_train": len(leaked_records),
                "fraction_of_split": len(leaked_records) / len(split_records[split]),
                "records_matching_a_train_label_set": same_label_set,
                "records_matching_a_train_canonical_name": same_name,
            }
        return {
            "duplicate_groups": len(duplicates),
            "records_in_duplicate_groups": sum(len(entries) for _, entries in duplicates),
            "cross_split_groups": len(cross_split),
            "largest_group_size": max((len(entries) for _, entries in duplicates), default=0),
            "groups_with_multiple_label_sets": sum(len({tuple(entry["labels"]) for entry in entries}) > 1 for _, entries in duplicates),
            "records_in_groups_with_multiple_label_sets": sum(len(entries) for _, entries in conflicting),
            "minimum_majority_label_disagreements_within_duplicate_groups": majority_label_disagreements,
            "groups_with_multiple_cuisines": sum(len({entry["cuisine"] for entry in entries}) > 1 for _, entries in duplicates),
            "evaluation_records_with_train_duplicate": evaluation_leakage,
            "largest_groups": [{"hash": key, "records": entries[:30], "group_size": len(entries)} for key, entries in ordered[:20]],
            "cross_split_examples": [{"hash": key, "records": entries[:10]} for key, entries in cross_split[:30]],
        }

    return (
        {
            "audited_files": len(widths),
            "metadata_image_references_by_split": metadata_filenames_by_split,
            "missing_files": missing,
            "extra_jpg_files_not_in_metadata_by_split": extra_files_by_split,
            "extra_jpg_file_counts_by_split": extra_file_counts_by_split,
            "decode_errors": decode_errors,
            "width": distribution(widths),
            "height": distribution(heights),
            "aspect_ratio_width_over_height": distribution(aspect_ratios),
            "byte_size": distribution(byte_sizes),
            "orientation_counts": {
                "landscape": sum(width > height for width, height in zip(widths, heights)),
                "square": sum(width == height for width, height in zip(widths, heights)),
                "portrait": sum(width < height for width, height in zip(widths, heights)),
            },
            "images_smaller_than_224_on_any_side": sum(width < 224 or height < 224 for width, height in zip(widths, heights)),
            "dimensions": [
                {"width": size[0], "height": size[1], "count": count}
                for size, count in dimensions.most_common()
            ],
            "modes": dict(sorted(modes.items())),
            "formats": dict(sorted(formats.items())),
            "exif_orientation": dict(sorted(orientations.items())),
            "exact_content_duplicates": duplicate_summary(exact_hash_groups),
            "identical_difference_hash_candidates": duplicate_summary(perceptual_hash_groups),
        },
        available_samples,
    )


def create_contact_sheet(samples: list[tuple[str, dict[str, Any], Path]], output_path: Path) -> None:
    by_cuisine: dict[str, list[tuple[str, dict[str, Any], Path]]] = defaultdict(list)
    for sample in samples:
        cuisine = str(sample[1].get("cuisine", "unknown")).casefold()
        by_cuisine[cuisine].append(sample)

    selected = []
    for cuisine in sorted(by_cuisine):
        candidates = by_cuisine[cuisine]
        indices = sorted({0, len(candidates) // 2, len(candidates) - 1})
        selected.extend(candidates[index] for index in indices)

    cell_width, image_height, caption_height = 260, 190, 58
    columns = 3
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * (image_height + caption_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (split, record, path) in enumerate(selected):
        x = (index % columns) * cell_width
        y = (index // columns) * (image_height + caption_height)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((cell_width, image_height), Image.Resampling.LANCZOS)
            offset_x = x + (cell_width - image.width) // 2
            offset_y = y + (image_height - image.height) // 2
            sheet.paste(image, (offset_x, offset_y))
        cuisine = str(record.get("cuisine", "unknown"))
        name = str(record.get("name", ""))[:38]
        caption = f"{cuisine} | {split}\n{name}\n{record.get('image')}"
        draw.multiline_text((x + 4, y + image_height + 3), caption, fill="black", font=font, spacing=2)
    sheet.save(output_path, quality=92)


def write_label_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-images", action="store_true", help="Skip decoding and hashing the processed images")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_records = {split: load_json(YUMMLY_INPUT / split / "metadata.json") for split in SPLITS}

    report_path = OUTPUT_DIR / "yummly_audit.json"
    metadata = metadata_audit(split_records)
    report: dict[str, Any] = {
        "dataset": "Yummly",
        "repository_root": ".",
        "input_path": "data/input/yummly",
        "raw_path": "data/raw_input/yummly",
        "metadata": metadata,
        "raw_alignment": raw_alignment_audit(split_records),
    }

    if not args.skip_images:
        images, samples = image_audit(split_records)
        report["images"] = images
        create_contact_sheet(samples, OUTPUT_DIR / "yummly_sample_contact_sheet.jpg")
    elif report_path.exists():
        previous_report = load_json(report_path)
        if "images" in previous_report:
            report["images"] = previous_report["images"]

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    write_label_csv(metadata["ingredient_frequency"], OUTPUT_DIR / "ingredient_frequency.csv")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
