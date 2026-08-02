"""Decision-oriented second-pass audit of the local Yummly benchmark.

This script complements ``yummly_audit.py``. It does not modify source data.
It tests whether the legacy 182-label target can be reproduced from the
checked-in historical preprocessor, audits systematic substring collisions,
quantifies obvious taxonomy fragments and cuisine shortcuts, and evaluates
candidate grouping rules for a leakage-controlled replacement split.

Run from the repository root with::

    python src_scratches/data_anlysis/yummly_deep_audit.py

Use ``--skip-images`` to reuse metadata findings without recomputing image
hashes and the duplicate review sheet.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
YUMMLY_INPUT = REPOSITORY_ROOT / "data" / "input" / "yummly"
YUMMLY_RAW = REPOSITORY_ROOT / "data" / "raw_input" / "yummly"
HISTORICAL_PREPROCESSOR = REPOSITORY_ROOT / "prev_attempts" / "attempt1" / "preprocessing_v2.py"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
REPORT_PATH = OUTPUT_DIR / "yummly_deep_audit.json"
SPLITS = ("train", "val", "test")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalized_text(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(value.casefold()))


def canonical_name(value: str) -> str:
    return "".join(TOKEN_PATTERN.findall(value.casefold()))


def percentile(values: Iterable[float | int], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))


def distribution(values: Iterable[float | int]) -> dict[str, float | int | None]:
    values = list(values)
    if not values:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def load_processed_records() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_split = {split: load_json(YUMMLY_INPUT / split / "metadata.json") for split in SPLITS}
    records = []
    for split in SPLITS:
        for record in by_split[split]:
            copy = dict(record)
            copy["_split"] = split
            copy["_index"] = len(records)
            records.append(copy)
    return records, by_split


def load_original_metadata() -> list[dict[str, Any]]:
    records = []
    for path in sorted((YUMMLY_RAW / "metadata").glob("*.json")):
        records.extend(load_json(path))
    return records


def load_historical_process_function() -> Callable[[str], str]:
    """Load only pure preprocessing functions, avoiding legacy dependencies."""
    required = {
        "remove_paranthesis_and_markups",
        "remove_quantifiers",
        "remove_list_of_elements",
        "remove_textual_quantifiers",
        "remove_treatments",
        "remove_temperature",
        "process_ingredient",
    }
    source = HISTORICAL_PREPROCESSOR.read_text(encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source, filename=str(HISTORICAL_PREPROCESSOR))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in required]
    missing = required - {node.name for node in functions}
    if missing:
        raise RuntimeError(f"Historical functions not found: {sorted(missing)}")
    namespace: dict[str, Any] = {"re": re}
    module = ast.Module(body=functions, type_ignores=[])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        exec(compile(module, str(HISTORICAL_PREPROCESSOR), "exec"), namespace)
    return namespace["process_ingredient"]


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, start=1):
        current = [row]
        for column, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def normalized_levenshtein_similarity(left: str, right: str) -> float:
    denominator = max(len(left), len(right))
    return 1.0 if denominator == 0 else 1.0 - levenshtein_distance(left, right) / denominator


def historical_reproduction_audit(
    original_records: list[dict[str, Any]], processed_records: list[dict[str, Any]]
) -> dict[str, Any]:
    process = load_historical_process_function()
    raw_occurrences = Counter(
        str(line)
        for record in original_records
        for line in (record.get("ingredients") or [])
    )
    process_cache = {line: process(line) for line in raw_occurrences}

    # The historical code thresholds value_counts() over the one-row-per-raw-
    # string mapping table. It therefore counts distinct textual variants, not
    # recipes or ingredient occurrences.
    normalized_distinct_variant_counts = Counter(process_cache.values())
    normalized_occurrence_counts = Counter()
    for line, count in raw_occurrences.items():
        normalized_occurrence_counts[process_cache[line]] += count
    threshold_candidates = {
        label for label, count in normalized_distinct_variant_counts.items() if label and count >= 50
    }
    current_vocabulary = {
        str(label)
        for record in processed_records
        for label in (record.get("ingredients_ok") or [])
    }
    checked_in_raw_recipes = []
    for path in sorted((YUMMLY_RAW / "recipes").glob("*.json")):
        checked_in_raw_recipes.extend(load_json(path))
    checked_in_raw_recipe_ids = {str(record.get("id")) for record in checked_in_raw_recipes}
    eligible_before_similarity_ids = set()
    for record in original_records:
        generated = {process_cache[str(line)] for line in (record.get("ingredients") or [])} & threshold_candidates
        if len(generated) >= 3:
            eligible_before_similarity_ids.add(str(record.get("id")))

    similarities = []
    ordered_candidates = sorted(threshold_candidates)
    for index, left in enumerate(ordered_candidates):
        for right in ordered_candidates[index + 1 :]:
            score = normalized_levenshtein_similarity(left, right)
            if score > 0.85:
                similarities.append({"left": left, "right": right, "similarity": score})

    true_positives = false_positives = false_negatives = exact_matches = 0
    jaccards = []
    per_label_true = Counter()
    per_label_false_negative = Counter()
    per_label_false_positive = Counter()
    mismatch_examples = []
    for record in processed_records:
        generated = set()
        for line in (record.get("ingredients") or []):
            raw_line = str(line)
            if raw_line not in process_cache:
                process_cache[raw_line] = process(raw_line)
            generated.add(process_cache[raw_line])
        generated &= threshold_candidates
        observed = set(map(str, record.get("ingredients_ok") or []))
        intersection = generated & observed
        true_positives += len(intersection)
        false_positives += len(generated - observed)
        false_negatives += len(observed - generated)
        per_label_true.update(intersection)
        per_label_false_positive.update(generated - observed)
        per_label_false_negative.update(observed - generated)
        exact_matches += generated == observed
        union = generated | observed
        jaccards.append(len(intersection) / len(union) if union else 1.0)
        if generated != observed and len(mismatch_examples) < 20:
            mismatch_examples.append(
                {
                    "split": record["_split"],
                    "old_id": record.get("old_id"),
                    "name": record.get("name"),
                    "observed": sorted(observed),
                    "historical_script_before_similarity_merge": sorted(generated),
                }
            )

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)
    label_rows = []
    for label in sorted(current_vocabulary | threshold_candidates):
        label_rows.append(
            {
                "label": label,
                "matching_recipe_occurrences": per_label_true[label],
                "historical_only_recipe_occurrences": per_label_false_positive[label],
                "current_only_recipe_occurrences": per_label_false_negative[label],
                "distinct_raw_variants_after_historical_processing": normalized_distinct_variant_counts[label],
                "raw_line_occurrences_after_historical_processing": normalized_occurrence_counts[label],
            }
        )

    return {
        "historical_script": "prev_attempts/attempt1/preprocessing_v2.py",
        "original_metadata_records": len(original_records),
        "unique_raw_ingredient_strings": len(raw_occurrences),
        "historical_threshold_semantics": "at least 50 distinct raw ingredient strings mapping to a normalized value",
        "labels_surviving_historical_threshold_before_similarity_merge": len(threshold_candidates),
        "current_vocabulary_size": len(current_vocabulary),
        "vocabulary_intersection": len(current_vocabulary & threshold_candidates),
        "current_labels_absent_from_historical_candidates": sorted(current_vocabulary - threshold_candidates),
        "historical_candidates_absent_from_current_labels": sorted(threshold_candidates - current_vocabulary),
        "recipe_filter_reproduction_before_similarity_merge": {
            "checked_in_raw_recipe_records": len(checked_in_raw_recipes),
            "checked_in_unique_recipe_ids": len(checked_in_raw_recipe_ids),
            "historical_script_eligible_source_records": len(eligible_before_similarity_ids),
            "shared_recipe_ids": len(checked_in_raw_recipe_ids & eligible_before_similarity_ids),
            "checked_in_ids_not_eligible_under_historical_script": len(
                checked_in_raw_recipe_ids - eligible_before_similarity_ids
            ),
            "checked_in_ids_not_eligible_examples": sorted(
                checked_in_raw_recipe_ids - eligible_before_similarity_ids
            )[:30],
            "historically_eligible_ids_absent_from_checked_in_recipes": len(
                eligible_before_similarity_ids - checked_in_raw_recipe_ids
            ),
            "historically_eligible_ids_absent_examples": sorted(
                eligible_before_similarity_ids - checked_in_raw_recipe_ids
            )[:30],
        },
        "similarity_pairs_above_historical_0_85_threshold": sorted(
            similarities, key=lambda row: (-row["similarity"], row["left"], row["right"])
        ),
        "similarity_pairs_where_both_labels_survive_in_current_vocabulary": [
            row for row in similarities if row["left"] in current_vocabulary and row["right"] in current_vocabulary
        ],
        "recipe_set_comparison_before_similarity_merge": {
            "records": len(processed_records),
            "exact_set_matches": exact_matches,
            "exact_set_match_rate": exact_matches / len(processed_records),
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1,
            "jaccard": distribution(jaccards),
            "mismatch_examples": mismatch_examples,
        },
        "label_comparison": label_rows,
        "top_historical_candidates_by_distinct_variants": [
            {
                "label": label,
                "distinct_raw_variants": count,
                "raw_line_occurrences": normalized_occurrence_counts[label],
            }
            for label, count in normalized_distinct_variant_counts.most_common(250)
            if label
        ],
    }


COLLISION_RULES = {
    "apple_from_pineapple": {
        "target": "apple",
        "trigger": re.compile(r"\bpineapples?\b", re.IGNORECASE),
        "legitimate": re.compile(r"(?<!pine)\bapples?\b", re.IGNORECASE),
    },
    "pear_from_pearl": {
        "target": "pear",
        "trigger": re.compile(r"\bpearl\w*\b", re.IGNORECASE),
        "legitimate": re.compile(r"\bpears?\b", re.IGNORECASE),
    },
    "pear_from_spears": {
        "target": "pear",
        "trigger": re.compile(r"\bspears?\b", re.IGNORECASE),
        "legitimate": re.compile(r"\bpears?\b", re.IGNORECASE),
    },
    "butter_from_butternut": {
        "target": "butter",
        "trigger": re.compile(r"\bbutternut\b", re.IGNORECASE),
        "legitimate": re.compile(r"\bbutter\b", re.IGNORECASE),
    },
    "butter_from_buttermilk": {
        "target": "butter",
        "trigger": re.compile(r"\bbuttermilk\b", re.IGNORECASE),
        "legitimate": re.compile(r"\bbutter\b", re.IGNORECASE),
    },
    "lemon_from_lemongrass": {
        "target": "lemon",
        "trigger": re.compile(r"\blemongrass\b", re.IGNORECASE),
        "legitimate": re.compile(r"\blemons?\b", re.IGNORECASE),
    },
    "beans_from_bean_sprouts": {
        "target": "beans",
        "trigger": re.compile(r"\bbean\s?sprouts?\b", re.IGNORECASE),
        "legitimate": re.compile(r"\bbeans\b(?!\s?sprout)", re.IGNORECASE),
    },
    "radish_from_horseradish": {
        "target": "radish",
        "trigger": re.compile(r"\bhorseradish\b", re.IGNORECASE),
        "legitimate": re.compile(r"(?<!horse)\bradishes?\b", re.IGNORECASE),
    },
    "grape_from_grapefruit": {
        "target": "grape",
        "trigger": re.compile(r"\bgrapefruits?\b", re.IGNORECASE),
        "legitimate": re.compile(r"\bgrapes?\b(?!fruit)", re.IGNORECASE),
    },
    "pepper_from_pepperoni": {
        "target": "pepper",
        "trigger": re.compile(r"\bpepperoni\b", re.IGNORECASE),
        "legitimate": re.compile(r"\bpeppers?\b(?!oni)", re.IGNORECASE),
    },
    "pepper_from_peppermint": {
        "target": "pepper",
        "trigger": re.compile(r"\bpeppermint\b", re.IGNORECASE),
        "legitimate": re.compile(r"\bpeppers?\b(?!mint)", re.IGNORECASE),
    },
    "water_from_watercress": {
        "target": "water",
        "trigger": re.compile(r"\bwatercress\b", re.IGNORECASE),
        "legitimate": re.compile(r"\bwater\b(?!cress)", re.IGNORECASE),
    },
    "egg_from_veggies": {
        "target": "egg",
        "trigger": re.compile(r"\bveggies\b", re.IGNORECASE),
        "legitimate": re.compile(r"\beggs?\b", re.IGNORECASE),
    },
    "tea_from_measurement_words": {
        "target": "tea",
        "trigger": re.compile(r"\btea\s?spoons?\b", re.IGNORECASE),
        "legitimate": re.compile(r"\btea\b(?!\s?spoon)", re.IGNORECASE),
    },
    "cream_from_creamy": {
        "target": "cream",
        "trigger": re.compile(r"\bcreamy\b", re.IGNORECASE),
        "legitimate": re.compile(r"\bcream\b", re.IGNORECASE),
    },
    "corn_from_acorn": {
        "target": "corn",
        "trigger": re.compile(r"\bacorn\b", re.IGNORECASE),
        "legitimate": re.compile(r"(?<!a)\bcorn\b", re.IGNORECASE),
    },
    "broccoli_from_broccolini": {
        "target": "broccoli",
        "trigger": re.compile(r"\bbroccolini\b", re.IGNORECASE),
        "legitimate": re.compile(r"\bbroccoli\b(?!ni)", re.IGNORECASE),
    },
    "liquor_from_ginger": {
        "target": "liquor",
        "trigger": re.compile(r"\bginger\w*\b", re.IGNORECASE),
        "legitimate": re.compile(
            r"\b(vodka|gin|rum|whiske?y|tequila|brandy|liqueur|sake|cognac|vermouth|sherry|port|champagne|prosecco|cider|bourbon|marsala|calvados|grappa|schnapps|kirsch|amaretto|frangelico|cointreau)\b",
            re.IGNORECASE,
        ),
    },
    "liquor_from_portobello": {
        "target": "liquor",
        "trigger": re.compile(r"\bportobellos?\b", re.IGNORECASE),
        "legitimate": re.compile(
            r"\b(vodka|gin|rum|whiske?y|tequila|brandy|liqueur|sake|cognac|vermouth|sherry|port|champagne|prosecco|cider|bourbon|marsala|calvados|grappa|schnapps|kirsch|amaretto|frangelico|cointreau)\b",
            re.IGNORECASE,
        ),
    },
    "liquor_from_crumb": {
        "target": "liquor",
        "trigger": re.compile(r"\b\w*crumb\w*\b", re.IGNORECASE),
        "legitimate": re.compile(
            r"\b(vodka|gin|rum|whiske?y|tequila|brandy|liqueur|sake|cognac|vermouth|sherry|port|champagne|prosecco|cider|bourbon|marsala|calvados|grappa|schnapps|kirsch|amaretto|frangelico|cointreau)\b",
            re.IGNORECASE,
        ),
    },
    "cheese_from_cantaloupe": {
        "target": "cheese",
        "trigger": re.compile(r"\bcantaloupes?\b", re.IGNORECASE),
        "legitimate": re.compile(
            r"\b(cheese|ricotta|parmesan|parmigiano|mozzarella|pecorino|provolone|feta|gouda|manchego|brie|camembert|gorgonzola|roquefort|emmental|gruyere|asiago|halloumi|paneer|cheddar|mascarpone|cotija|queso)\b",
            re.IGNORECASE,
        ),
    },
    "cheese_from_jackfruit": {
        "target": "cheese",
        "trigger": re.compile(r"\bjackfruits?\b", re.IGNORECASE),
        "legitimate": re.compile(
            r"\b(cheese|ricotta|parmesan|parmigiano|mozzarella|pecorino|provolone|feta|gouda|manchego|brie|camembert|gorgonzola|roquefort|emmental|gruyere|asiago|halloumi|paneer|cheddar|mascarpone|cotija|queso)\b",
            re.IGNORECASE,
        ),
    },
    "egg_from_eggplant": {
        "target": "egg",
        "trigger": re.compile(r"\beggplants?\b", re.IGNORECASE),
        "legitimate": re.compile(r"\beggs?\b(?!plant)", re.IGNORECASE),
    },
    "sage_from_sausage": {
        "target": "sage",
        "trigger": re.compile(r"\bsausages?\b", re.IGNORECASE),
        "legitimate": re.compile(r"\bsage\b", re.IGNORECASE),
    },
    "tea_from_steak": {
        "target": "tea",
        "trigger": re.compile(r"\bsteaks?\b", re.IGNORECASE),
        "legitimate": re.compile(r"\btea\b", re.IGNORECASE),
    },
    "oil_from_boiling": {
        "target": "oil",
        "trigger": re.compile(r"\bboil\w*\b", re.IGNORECASE),
        "legitimate": re.compile(r"\boils?\b", re.IGNORECASE),
    },
    "rice_from_licorice": {
        "target": "rice",
        "trigger": re.compile(r"\blicorice\b", re.IGNORECASE),
        "legitimate": re.compile(r"\brice\b", re.IGNORECASE),
    },
    "oat_from_coating": {
        "target": "oat",
        "trigger": re.compile(r"\bcoat(?:ing|ed|s)?\b", re.IGNORECASE),
        "legitimate": re.compile(r"\boats?(?:meal)?\b", re.IGNORECASE),
    },
}


def collision_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    affected_record_indices = set()
    label_counts = Counter(label for record in records for label in set(record.get("ingredients_ok") or []))
    for name, rule in COLLISION_RULES.items():
        examples = []
        indices = []
        for record in records:
            labels = set(map(str, record.get("ingredients_ok") or []))
            if rule["target"] not in labels:
                continue
            text = " | ".join(map(str, record.get("ingredients") or []))
            if rule["trigger"].search(text) and not rule["legitimate"].search(text):
                indices.append(record["_index"])
                if len(examples) < 12:
                    examples.append(
                        {
                            "split": record["_split"],
                            "old_id": record.get("old_id"),
                            "name": record.get("name"),
                            "ingredients": record.get("ingredients"),
                        }
                    )
        affected_record_indices.update(indices)
        denominator = label_counts[rule["target"]]
        output[name] = {
            "target": rule["target"],
            "conservative_affected_records": len(indices),
            "fraction_of_target_occurrences": len(indices) / denominator if denominator else None,
            "examples": examples,
        }

    # Surface larger-token hosts for every single-word label so future audits
    # are not limited to the hand-written collisions above.
    host_counts: dict[str, Counter[str]] = defaultdict(Counter)
    host_record_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        raw_tokens = TOKEN_PATTERN.findall(" ".join(map(str, record.get("ingredients") or [])).casefold())
        raw_token_counts = Counter(raw_tokens)
        for label in set(map(str, record.get("ingredients_ok") or [])):
            normalized = normalized_text(label)
            if " " in normalized or len(normalized) < 3:
                continue
            hosts = {token for token in raw_tokens if normalized in token and token != normalized}
            for host in hosts:
                host_counts[label][host] += raw_token_counts[host]
            host_record_counts[label].update(hosts)
    host_rows = []
    for label, counter in host_record_counts.items():
        for host, recipe_count in counter.most_common(15):
            host_rows.append(
                {
                    "label": label,
                    "host_token": host,
                    "recipe_count": recipe_count,
                    "token_occurrences": host_counts[label][host],
                }
            )

    return {
        "rules": output,
        "unique_records_affected_by_at_least_one_conservative_rule": len(affected_record_indices),
        "fraction_of_dataset_affected_by_at_least_one_conservative_rule": len(affected_record_indices) / len(records),
        "substring_host_tokens": sorted(host_rows, key=lambda row: (-row["recipe_count"], row["label"], row["host_token"]))[:500],
    }


ALIAS_GROUPS = {
    "arugula": ["arugula", "baby arugula"],
    "bay leaf": ["bay leaf", "bay leaves"],
    "celery": ["celery", "celery ribs", "celery stalks", "stalks celery"],
    "cherry": ["cherry", "cherries"],
    "cilantro": ["cilantro", "cilantro leaves", "bunch cilantro"],
    "garam masala": ["garam masala", "garam masala powder"],
    "leek": ["leek", "leeks"],
    "lime": ["lime", "limes"],
    "raspberry": ["raspberry", "raspberries"],
    "strawberry": ["strawberry", "strawberries"],
    "tomato": ["tomato", "tomate"],
}


MANUAL_EXACT_IMAGE_EXCLUSIONS = {
    "001474____mexican.jpg": "Cooking Light advertisement rather than a recipe-specific dish image",
    "002383____french.jpg": "publisher/logo graphic rather than food",
    "005016____chinese.jpg": "generic empty place-setting category image",
    "013306____japanese.jpg": "publisher/brand graphic rather than food",
    "002293____thai.jpg": "featureless person-silhouette placeholder",
    "001300____indian.jpg": "generic condiment place-setting category image shared by unrelated recipes",
    "000073____french.jpg": "BBC logo rather than food",
    "002457____thai.jpg": "generic cutlery place-setting image shared by unrelated recipes",
}


def taxonomy_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    labels_to_remove = set()
    for canonical, aliases in ALIAS_GROUPS.items():
        recipe_indices = set()
        summed_occurrences = 0
        overlap = 0
        per_alias = Counter()
        for record in records:
            present = set(map(str, record.get("ingredients_ok") or [])) & set(aliases)
            if present:
                recipe_indices.add(record["_index"])
                summed_occurrences += len(present)
                overlap += max(0, len(present) - 1)
                per_alias.update(present)
        labels_to_remove.update(set(aliases) - {canonical})
        rows.append(
            {
                "canonical": canonical,
                "aliases": aliases,
                "per_alias_recipe_occurrences": dict(per_alias),
                "summed_occurrences": summed_occurrences,
                "unique_recipes_after_merge": len(recipe_indices),
                "within_recipe_duplicate_labels_removed": overlap,
            }
        )
    vocabulary = {label for record in records for label in map(str, record.get("ingredients_ok") or [])}
    return {
        "obvious_legacy_alias_groups": rows,
        "current_vocabulary_size": len(vocabulary),
        "vocabulary_size_after_only_obvious_alias_merges": len(vocabulary - labels_to_remove),
        "semantic_non_merges": {
            "cilantro_vs_coriander": "Keep distinct unless raw-line review proves leaf/seed equivalence.",
            "fennel_bulb_vs_fennel_seeds": "Keep distinct plant parts.",
            "named_chiles": "Keep jalapeno, poblano, serrano, ancho and generic green chiles distinct; optionally add a parent chile node.",
            "prepared_sauces": "Keep named sauces distinct from generic sauce; optionally model components hierarchically.",
        },
    }


def legacy_repair_sensitivity_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure impact of conservative removals and obvious alias merges.

    This is a sensitivity analysis, not a replacement ground truth. A full
    regeneration must work from raw ingredient lines and a reviewed ontology.
    """
    alias_lookup = {
        alias: canonical
        for canonical, aliases in ALIAS_GROUPS.items()
        for alias in aliases
    }
    original_cardinality = []
    repaired_cardinality = []
    removed_by_target = Counter()
    changed_records = 0
    records_below_three = 0
    original_positive_count = repaired_positive_count = 0
    repaired_vocabulary = set()
    for record in records:
        original = set(map(str, record.get("ingredients_ok") or []))
        repaired = set(original)
        text = " | ".join(map(str, record.get("ingredients") or []))
        for rule in COLLISION_RULES.values():
            target = str(rule["target"])
            if target in repaired and rule["trigger"].search(text) and not rule["legitimate"].search(text):
                repaired.remove(target)
                removed_by_target[target] += 1
        repaired = {alias_lookup.get(label, label) for label in repaired}
        changed_records += repaired != original
        records_below_three += len(repaired) < 3
        original_cardinality.append(len(original))
        repaired_cardinality.append(len(repaired))
        original_positive_count += len(original)
        repaired_positive_count += len(repaired)
        repaired_vocabulary.update(repaired)
    return {
        "interpretation": "Sensitivity analysis only; do not use as corrected ground truth.",
        "changed_records": changed_records,
        "changed_record_fraction": changed_records / len(records),
        "removed_candidate_positive_labels_by_target": dict(removed_by_target.most_common()),
        "original_positive_labels": original_positive_count,
        "remaining_positive_labels": repaired_positive_count,
        "positive_label_reduction": original_positive_count - repaired_positive_count,
        "positive_label_reduction_fraction": (original_positive_count - repaired_positive_count) / original_positive_count,
        "original_cardinality": distribution(original_cardinality),
        "sensitivity_cardinality": distribution(repaired_cardinality),
        "records_below_three_labels_after_sensitivity_repair": records_below_three,
        "sensitivity_vocabulary_size": len(repaired_vocabulary),
    }


def binary_entropy(probability: float) -> float:
    if probability <= 0 or probability >= 1:
        return 0.0
    return -probability * math.log(probability) - (1 - probability) * math.log(1 - probability)


def average_precision_with_ties(targets: list[int], scores: list[float]) -> float | None:
    positives = sum(targets)
    if positives == 0:
        return None
    grouped = defaultdict(lambda: [0, 0])
    for target, score in zip(targets, scores):
        grouped[score][0] += int(target)
        grouped[score][1] += 1
    true_positive = predicted_positive = 0
    average_precision = 0.0
    previous_recall = 0.0
    for score in sorted(grouped, reverse=True):
        group_true, group_total = grouped[score]
        true_positive += group_true
        predicted_positive += group_total
        recall = true_positive / positives
        precision = true_positive / predicted_positive
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return average_precision


def cuisine_shortcut_audit(
    records: list[dict[str, Any]], by_split: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    cuisines = sorted({str(record.get("cuisine", "<missing>")).casefold() for record in records})
    labels = sorted({str(label) for record in records for label in (record.get("ingredients_ok") or [])})
    total = len(records)
    cuisine_counts = Counter(str(record.get("cuisine", "<missing>")).casefold() for record in records)
    label_counts = Counter(label for record in records for label in set(map(str, record.get("ingredients_ok") or [])))
    joint_positive = defaultdict(Counter)
    for record in records:
        cuisine = str(record.get("cuisine", "<missing>")).casefold()
        joint_positive[cuisine].update(set(map(str, record.get("ingredients_ok") or [])))

    train_cuisine_counts = Counter(str(record.get("cuisine", "<missing>")).casefold() for record in by_split["train"])
    train_joint_positive = defaultdict(Counter)
    for record in by_split["train"]:
        cuisine = str(record.get("cuisine", "<missing>")).casefold()
        train_joint_positive[cuisine].update(set(map(str, record.get("ingredients_ok") or [])))

    rows = []
    for label in labels:
        p_y = label_counts[label] / total
        mutual_information = 0.0
        for cuisine in cuisines:
            cuisine_total = cuisine_counts[cuisine]
            positive = joint_positive[cuisine][label]
            for value, joint_count in ((1, positive), (0, cuisine_total - positive)):
                if joint_count == 0:
                    continue
                p_joint = joint_count / total
                p_value = p_y if value else 1 - p_y
                mutual_information += p_joint * math.log(p_joint / ((cuisine_total / total) * p_value))
        entropy = binary_entropy(p_y)
        normalized_mi = mutual_information / entropy if entropy else 0.0

        targets = []
        scores = []
        for record in by_split["test"]:
            cuisine = str(record.get("cuisine", "<missing>")).casefold()
            targets.append(label in set(map(str, record.get("ingredients_ok") or [])))
            scores.append(train_joint_positive[cuisine][label] / train_cuisine_counts[cuisine])
        cuisine_ap = average_precision_with_ties(targets, scores)
        test_prevalence = sum(targets) / len(targets)
        rows.append(
            {
                "label": label,
                "dataset_prevalence": p_y,
                "normalized_mutual_information_with_cuisine": normalized_mi,
                "test_prevalence": test_prevalence,
                "test_average_precision_from_train_cuisine_prior": cuisine_ap,
                "average_precision_gain_over_test_prevalence": cuisine_ap - test_prevalence if cuisine_ap is not None else None,
            }
        )
    valid_ap = [row["test_average_precision_from_train_cuisine_prior"] for row in rows if row["test_average_precision_from_train_cuisine_prior"] is not None]
    return {
        "per_label": sorted(rows, key=lambda row: (-row["normalized_mutual_information_with_cuisine"], row["label"])),
        "macro_average_precision_of_cuisine_prior": statistics.fmean(valid_ap),
        "top_labels_by_cuisine_information": sorted(rows, key=lambda row: -row["normalized_mutual_information_with_cuisine"])[:30],
        "top_labels_by_cuisine_prior_ap_gain": sorted(rows, key=lambda row: -(row["average_precision_gain_over_test_prevalence"] or 0))[:30],
    }


def support_reliability_audit(by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    counts = {split: Counter() for split in SPLITS}
    for split in SPLITS:
        for record in by_split[split]:
            counts[split].update(set(map(str, record.get("ingredients_ok") or [])))
    labels = sorted(set(counts["train"]) | set(counts["val"]) | set(counts["test"]))
    rows = [
        {"label": label, **{split: counts[split][label] for split in SPLITS}}
        for label in labels
    ]
    return {
        "threshold_counts": {
            f"train_at_least_{threshold}": sum(counts["train"][label] >= threshold for label in labels)
            for threshold in (50, 100, 200, 500, 1000)
        }
        | {
            f"val_and_test_each_at_least_{threshold}": sum(
                counts["val"][label] >= threshold and counts["test"][label] >= threshold for label in labels
            )
            for threshold in (5, 10, 20, 50, 100)
        },
        "labels_with_fewer_than_100_train_examples": [row for row in rows if row["train"] < 100],
        "labels_with_fewer_than_20_examples_in_val_or_test": [
            row for row in rows if row["val"] < 20 or row["test"] < 20
        ],
        "per_label": rows,
    }


def difference_hash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.int16)
    bits = pixels[:, :-1] > pixels[:, 1:]
    return sum(int(bit) << index for index, bit in enumerate(bits.ravel()))


def dct_basis(size: int = 32, frequencies: int = 8) -> np.ndarray:
    positions = np.arange(size)
    basis = np.empty((frequencies, size), dtype=np.float64)
    basis[0] = 1 / math.sqrt(size)
    for frequency in range(1, frequencies):
        basis[frequency] = math.sqrt(2 / size) * np.cos(math.pi * (2 * positions + 1) * frequency / (2 * size))
    return basis


DCT_BASIS = dct_basis()


def perceptual_hash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float64)
    low_frequency = DCT_BASIS @ pixels @ DCT_BASIS.T
    median = float(np.median(low_frequency.ravel()[1:]))
    bits = low_frequency > median
    return sum(int(bit) << index for index, bit in enumerate(bits.ravel()))


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def union_groups(dsu: DisjointSet, groups: Iterable[list[int]]) -> None:
    for group in groups:
        if len(group) > 1:
            anchor = group[0]
            for value in group[1:]:
                dsu.union(anchor, value)


def component_summary(dsu: DisjointSet, records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(list)
    for record in records:
        groups[dsu.find(record["_index"])].append(record)
    duplicates = [group for group in groups.values() if len(group) > 1]
    cross_split = [group for group in duplicates if len({record["_split"] for record in group}) > 1]
    evaluation_overlap = {}
    for split in ("val", "test"):
        affected = sum(
            1
            for group in groups.values()
            if any(record["_split"] == "train" for record in group)
            for record in group
            if record["_split"] == split
        )
        split_size = sum(record["_split"] == split for record in records)
        evaluation_overlap[split] = {
            "records_grouped_with_at_least_one_train_record": affected,
            "fraction_of_split": affected / split_size,
        }
    return {
        "components": len(groups),
        "multi_record_components": len(duplicates),
        "records_in_multi_record_components": sum(map(len, duplicates)),
        "largest_component": max(map(len, groups.values())),
        "cross_split_components_under_current_split": len(cross_split),
        "records_in_cross_split_components_under_current_split": sum(map(len, cross_split)),
        "evaluation_records_grouped_with_train_under_current_split": evaluation_overlap,
        "component_size_distribution": distribution(map(len, groups.values())),
        "largest_components": [
            {
                "size": len(group),
                "splits": dict(Counter(record["_split"] for record in group)),
                "records": [
                    {
                        "split": record["_split"],
                        "old_id": record.get("old_id"),
                        "name": record.get("name"),
                        "image": record.get("image"),
                    }
                    for record in group[:20]
                ],
            }
            for group in sorted(groups.values(), key=len, reverse=True)[:20]
        ],
    }


def normalized_raw_signature(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(normalized_text(str(line)) for line in (record.get("ingredients") or [])))


def name_similarity_edges(records: list[dict[str, Any]], threshold: float = 0.8) -> list[list[int]]:
    by_name = defaultdict(list)
    for record in records:
        name = canonical_name(str(record.get("name", "")))
        if name:
            by_name[name].append(record)
    edges = []
    for group in by_name.values():
        if len(group) < 2:
            continue
        line_sets = {record["_index"]: set(normalized_raw_signature(record)) for record in group}
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                left_lines = line_sets[left["_index"]]
                right_lines = line_sets[right["_index"]]
                union = left_lines | right_lines
                similarity = len(left_lines & right_lines) / len(union) if union else 1.0
                if similarity >= threshold:
                    edges.append([left["_index"], right["_index"]])
    return edges


def create_duplicate_review_sheet(
    groups: list[tuple[str, list[int]]], records: list[dict[str, Any]], output_path: Path
) -> None:
    selected = groups[:40]
    cell_width, image_height, caption_height = 240, 160, 54
    columns = 4
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * (image_height + caption_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for position, (kind, indices) in enumerate(selected):
        record = records[indices[0]]
        path = YUMMLY_INPUT / record["_split"] / str(record.get("image"))
        x = (position % columns) * cell_width
        y = (position // columns) * (image_height + caption_height)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((cell_width, image_height), Image.Resampling.LANCZOS)
            sheet.paste(image, (x + (cell_width - image.width) // 2, y + (image_height - image.height) // 2))
        split_counts = ",".join(f"{key}:{value}" for key, value in sorted(Counter(records[i]["_split"] for i in indices).items()))
        caption = f"{position + 1}. {kind} n={len(indices)} {split_counts}\n{str(record.get('name', ''))[:34]}\n{record.get('image')}"
        draw.multiline_text((x + 4, y + image_height + 2), caption, fill="black", font=font, spacing=1)
    sheet.save(output_path, quality=92)


def image_and_grouping_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    exact_groups = defaultdict(list)
    dhash_groups = defaultdict(list)
    high_confidence_perceptual_groups = defaultdict(list)
    exact_hash_by_index = {}
    image_rows = []
    for record in records:
        path = YUMMLY_INPUT / record["_split"] / str(record.get("image"))
        content = path.read_bytes()
        exact_hash = hashlib.sha256(content).hexdigest()
        with Image.open(path) as image:
            image.load()
            dhash = difference_hash(image)
            phash = perceptual_hash(image)
        index = record["_index"]
        exact_groups[exact_hash].append(index)
        exact_hash_by_index[index] = exact_hash
        dhash_groups[dhash].append(index)
        high_confidence_perceptual_groups[(dhash, phash)].append(index)
        image_rows.append(
            {
                "index": index,
                "split": record["_split"],
                "image": record.get("image"),
                "sha256": exact_hash,
                "dhash": f"{dhash:016x}",
                "phash": f"{phash:016x}",
            }
        )

    exact_duplicates = [group for group in exact_groups.values() if len(group) > 1]
    dhash_candidates = [group for group in dhash_groups.values() if len(group) > 1]
    high_confidence = [group for group in high_confidence_perceptual_groups.values() if len(group) > 1]
    exact_pair_keys = {tuple(sorted(group)) for group in exact_duplicates}
    non_exact_high_confidence = [group for group in high_confidence if tuple(sorted(group)) not in exact_pair_keys]

    raw_groups = defaultdict(list)
    for record in records:
        raw_groups[normalized_raw_signature(record)].append(record["_index"])
    exact_raw_groups = [group for group in raw_groups.values() if len(group) > 1]
    name_edges = name_similarity_edges(records)

    policies = {}
    dsu = DisjointSet(len(records))
    union_groups(dsu, exact_duplicates)
    policies["exact_images_only"] = component_summary(dsu, records)

    dsu = DisjointSet(len(records))
    union_groups(dsu, exact_duplicates)
    union_groups(dsu, exact_raw_groups)
    policies["exact_images_and_exact_normalized_raw_ingredient_lists"] = component_summary(dsu, records)

    dsu = DisjointSet(len(records))
    union_groups(dsu, exact_duplicates)
    union_groups(dsu, high_confidence)
    union_groups(dsu, exact_raw_groups)
    union_groups(dsu, name_edges)
    policies[
        "recommended_high_precision_grouping_candidate"
    ] = component_summary(dsu, records)

    review_groups = [("exact", group) for group in exact_duplicates]
    review_groups += [("perceptual", group) for group in non_exact_high_confidence]
    review_groups.sort(key=lambda item: len(item[1]), reverse=True)
    create_duplicate_review_sheet(review_groups, records, OUTPUT_DIR / "duplicate_group_review.jpg")

    record_by_image = {str(record.get("image")): record for record in records}
    manual_exclusions = []
    excluded_indices = set()
    for image_name, reason in MANUAL_EXACT_IMAGE_EXCLUSIONS.items():
        representative = record_by_image[image_name]
        indices = exact_groups[exact_hash_by_index[representative["_index"]]]
        excluded_indices.update(indices)
        manual_exclusions.append(
            {
                "representative_image": image_name,
                "reason": reason,
                "exact_group_size": len(indices),
                "split_counts": dict(Counter(records[index]["_split"] for index in indices)),
                "cuisine_count": len({records[index].get("cuisine") for index in indices}),
                "label_set_count": len({tuple(sorted(records[index].get("ingredients_ok") or [])) for index in indices}),
            }
        )

    review_rows = []
    for rank, (kind, indices) in enumerate(review_groups, start=1):
        if rank > 500:
            break
        review_rows.append(
            {
                "rank": rank,
                "kind": kind,
                "size": len(indices),
                "cross_split": len({records[index]["_split"] for index in indices}) > 1,
                "split_counts": json.dumps(dict(Counter(records[index]["_split"] for index in indices)), sort_keys=True),
                "cuisine_count": len({records[index].get("cuisine") for index in indices}),
                "label_set_count": len({tuple(sorted(records[index].get("ingredients_ok") or [])) for index in indices}),
                "representative_split": records[indices[0]]["_split"],
                "representative_image": records[indices[0]].get("image"),
                "representative_name": records[indices[0]].get("name"),
            }
        )
    with (OUTPUT_DIR / "duplicate_group_review.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)

    return {
        "image_hash_rows": len(image_rows),
        "exact_duplicate_groups": len(exact_duplicates),
        "dhash_only_candidate_groups": len(dhash_candidates),
        "matching_dhash_and_phash_groups": len(high_confidence),
        "matching_dhash_and_phash_groups_not_identical_to_an_exact_group": len(non_exact_high_confidence),
        "exact_normalized_raw_ingredient_list_groups": len(exact_raw_groups),
        "canonical_name_and_raw_line_jaccard_at_least_0_8_edges": len(name_edges),
        "manual_visual_review_of_largest_exact_groups": {
            "confirmed_exclusion_groups": len(manual_exclusions),
            "confirmed_exclusion_records": len(excluded_indices),
            "scope_note": "Conservative, non-exhaustive review; remaining groups and singleton images still require review.",
            "decisions": manual_exclusions,
        },
        "grouping_policy_comparison": policies,
        "review_artifacts": {
            "contact_sheet": "src_scratches/data_anlysis/outputs/duplicate_group_review.jpg",
            "manifest": "src_scratches/data_anlysis/outputs/duplicate_group_review.csv",
        },
    }


def write_target_review_csv(report: dict[str, Any]) -> None:
    reproduction = {
        row["label"]: row for row in report["historical_reproduction"]["label_comparison"]
    }
    support = {row["label"]: row for row in report["support_reliability"]["per_label"]}
    shortcut = {row["label"]: row for row in report["cuisine_shortcuts"]["per_label"]}
    labels = sorted(set(reproduction) | set(support) | set(shortcut))
    rows = []
    for label in labels:
        rows.append(
            {"label": label}
            | {f"reproduction_{key}": value for key, value in reproduction.get(label, {}).items() if key != "label"}
            | {f"support_{key}": value for key, value in support.get(label, {}).items() if key != "label"}
            | {f"cuisine_{key}": value for key, value in shortcut.get(label, {}).items() if key != "label"}
        )
    with (OUTPUT_DIR / "target_review.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-images", action="store_true", help="Skip image hashing and grouping recomputation")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records, by_split = load_processed_records()
    original_records = load_original_metadata()
    report = {
        "dataset": "Yummly-66K local derivative",
        "repository_root": ".",
        "processed_records": len(records),
        "historical_reproduction": historical_reproduction_audit(original_records, records),
        "substring_collisions": collision_audit(records),
        "taxonomy": taxonomy_audit(records),
        "legacy_repair_sensitivity": legacy_repair_sensitivity_audit(records),
        "support_reliability": support_reliability_audit(by_split),
        "cuisine_shortcuts": cuisine_shortcut_audit(records, by_split),
    }
    if not args.skip_images:
        report["images_and_grouping"] = image_and_grouping_audit(records)
    elif REPORT_PATH.exists():
        previous = load_json(REPORT_PATH)
        if "images_and_grouping" in previous:
            report["images_and_grouping"] = previous["images_and_grouping"]

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    write_target_review_csv(report)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
