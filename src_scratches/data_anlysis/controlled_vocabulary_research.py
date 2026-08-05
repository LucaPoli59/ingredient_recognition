"""Reproduce the aggregate lexical-coverage study for Data Work package 2.2c.

The script evaluates offline vocabulary releases against the original Yummly
ingredient lines. It never writes metadata or a per-line mapping artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_processing.ingredient_standardization import (
    _normalize_source_text,
    normalize_ingredient_targets,
)


FOODON_PRODUCT_ROOT = "FOODON_00001002"
LANGUAL_PRODUCT_TYPE_ROOT = "A0361"
_CURIE = re.compile(r"/([^/<>]+)>?$")
_QUOTED_VALUE = re.compile(r'^"(.*)"(?:@[A-Za-z-]+)?$')
_FOODEX_LABEL = re.compile(r"^\d+\s*-\s*(.+?)\s*\(efsa foodex2\)$", re.IGNORECASE)


@dataclass(frozen=True)
class Vocabulary:
    name: str
    preferred_labels: dict[str, str]
    preferred_surface_to_ids: dict[str, frozenset[str]]
    surface_to_ids: dict[str, frozenset[str]]


@dataclass(frozen=True)
class Coverage:
    vocabulary: str
    line_occurrences: int
    direct_occurrences: int
    fallback_occurrences: int
    unmatched_occurrences: int
    ambiguous_occurrences: int
    unique_source_terms: int
    direct_unique_terms: int
    fallback_unique_terms: int
    unmatched_unique_terms: int
    ambiguous_unique_terms: int


def surface_key(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = value.lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _curie(uri: str) -> str:
    match = _CURIE.search(uri)
    return match.group(1) if match else uri.strip("<>")


def _quoted(value: str) -> str:
    match = _QUOTED_VALUE.fullmatch(value)
    value = match.group(1).replace(r'\"', '"') if match else value
    value = re.sub(r"@[A-Za-z-]+$", "", value)
    return re.sub(r"\^\^<[^>]+>$", "", value)


def _descendants(parents: dict[str, set[str]], root: str) -> set[str]:
    children: dict[str, set[str]] = defaultdict(set)
    for child, parent_ids in parents.items():
        for parent in parent_ids:
            children[parent].add(child)
    result = {root}
    frontier = [root]
    while frontier:
        child = frontier.pop()
        for descendant in children.get(child, ()):
            if descendant not in result:
                result.add(descendant)
                frontier.append(descendant)
    return result


def load_foodon(path: Path, include_unspecified_synonyms: bool = False) -> Vocabulary:
    parents: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, str] = {}
    lexical_rows: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) != 4:
                continue
            class_uri, parent_uri, kind, raw_label = row
            concept_id = _curie(class_uri)
            if parent_uri:
                parents[concept_id].add(_curie(parent_uri))
            if not raw_label:
                continue
            kind = _quoted(kind)
            label = _quoted(raw_label)
            lexical_rows.append((concept_id, kind, label))
            if kind == "label":
                labels[concept_id] = label

    product_ids = _descendants(parents, FOODON_PRODUCT_ROOT)
    accepted_kinds = {"label", "synonym (exact)"}
    if include_unspecified_synonyms:
        accepted_kinds.add("synonym")
    surfaces: dict[str, set[str]] = defaultdict(set)
    preferred_surfaces: dict[str, set[str]] = defaultdict(set)
    for concept_id, kind, label in lexical_rows:
        if concept_id in product_ids and kind in accepted_kinds:
            key = surface_key(label)
            if key:
                surfaces[key].add(concept_id)
                if kind == "label":
                    preferred_surfaces[key].add(concept_id)
                if kind == "label" and key.endswith(" food product"):
                    surfaces[key.removesuffix(" food product")].add(concept_id)
    return Vocabulary(
        "FoodOn food-product branch",
        {key: labels[key] for key in product_ids if key in labels},
        {key: frozenset(value) for key, value in preferred_surfaces.items()},
        {key: frozenset(value) for key, value in surfaces.items()},
    )


def load_langual(path: Path) -> tuple[Vocabulary, Vocabulary]:
    tree = ET.parse(path)
    labels: dict[str, str] = {}
    parents: dict[str, set[str]] = defaultdict(set)
    lexical_rows: list[tuple[str, str]] = []
    active_ids: set[str] = set()
    for descriptor in tree.findall(".//DESCRIPTOR"):
        concept_id = (descriptor.findtext("FTC") or "").strip()
        if not concept_id:
            continue
        if (descriptor.findtext("ACTIVE") or "").strip().lower() == "true":
            active_ids.add(concept_id)
        parent = (descriptor.findtext("BT") or "").strip()
        if parent:
            parents[concept_id].add(parent)
        label_node = descriptor.find("TERM")
        if label_node is not None and label_node.text:
            label = label_node.text.strip()
            labels[concept_id] = label
            lexical_rows.append((concept_id, label))
        for synonym in descriptor.findall("./SYNONYMS/SYNONYM"):
            if synonym.text:
                lexical_rows.append((concept_id, synonym.text.strip()))

    product_ids = _descendants(parents, LANGUAL_PRODUCT_TYPE_ROOT)

    def build(name: str, allowed: set[str]) -> Vocabulary:
        surfaces: dict[str, set[str]] = defaultdict(set)
        preferred_surfaces: dict[str, set[str]] = defaultdict(set)
        for concept_id, label in lexical_rows:
            if concept_id in active_ids and concept_id in allowed:
                key = surface_key(label)
                if key:
                    surfaces[key].add(concept_id)
                    if labels.get(concept_id) == label:
                        preferred_surfaces[key].add(concept_id)
        return Vocabulary(
            name,
            {key: labels[key] for key in allowed if key in active_ids and key in labels},
            {key: frozenset(value) for key, value in preferred_surfaces.items()},
            {key: frozenset(value) for key, value in surfaces.items()},
        )

    return build("LanguaL product-type facet", product_ids), build("LanguaL all active facets", active_ids)


def load_fdc_foundation(path: Path) -> Vocabulary:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("FoundationFoods", payload) if isinstance(payload, dict) else payload
    labels: dict[str, str] = {}
    surfaces: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if not isinstance(record, dict) or "fdcId" not in record:
            continue
        concept_id = f"FDC_{record['fdcId']}"
        label = record.get("description", "").strip()
        if not label:
            continue
        labels[concept_id] = label
        surfaces[surface_key(label)].add(concept_id)
    return Vocabulary(
        "FoodData Central Foundation Foods",
        labels,
        {key: frozenset(value) for key, value in surfaces.items()},
        {key: frozenset(value) for key, value in surfaces.items()},
    )


def load_yummly(dataset_root: Path) -> dict[str, list[dict[str, object]]]:
    result = {}
    for split in ("train", "val", "test"):
        with (dataset_root / split / "metadata.json").open(encoding="utf-8") as handle:
            result[split] = json.load(handle)
    return result


def _candidate_ids(vocabulary: Vocabulary, surface: str) -> frozenset[str]:
    key = surface_key(surface)
    preferred = vocabulary.preferred_surface_to_ids.get(key, frozenset())
    if len(preferred) == 1:
        return preferred
    return vocabulary.surface_to_ids.get(key, frozenset())


def _unique_match(vocabulary: Vocabulary, surface: str) -> str | None:
    candidates = _candidate_ids(vocabulary, surface)
    return next(iter(candidates)) if len(candidates) == 1 else None


def _plural_candidates(surface: str) -> Iterable[str]:
    words = surface.split()
    if not words:
        return
    last = words[-1]
    stems: list[str] = []
    if last.endswith("ies") and len(last) > 3:
        stems.append(last[:-3] + "y")
    if last.endswith("es") and len(last) > 2:
        stems.append(last[:-2])
    if last.endswith("s") and not last.endswith("ss") and len(last) > 1:
        stems.append(last[:-1])
    for stem in stems:
        yield " ".join([*words[:-1], stem])


def associate_line(vocabulary: Vocabulary, raw_line: object) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    direct_surface = _normalize_source_text(raw_line)
    if not direct_surface:
        return "unmatched", (), ()
    candidates = _candidate_ids(vocabulary, direct_surface)
    if len(candidates) == 1:
        return "direct", tuple(candidates), ()
    direct_is_ambiguous = len(candidates) > 1

    fallback_targets = normalize_ingredient_targets(raw_line)
    matched: list[str] = []
    standalone: list[str] = []
    for target in fallback_targets or (direct_surface,):
        candidate_ids = _candidate_ids(vocabulary, target)
        if len(candidate_ids) == 1:
            matched.extend(candidate_ids)
            continue
        plural_hits = {
            concept_id
            for singular in _plural_candidates(target)
            for concept_id in _candidate_ids(vocabulary, singular)
        }
        if len(plural_hits) == 1:
            matched.extend(plural_hits)
        else:
            standalone.append(target)
    if matched and not standalone:
        return "fallback", tuple(sorted(set(matched))), ()
    if direct_is_ambiguous:
        return "ambiguous", (), tuple(sorted(set(standalone or (direct_surface,))))
    return "unmatched", tuple(sorted(set(matched))), tuple(sorted(set(standalone)))


def evaluate(
    vocabulary: Vocabulary,
    records: list[dict[str, object]],
    association_cache: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] | None = None,
) -> tuple[Coverage, Counter[str], dict[str, object]]:
    association_cache = association_cache if association_cache is not None else {}
    occurrence_counts = Counter()
    unique_statuses: dict[str, str] = {}
    unmatched = Counter()
    ambiguous = Counter()
    recipe_concepts: list[set[str]] = []
    for record in records:
        concepts: set[str] = set()
        for raw_line in record.get("ingredients", []):
            direct_surface = _normalize_source_text(raw_line) or ""
            cache_key = raw_line if isinstance(raw_line, str) else repr(raw_line)
            if cache_key not in association_cache:
                association_cache[cache_key] = associate_line(vocabulary, raw_line)
            status, matched, standalone = association_cache[cache_key]
            occurrence_counts[status] += 1
            if status == "ambiguous":
                ambiguous[direct_surface] += 1
            unique_statuses.setdefault(direct_surface, status)
            if status == "direct" or status == "fallback":
                unique_statuses[direct_surface] = status
            for concept_id in matched:
                concepts.add(f"external:{concept_id}")
            for target in standalone:
                concepts.add(f"local:{target}")
                unmatched[target] += 1
        recipe_concepts.append(concepts)

    support = Counter(concept for concepts in recipe_concepts for concept in concepts)
    supported = {concept for concept, count in support.items() if count >= 500}
    retained = sum(len(concepts & supported) >= 3 for concepts in recipe_concepts)
    support_sensitivity = []
    total_assignments = sum(support.values())
    for threshold in (1, 10, 25, 50, 100, 250, 500):
        threshold_concepts = {concept for concept, count in support.items() if count >= threshold}
        retained_assignments = sum(support[concept] for concept in threshold_concepts)
        support_sensitivity.append(
            {
                "minimum_recipe_support": threshold,
                "concepts": len(threshold_concepts),
                "external_concepts": sum(key.startswith("external:") for key in threshold_concepts),
                "local_concepts": sum(key.startswith("local:") for key in threshold_concepts),
                "recipes_with_at_least_three_concepts": sum(
                    len(concepts & threshold_concepts) >= 3 for concepts in recipe_concepts
                ),
                "retained_concept_assignments": retained_assignments,
                "retained_assignment_fraction": retained_assignments / total_assignments,
            }
        )

    tracked_support = {}
    for surface in (
        "english muffins",
        "fish fillets",
        "chicken broth",
        "tomato sauce",
        "garam masala",
        "mirin",
        "gochujang",
    ):
        status, matched, standalone = associate_line(vocabulary, surface)
        concept_keys = [*(f"external:{item}" for item in matched), *(f"local:{item}" for item in standalone)]
        tracked_support[surface] = {
            "status": status,
            "concepts": concept_keys,
            "train_recipe_support": {key: support.get(key, 0) for key in concept_keys},
        }
    threshold_summary = {
        "concepts_before_threshold": len(support),
        "external_concepts_before_threshold": sum(key.startswith("external:") for key in support),
        "local_concepts_before_threshold": sum(key.startswith("local:") for key in support),
        "concepts_at_support_500": len(supported),
        "external_concepts_at_support_500": sum(key.startswith("external:") for key in supported),
        "local_concepts_at_support_500": sum(key.startswith("local:") for key in supported),
        "recipes_with_at_least_three_supported_concepts": retained,
        "recipes_total": len(records),
        "support_sensitivity": support_sensitivity,
        "tracked_concept_support": tracked_support,
        "top_ambiguous_source_terms": ambiguous.most_common(50),
    }
    coverage = Coverage(
        vocabulary=vocabulary.name,
        line_occurrences=sum(occurrence_counts.values()),
        direct_occurrences=occurrence_counts["direct"],
        fallback_occurrences=occurrence_counts["fallback"],
        unmatched_occurrences=occurrence_counts["unmatched"],
        ambiguous_occurrences=occurrence_counts["ambiguous"],
        unique_source_terms=len(unique_statuses),
        direct_unique_terms=sum(value == "direct" for value in unique_statuses.values()),
        fallback_unique_terms=sum(value == "fallback" for value in unique_statuses.values()),
        unmatched_unique_terms=sum(value == "unmatched" for value in unique_statuses.values()),
        ambiguous_unique_terms=sum(value == "ambiguous" for value in unique_statuses.values()),
    )
    return coverage, unmatched, threshold_summary


def edge_case_report(vocabularies: list[Vocabulary]) -> list[dict[str, object]]:
    cases = [
        "2 English muffins, split",
        "4 white fish fillets",
        "chicken stock",
        "low-sodium chicken broth",
        "dark soy sauce",
        "tomato paste",
        "tomato sauce",
        "fresh tomatoes",
        "red pepper flakes",
        "green bell peppers",
        "Greek yogurt",
        "toasted sesame oil",
        "gochujang",
        "mirin",
        "garam masala",
    ]
    rows = []
    for raw_line in cases:
        row: dict[str, object] = {"raw_line": raw_line, "direct_surface": _normalize_source_text(raw_line)}
        for vocabulary in vocabularies:
            status, matched, standalone = associate_line(vocabulary, raw_line)
            row[vocabulary.name] = {
                "status": status,
                "matched": [vocabulary.preferred_labels.get(item, item) for item in matched],
                "standalone": list(standalone),
            }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/input/yummly"))
    parser.add_argument("--foodon-tsv", type=Path, required=True)
    parser.add_argument("--langual-xml", type=Path)
    parser.add_argument("--fdc-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records_by_split = load_yummly(args.dataset_root)
    train_records = records_by_split["train"]
    all_records = [record for split in ("train", "val", "test") for record in records_by_split[split]]
    vocabularies = [load_foodon(args.foodon_tsv)]
    if args.langual_xml:
        vocabularies.extend(load_langual(args.langual_xml))
    if args.fdc_json:
        vocabularies.append(load_fdc_foundation(args.fdc_json))

    report: dict[str, object] = {
        "method": {
            "direct": "Exact, unambiguous label or exact-synonym match after mechanical recipe-line parsing only.",
            "fallback": "Exact match after the existing bounded standardizer plus vocabulary-validated final-token singularization.",
            "unmatched": "Retained as a local standalone concept; never discarded or mapped to UNK.",
            "threshold": "Support >= 500 recipes and >= 3 supported concepts, evaluated on the legacy train split only.",
        },
        "corpus": {key: len(value) for key, value in records_by_split.items()},
        "vocabularies": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for vocabulary in vocabularies:
        association_cache: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {}
        train_coverage, train_unmatched, train_threshold = evaluate(
            vocabulary, train_records, association_cache
        )
        full_coverage, _, _ = evaluate(vocabulary, all_records, association_cache)
        report["vocabularies"][vocabulary.name] = {
            "vocabulary_concepts": len(vocabulary.preferred_labels),
            "unambiguous_surfaces": sum(len(ids) == 1 for ids in vocabulary.surface_to_ids.values()),
            "ambiguous_surfaces": sum(len(ids) > 1 for ids in vocabulary.surface_to_ids.values()),
            "train_coverage": asdict(train_coverage),
            "full_corpus_coverage_diagnostic": asdict(full_coverage),
            "train_threshold_effect": train_threshold,
            "top_train_unmatched_standalone_terms": train_unmatched.most_common(100),
        }

    with (args.output_dir / "aggregate_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with (args.output_dir / "edge_cases.json").open("w", encoding="utf-8") as handle:
        json.dump(edge_case_report(vocabularies), handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
