"""FoodOn-first generation of the shared ``ingredients_target`` vocabulary."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

from src.data_processing.foodon_lexicon import FoodOnLexicon, surface_key, unique_id_with_plural_fallback
from src.data_processing.ingredient_standardization import _normalize_source_text, normalize_ingredient_targets


@dataclass(frozen=True)
class AssociationResult:
    """All concepts emitted by one source ingredient line."""

    status: str
    external_ids: tuple[str, ...]
    local_targets: tuple[str, ...]


@dataclass(frozen=True)
class ControlledTargetGenerationResult:
    targets_by_record: list[list[str]]
    train_support: Mapping[str, int]
    retained_records: list[int]
    association_statuses: Mapping[str, int]
    unfiltered_targets_by_record: list[list[str]]


def associate_ingredient_line(lexicon: FoodOnLexicon, raw_line: object) -> AssociationResult:
    """Associate a line using exact FoodOn, fallback normalization, and exact retry."""
    direct_surface = _normalize_source_text(raw_line)
    if not direct_surface:
        return AssociationResult("unmatched", (), ())

    direct_candidates = lexicon.candidate_ids(direct_surface)
    if len(direct_candidates) == 1:
        return AssociationResult("direct", tuple(sorted(direct_candidates)), ())

    fallback_targets = normalize_ingredient_targets(raw_line)
    external_ids: set[str] = set()
    local_targets: set[str] = set()
    for target in fallback_targets or (direct_surface,):
        concept_id = unique_id_with_plural_fallback(lexicon, target)
        if concept_id is not None:
            external_ids.add(concept_id)
        else:
            canonical_local = surface_key(target)
            if canonical_local:
                local_targets.add(canonical_local)

    if external_ids and not local_targets:
        status = "fallback" if len(direct_candidates) == 0 else "ambiguous_resolved_by_fallback"
    elif direct_candidates:
        status = "ambiguous"
    else:
        status = "unmatched"
    return AssociationResult(status, tuple(sorted(external_ids)), tuple(sorted(local_targets)))


def record_concepts(
    lexicon: FoodOnLexicon,
    record: Mapping[str, object],
    cache: dict[str, AssociationResult] | None = None,
) -> tuple[list[str], Counter[str]]:
    """Return canonical strings and aggregate association statuses for a record."""
    cache = cache if cache is not None else {}
    concepts: set[str] = set()
    statuses: Counter[str] = Counter()
    ingredients = record.get("ingredients", ())
    if not isinstance(ingredients, list):
        return [], statuses
    for raw_line in ingredients:
        cache_key = raw_line if isinstance(raw_line, str) else repr(raw_line)
        if cache_key not in cache:
            cache[cache_key] = associate_ingredient_line(lexicon, raw_line)
        association = cache[cache_key]
        statuses[association.status] += 1
        concepts.update(lexicon.canonical_label(concept_id) for concept_id in association.external_ids)
        concepts.update(association.local_targets)
    return sorted(concepts), statuses


def derive_controlled_targets(
    records: Iterable[Mapping[str, object]],
    train_records: Iterable[Mapping[str, object]],
    lexicon: FoodOnLexicon,
    min_recipe_support: int = 500,
    min_targets_per_recipe: int = 3,
) -> ControlledTargetGenerationResult:
    """Map all records while deriving support and retention from train only."""
    if min_recipe_support < 1 or min_targets_per_recipe < 1:
        raise ValueError("support and minimum target thresholds must be positive")

    records = list(records)
    train_records = list(train_records)
    cache: dict[str, AssociationResult] = {}
    all_targets: list[list[str]] = []
    statuses: Counter[str] = Counter()
    for record in records:
        targets, record_statuses = record_concepts(lexicon, record, cache)
        all_targets.append(targets)
        statuses.update(record_statuses)

    train_targets: list[list[str]] = []
    for record in train_records:
        targets, _ = record_concepts(lexicon, record, cache)
        train_targets.append(targets)
    train_support = Counter(target for targets in train_targets for target in set(targets))
    retained_vocabulary = {target for target, support in train_support.items() if support >= min_recipe_support}
    targets_by_record = [
        [target for target in targets if target in retained_vocabulary]
        for targets in all_targets
    ]
    retained_records = [
        index for index, targets in enumerate(targets_by_record) if len(targets) >= min_targets_per_recipe
    ]
    return ControlledTargetGenerationResult(
        targets_by_record,
        dict(sorted(train_support.items())),
        retained_records,
        dict(sorted(statuses.items())),
        all_targets,
    )
