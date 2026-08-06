"""Evaluate whether bounded fuzzy FoodOn recovery is safe for Yummly.

This is research-only code for Data work package 2.2c.  It evaluates fuzzy
matching only after the existing exact-plus-fallback protocol has left a local
concept.  It writes aggregate counts and a small fixed diagnostic set; it does
not generate metadata or persist a raw-line mapping table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from controlled_vocabulary_research import (  # noqa: E402
    Vocabulary,
    _normalize_source_text,
    associate_line,
    load_foodon,
    load_yummly,
    surface_key,
)


@dataclass(frozen=True)
class MatchResult:
    status: str
    concept_id: str | None
    distance: int | None
    competing_concepts: int


def damerau_levenshtein(left: str, right: str) -> int:
    """Return the unrestricted Damerau-Levenshtein edit distance."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    alphabet = {character: index + 1 for index, character in enumerate(set(left + right))}
    rows, columns = len(left) + 2, len(right) + 2
    matrix = [[0] * columns for _ in range(rows)]
    maximum = len(left) + len(right)
    matrix[0][0] = maximum
    for index in range(len(left) + 1):
        matrix[index + 1][0] = maximum
        matrix[index + 1][1] = index
    for index in range(len(right) + 1):
        matrix[0][index + 1] = maximum
        matrix[1][index + 1] = index

    last_seen = defaultdict(int)
    for left_index, left_character in enumerate(left, start=1):
        previous_match_column = 0
        for right_index, right_character in enumerate(right, start=1):
            previous_match_row = last_seen[right_character]
            substitution_column = previous_match_column
            cost = 0 if left_character == right_character else 1
            if cost == 0:
                previous_match_column = right_index
            matrix[left_index + 1][right_index + 1] = min(
                matrix[left_index][right_index] + cost,
                matrix[left_index + 1][right_index] + 1,
                matrix[left_index][right_index + 1] + 1,
                matrix[previous_match_row][substitution_column]
                + (left_index - previous_match_row - 1)
                + 1
                + (right_index - substitution_column - 1),
            )
        last_seen[left_character] = left_index
    return matrix[-1][-1]


def character_bigrams(word: str) -> set[str]:
    return {word[index : index + 2] for index in range(len(word) - 1)}


class BoundedTokenMatcher:
    """Typo-only fuzzy matcher with a deliberately narrow candidate space.

    It considers only ontology surfaces with the same number and order of
    tokens, and only one changed token.  This explicitly excludes substring,
    token-set, word-order, semantic, and hierarchy-based matching.
    """

    def __init__(self, vocabulary: Vocabulary, maximum_distance: int = 2) -> None:
        self.vocabulary = vocabulary
        self.maximum_distance = maximum_distance
        self.word_bigrams: dict[str, set[str]] = defaultdict(set)
        self.patterns: dict[tuple[str, ...], dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )

        for surface in vocabulary.surface_to_ids:
            tokens = tuple(surface.split())
            for index, word in enumerate(tokens):
                pattern = (*tokens[:index], "*", *tokens[index + 1 :])
                self.patterns[pattern][word].add(surface)
                for bigram in character_bigrams(word):
                    self.word_bigrams[bigram].add(word)

    def match(self, value: str, maximum_distance: int, minimum_word_length: int) -> MatchResult:
        query = surface_key(value)
        tokens = tuple(query.split())
        if not tokens or query in self.vocabulary.surface_to_ids:
            return MatchResult("not-applicable", None, None, 0)

        best_by_concept: dict[str, int] = {}
        for index, word in enumerate(tokens):
            if len(word) < minimum_word_length:
                continue
            pattern = (*tokens[:index], "*", *tokens[index + 1 :])
            candidate_words = self.patterns.get(pattern, {})
            if not candidate_words:
                continue
            bigrams = character_bigrams(word)
            minimum_shared_bigrams = max(1, len(bigrams) - 2 * maximum_distance)
            candidate_counts: Counter[str] = Counter(
                candidate
                for bigram in bigrams
                for candidate in self.word_bigrams.get(bigram, ())
            )
            nearby_words = {
                candidate
                for candidate, shared_bigrams in candidate_counts.items()
                if shared_bigrams >= minimum_shared_bigrams
                and candidate in candidate_words
                and abs(len(candidate) - len(word)) <= maximum_distance
            }
            for candidate_word in nearby_words:
                distance = damerau_levenshtein(word, candidate_word)
                if not 0 < distance <= maximum_distance:
                    continue
                for surface in candidate_words.get(candidate_word, ()):
                    for concept_id in self.vocabulary.surface_to_ids[surface]:
                        best_by_concept[concept_id] = min(
                            best_by_concept.get(concept_id, maximum_distance + 1), distance
                        )

        if not best_by_concept:
            return MatchResult("rejected-no-candidate", None, None, 0)
        smallest_distance = min(best_by_concept.values())
        winners = sorted(
            concept_id for concept_id, distance in best_by_concept.items() if distance == smallest_distance
        )
        if len(winners) != 1:
            return MatchResult("rejected-tie", None, smallest_distance, len(winners))
        return MatchResult("accepted", winners[0], smallest_distance, len(best_by_concept))


def _perturbations(surface: str) -> dict[str, str]:
    """Create three deterministic, single-word typo-like variants."""
    tokens = surface.split()
    target_index = max(range(len(tokens)), key=lambda index: (len(tokens[index]), -index))
    word = tokens[target_index]
    candidates = [index for index, character in enumerate(word) if character.isalpha()]
    if len(candidates) < 2:
        return {}
    digest = int(hashlib.sha256(surface.encode("utf-8")).hexdigest(), 16)
    position = candidates[digest % len(candidates)]
    replacement = chr(((ord(word[position]) - ord("a") + 1) % 26) + ord("a"))

    def replace_word(updated: str) -> str:
        values = list(tokens)
        values[target_index] = updated
        return " ".join(values)

    result = {
        "substitution": replace_word(word[:position] + replacement + word[position + 1 :]),
        "deletion": replace_word(word[:position] + word[position + 1 :]),
    }
    transpose_at = candidates[(digest // 31) % (len(candidates) - 1)]
    if transpose_at + 1 < len(word) and word[transpose_at] != word[transpose_at + 1]:
        result["transposition"] = replace_word(
            word[:transpose_at]
            + word[transpose_at + 1]
            + word[transpose_at]
            + word[transpose_at + 2 :]
        )
    return {kind: value for kind, value in result.items() if value != surface}


def prepare_train_terms(
    vocabulary: Vocabulary, records: Iterable[dict[str, object]]
) -> tuple[dict[str, str], Counter[str], Counter[str]]:
    """Derive both benchmark gold terms and unmatched local terms in one pass."""
    gold_terms: dict[str, str] = {}
    occurrences: Counter[str] = Counter()
    recipe_support: Counter[str] = Counter()
    association_cache: dict[object, tuple[str, tuple[str, ...], tuple[str, ...]]] = {}
    for record in records:
        local_terms: set[str] = set()
        for raw_line in record.get("ingredients", []):
            cache_key = raw_line if isinstance(raw_line, str) else repr(raw_line)
            if cache_key not in association_cache:
                association_cache[cache_key] = associate_line(vocabulary, raw_line)
            status, matched, standalone = association_cache[cache_key]
            surface = _normalize_source_text(raw_line)
            if status == "direct" and len(matched) == 1 and surface:
                gold_terms[surface_key(surface)] = matched[0]
            if status == "unmatched":
                terms = {surface_key(term) for term in standalone if surface_key(term)}
                local_terms.update(terms)
                occurrences.update(terms)
        for term in local_terms:
            recipe_support[term] += 1
    return gold_terms, occurrences, recipe_support


def evaluate_gold(
    vocabulary: Vocabulary,
    matcher: BoundedTokenMatcher,
    terms: dict[str, str],
    maximum_distance: int,
    minimum_word_length: int,
) -> dict[str, object]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for surface, expected_id in sorted(terms.items()):
        for kind, variant in _perturbations(surface).items():
            if variant in vocabulary.surface_to_ids:
                continue
            result = matcher.match(variant, maximum_distance, minimum_word_length)
            totals[kind]["tested"] += 1
            totals[kind][result.status] += 1
            if result.status == "accepted":
                if result.concept_id == expected_id:
                    totals[kind]["correct"] += 1
                else:
                    totals[kind]["wrong"] += 1
    combined = Counter()
    for counts in totals.values():
        combined.update(counts)
    accepted = combined["accepted"]
    return {
        "by_perturbation": {key: dict(value) for key, value in sorted(totals.items())},
        "combined": dict(combined),
        "accepted_precision": (combined["correct"] / accepted) if accepted else None,
        "accepted_recall": (combined["correct"] / combined["tested"]) if combined["tested"] else None,
    }


def evaluate_corpus(
    matcher: BoundedTokenMatcher,
    occurrences: Counter[str],
    recipe_support: Counter[str],
    maximum_distance: int,
    minimum_word_length: int,
) -> dict[str, object]:
    results: Counter[str] = Counter()
    occurrence_results: Counter[str] = Counter()
    rescued_support_500 = 0
    rescued_occurrences = 0
    for term, count in occurrences.items():
        result = matcher.match(term, maximum_distance, minimum_word_length)
        results[result.status] += 1
        occurrence_results[result.status] += count
        if result.status == "accepted":
            rescued_occurrences += count
            rescued_support_500 += recipe_support[term] >= 500
    return {
        "unique_local_terms": len(occurrences),
        "local_line_occurrences": sum(occurrences.values()),
        "unique_term_outcomes": dict(results),
        "line_occurrence_outcomes": dict(occurrence_results),
        "rescued_terms_at_train_support_500": rescued_support_500,
        "rescued_line_occurrences": rescued_occurrences,
    }


def profile(
    name: str,
    vocabulary: Vocabulary,
    matcher: BoundedTokenMatcher,
    gold_terms: dict[str, str],
    occurrences: Counter[str],
    recipe_support: Counter[str],
    maximum_distance: int,
    minimum_word_length: int,
) -> dict[str, object]:
    return {
        "name": name,
        "maximum_single_token_damerau_levenshtein_distance": maximum_distance,
        "minimum_changed_word_length": minimum_word_length,
        "gold_typo_benchmark": evaluate_gold(
            vocabulary, matcher, gold_terms, maximum_distance, minimum_word_length
        ),
        "unmatched_local_corpus": evaluate_corpus(
            matcher, occurrences, recipe_support, maximum_distance, minimum_word_length
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/input/yummly"))
    parser.add_argument("--foodon-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    vocabulary = load_foodon(args.foodon_tsv)
    train_records = load_yummly(args.dataset_root)["train"]
    matcher = BoundedTokenMatcher(vocabulary, maximum_distance=2)
    gold_terms, occurrences, recipe_support = prepare_train_terms(vocabulary, train_records)
    profiles = [
        profile(
            "one-edit, changed word length >= 5",
            vocabulary,
            matcher,
            gold_terms,
            occurrences,
            recipe_support,
            maximum_distance=1,
            minimum_word_length=5,
        ),
        profile(
            "up-to-two-edits, changed word length >= 5",
            vocabulary,
            matcher,
            gold_terms,
            occurrences,
            recipe_support,
            maximum_distance=2,
            minimum_word_length=5,
        ),
    ]

    report = {
        "purpose": "Research-only bounded fuzzy FoodOn evaluation after exact-plus-fallback association.",
        "guardrails": [
            "Only terms still unmatched after exact association and fallback are evaluated.",
            "The matcher allows a typo in exactly one aligned token; it never adds/removes/reorders words.",
            "It rejects ties between FoodOn concepts instead of choosing one.",
            "It does not use substring, token-set, semantic, image, cuisine, title, or hierarchy signals.",
            "The gold benchmark uses deterministic synthetic typo perturbations of observed exact FoodOn matches.",
        ],
        "foodon": {
            "input_filename": args.foodon_tsv.name,
            "sha256": hashlib.sha256(args.foodon_tsv.read_bytes()).hexdigest(),
            "preferred_concepts": len(vocabulary.preferred_labels),
            "lexical_surfaces": len(vocabulary.surface_to_ids),
        },
        "train_gold_terms": len(gold_terms),
        "profiles": profiles,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "aggregate_report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
