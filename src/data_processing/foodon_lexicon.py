"""Offline FoodOn lexical index used by the controlled target generator.

The project stores a compact, generated index rather than querying an ontology
service while building metadata.  The index is restricted to the FoodOn
``food product`` descendant branch and keeps preferred labels separate from
exact synonyms so that an unambiguous preferred label wins deterministically.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

FOODON_PRODUCT_ROOT = "FOODON_00001002"
DEFAULT_INDEX_PATH = Path(__file__).with_name("resources") / "foodon_food_product_v2025_07_31.json"
_CURIE = re.compile(r"/([^/<>]+)>?$")
_QUOTED_VALUE = re.compile(r'^"(.*)"(?:@[A-Za-z-]+)?$')


def surface_key(value: str) -> str:
    """Return the deterministic lexical key used for FoodOn and local terms."""
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


def _descendants(parents: Mapping[str, set[str]], root: str) -> set[str]:
    children: dict[str, set[str]] = defaultdict(set)
    for child, parent_ids in parents.items():
        for parent in parent_ids:
            children[parent].add(child)
    result = {root}
    frontier = [root]
    while frontier:
        parent = frontier.pop()
        for child in children.get(parent, ()):
            if child not in result:
                result.add(child)
                frontier.append(child)
    return result


@dataclass(frozen=True)
class FoodOnLexicon:
    preferred_labels: Mapping[str, str]
    preferred_surface_to_ids: Mapping[str, frozenset[str]]
    surface_to_ids: Mapping[str, frozenset[str]]

    @property
    def concept_count(self) -> int:
        return len(self.preferred_labels)

    def candidate_ids(self, surface: str) -> frozenset[str]:
        key = surface_key(surface)
        preferred = self.preferred_surface_to_ids.get(key, frozenset())
        if len(preferred) == 1:
            return preferred
        return self.surface_to_ids.get(key, frozenset())

    def unique_id(self, surface: str) -> str | None:
        candidates = self.candidate_ids(surface)
        return next(iter(candidates)) if len(candidates) == 1 else None

    def canonical_label(self, concept_id: str) -> str:
        """Return the stable target string for a FoodOn concept."""
        return surface_key(self.preferred_labels[concept_id])


def _freeze(payload: Mapping[str, object]) -> FoodOnLexicon:
    preferred_labels = {str(key): str(value) for key, value in payload["preferred_labels"].items()}
    preferred = {
        str(key): frozenset(str(item) for item in values)
        for key, values in payload["preferred_surface_to_ids"].items()
    }
    surfaces = {
        str(key): frozenset(str(item) for item in values)
        for key, values in payload["surface_to_ids"].items()
    }
    if not preferred_labels or not surfaces:
        raise ValueError("FoodOn index is empty")
    return FoodOnLexicon(preferred_labels, preferred, surfaces)


def load_packaged_foodon(path: Path | None = None) -> FoodOnLexicon:
    """Load the pinned compact index bundled with the project."""
    index_path = (path or DEFAULT_INDEX_PATH).resolve()
    with index_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("branch_root") != FOODON_PRODUCT_ROOT:
        raise ValueError(f"unexpected FoodOn branch in {index_path}")
    return _freeze(payload)


def load_foodon_tsv(path: Path) -> tuple[FoodOnLexicon, dict[str, object]]:
    """Build an index from the pinned root ``foodon-synonyms.tsv`` export."""
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
    surfaces: dict[str, set[str]] = defaultdict(set)
    preferred_surfaces: dict[str, set[str]] = defaultdict(set)
    for concept_id, kind, label in lexical_rows:
        if concept_id not in product_ids or kind not in {"label", "synonym (exact)"}:
            continue
        key = surface_key(label)
        if not key:
            continue
        surfaces[key].add(concept_id)
        if kind == "label":
            preferred_surfaces[key].add(concept_id)
            if key.endswith(" food product"):
                surfaces[key.removesuffix(" food product")].add(concept_id)

    lexicon = FoodOnLexicon(
        {key: labels[key] for key in product_ids if key in labels},
        {key: frozenset(value) for key, value in preferred_surfaces.items()},
        {key: frozenset(value) for key, value in surfaces.items()},
    )
    provenance = {
        "source_file": path.name,
        "branch_root": FOODON_PRODUCT_ROOT,
        "preferred_concepts": lexicon.concept_count,
        "lexical_surfaces": len(lexicon.surface_to_ids),
    }
    return lexicon, provenance


def write_index(lexicon: FoodOnLexicon, output: Path, provenance: Mapping[str, object]) -> None:
    """Write a compact deterministic index for source-controlled use."""
    payload = {
        "format_version": 1,
        **dict(provenance),
        "preferred_labels": dict(sorted(lexicon.preferred_labels.items())),
        "preferred_surface_to_ids": {
            key: sorted(value) for key, value in sorted(lexicon.preferred_surface_to_ids.items())
        },
        "surface_to_ids": {key: sorted(value) for key, value in sorted(lexicon.surface_to_ids.items())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


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


def unique_id_with_plural_fallback(lexicon: FoodOnLexicon, surface: str) -> str | None:
    """Use only exact lexical keys and a vocabulary-validated final plural rule."""
    direct = lexicon.unique_id(surface)
    if direct:
        return direct
    candidates = {
        concept_id
        for singular in _plural_candidates(surface_key(surface))
        for concept_id in (lexicon.unique_id(singular),)
        if concept_id is not None
    }
    return next(iter(candidates)) if len(candidates) == 1 else None
