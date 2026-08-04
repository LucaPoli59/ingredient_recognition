"""Deterministic normalization for future Yummly ingredient targets.

The legacy ``ingredients_ok`` field remains untouched.  This module only
derives ``ingredients_target`` from the original recipe ingredient lines.
Rules are intentionally explicit: there is no fuzzy matching, corpus-order
dependent merge, or unbounded substring replacement.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


DEFAULT_MIN_RECIPE_SUPPORT = 500
DEFAULT_MIN_TARGETS_PER_RECIPE = 3

_LEADING_QUANTITY = re.compile(
    r"^\s*(?:\d+(?:[./]\d+)?|\d+\s+\d+/\d+|one|two|three|four|five|half|quarter|pinch|dash)"
    r"(?:\s*(?:-|to)\s*(?:\d+(?:[./]\d+)?|one|two|three|four|five))?\s*",
)
_LEADING_UNIT = re.compile(
    r"^\s*(?:cups?|tablespoons?|tbsp\.?|teaspoons?|tsp\.?|ounces?|oz\.?|pounds?|lbs?\.?|"
    r"grams?|g|kilograms?|kg|millilit(?:er|re)s?|ml|cans?|packages?|bags?|bunch(?:es)?|cloves?|"
    r"slices?|pieces?|heads?|stalks?|sprigs?|leaves?|strips?|large|medium|small)\b\s*",
)
_PARENTHETICAL = re.compile(r"\([^)]*\)|\[[^]]*\]")
_TRAILING_PREPARATION = re.compile(
    r"\b(?:for garnish|for serving|to taste|as needed|divided|plus more|if desired|optional)\b.*$"
)
_PREPARATION_WORDS = re.compile(
    r"\b(?:fresh|dried|chopped|finely|coarsely|roughly|minced|sliced|diced|cubed|crushed|"
    r"grated|shredded|peeled|seeded|drained|rinsed|thawed|softened|melted|unsalted|salted|"
    r"room temperature|at room temperature|packed|trimmed|halved|beaten|cooked|uncooked|"
    r"frozen|defrosted|washed|prepared|homemade|store bought|low fat|reduced fat|nonfat)\b"
)
_NON_WORD = re.compile(r"[^a-z0-9]+")

# Explicit phrase aliases and generalizations adopted from the useful portions
# of the historical attempt.  Longest phrases win; all matching is bounded.
_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:all[- ]purpose|plain) flour\b"), "flour"),
    (re.compile(r"\b(?:cloves? of )?garlic cloves?\b"), "garlic"),
    (re.compile(r"\b(?:sea|kosher|table) salt\b"), "salt"),
    (re.compile(r"\b(?:extra[- ]virgin )?olive oil\b"), "olive oil"),
    (re.compile(r"\b(?:scallions?|green onions?)\b"), "green onion"),
    (re.compile(r"\b(?:cilantro leaves?|fresh coriander)\b"), "cilantro"),
    (re.compile(r"\b(?:confectioners?|powdered|icing) sugar\b"), "sugar"),
    (re.compile(r"\b(?:chicken breasts?|chicken thighs?|chicken tenderloins?)\b"), "chicken"),
    (re.compile(r"\b(?:ground|lean ground) beef\b"), "beef"),
    (re.compile(r"\b(?:ground|lean ground) pork\b"), "pork"),
    (re.compile(r"\b(?:large|extra large) eggs?\b"), "egg"),
    (re.compile(r"\b(?:black|white|ground) pepper\b"), "pepper"),
    (re.compile(r"\b(?:chile|chili) powder\b"), "chili powder"),
    (re.compile(r"\b(?:grated |shredded |crumbled )?.*\b(?:cheese|queso)\b"), "cheese"),
    (re.compile(r"\b(?:spaghetti|penne|fusilli|rigatoni|linguine|tagliatelle|fettuccine|macaroni|farfalle|lasagna)\b"), "pasta"),
)

_SINGULARS = {
    "apples": "apple",
    "eggs": "egg",
    "onions": "onion",
    "tomatoes": "tomato",
    "potatoes": "potato",
    "carrots": "carrot",
    "strawberries": "strawberry",
    "blueberries": "blueberry",
    "raspberries": "raspberry",
    "mushrooms": "mushroom",
    "beans": "bean",
    "peppers": "pepper",
    "limes": "lime",
    "lemons": "lemon",
}


@dataclass(frozen=True)
class TargetStandardizationResult:
    targets_by_record: list[list[str]]
    recipe_support: Mapping[str, int]
    retained_records: list[int]


def normalize_ingredient_line(raw_line: object) -> str | None:
    """Normalize one raw ingredient line with only explicit, token-bounded rules."""
    if not isinstance(raw_line, str):
        return None
    ingredient = unicodedata.normalize("NFKD", raw_line).encode("ascii", "ignore").decode("ascii").lower()
    ingredient = _PARENTHETICAL.sub(" ", ingredient)
    ingredient = ingredient.split(",", maxsplit=1)[0]
    ingredient = _TRAILING_PREPARATION.sub("", ingredient)
    for _ in range(3):
        updated = _LEADING_QUANTITY.sub("", ingredient)
        updated = _LEADING_UNIT.sub("", updated)
        if updated == ingredient:
            break
        ingredient = updated
    ingredient = re.sub(r"^of\s+", "", ingredient)
    ingredient = _PREPARATION_WORDS.sub(" ", ingredient)
    ingredient = _NON_WORD.sub(" ", ingredient).strip()
    if not ingredient:
        return None
    for pattern, replacement in _ALIASES:
        if pattern.search(ingredient):
            return replacement
    if ingredient in _SINGULARS:
        return _SINGULARS[ingredient]
    return ingredient


def normalized_recipe_targets(ingredients: object) -> list[str]:
    if not isinstance(ingredients, Sequence) or isinstance(ingredients, str):
        return []
    return sorted({normalized for line in ingredients if (normalized := normalize_ingredient_line(line)) is not None})


def derive_ingredients_target(
    recipes: Iterable[Mapping[str, object]],
    min_recipe_support: int = DEFAULT_MIN_RECIPE_SUPPORT,
    min_targets_per_recipe: int = DEFAULT_MIN_TARGETS_PER_RECIPE,
) -> TargetStandardizationResult:
    """Derive target lists and retain records with sufficient supported targets."""
    if min_recipe_support < 1:
        raise ValueError("min_recipe_support must be positive")
    if min_targets_per_recipe < 1:
        raise ValueError("min_targets_per_recipe must be positive")

    unfiltered_targets = [normalized_recipe_targets(recipe.get("ingredients")) for recipe in recipes]
    recipe_support = Counter(target for targets in unfiltered_targets for target in targets)
    targets_by_record = [
        [target for target in targets if recipe_support[target] >= min_recipe_support]
        for targets in unfiltered_targets
    ]
    retained_records = [
        index for index, targets in enumerate(targets_by_record) if len(targets) >= min_targets_per_recipe
    ]
    return TargetStandardizationResult(targets_by_record, dict(sorted(recipe_support.items())), retained_records)
