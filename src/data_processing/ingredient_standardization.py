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
    r"frozen|defrosted|washed|prepared|homemade|store bought|low fat|reduced fat|nonfat|boneless|skinless)\b"
)
_NON_WORD = re.compile(r"[^a-z0-9]+")
_SALT_AND_PEPPER_WORDS = frozenset(
    {"salt", "pepper", "and", "sea", "kosher", "table", "coarse", "black", "white", "ground", "fresh", "freshly", "cracked"}
)
_GREEK_YOGURT_VARIANT = re.compile(
    r"(?:(?:nonfat|low fat|lowfat|fat free|whole milk|full fat) )*(?:plain )?greek(?: style)? yog(?:h)?urt\Z"
)


def _exact_patterns(*phrases: str) -> re.Pattern[str]:
    """Return a whole-normalized-phrase matcher for explicit source forms."""
    return re.compile(r"(?:" + "|".join(re.escape(phrase) for phrase in phrases) + r")\Z")

# These rules run before generic preparation-word removal.  They therefore
# retain the meaningful evidence in forms such as ``fresh coriander``,
# ``toasted sesame oil``, and ``crushed red pepper``.  Every matcher is a
# whole normalized phrase; no rule relies on an unbounded substring match.
_QUALIFIER_SENSITIVE_ALIASES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (_exact_patterns("salt and pepper"), ("salt", "pepper")),
    (_exact_patterns("fresh coriander", "fresh coriander leaf", "fresh coriander leaves", "fresh cilantro", "fresh cilantro leaf", "fresh cilantro leaves", "coriander leaf", "coriander leaves", "cilantro leaf", "cilantro leaves"), ("cilantro",)),
    (_exact_patterns("toasted sesame oil"), ("sesame oil",)),
    (_exact_patterns("low sodium chicken broth"), ("chicken broth",)),
    (_exact_patterns("light soy sauce", "dark soy sauce", "low sodium soy sauce"), ("soy sauce",)),
    (_exact_patterns("plain yogurt", "plain yoghurt", "greek yogurt", "greek yoghurt", "plain greek yogurt", "plain greek yoghurt"), ("yogurt",)),
    (_exact_patterns("ground cumin", "freshly ground cumin", "cumin powder"), ("cumin",)),
    (_exact_patterns("ground coriander", "freshly ground coriander", "coriander powder"), ("cilantro",)),
    (_exact_patterns("ground turmeric", "freshly ground turmeric", "turmeric powder"), ("turmeric",)),
    (_exact_patterns("ground cinnamon", "freshly ground cinnamon", "cinnamon powder"), ("cinnamon",)),
    (_exact_patterns("ground nutmeg", "freshly ground nutmeg", "nutmeg powder"), ("nutmeg",)),
    (_exact_patterns("ground ginger", "freshly ground ginger", "ginger powder"), ("ginger",)),
    (_exact_patterns("black pepper", "white pepper", "ground black pepper", "ground white pepper", "ground pepper", "freshly ground black pepper", "freshly ground white pepper", "freshly ground pepper", "cracked black pepper"), ("pepper",)),
    (_exact_patterns("chili powder", "chile powder", "red chili powder", "red chile powder", "hot chili powder", "hot chile powder", "ground red pepper", "crushed red pepper", "dried crushed red pepper", "red pepper flakes", "crushed red pepper flakes", "dried crushed red pepper flakes", "hot red pepper flakes", "chili flakes", "chile flakes"), ("chili",)),
)

# Explicit phrase aliases and generalizations adopted from the useful portions
# of the historical attempt and strengthened in Work package 2.2b.  Rules are
# deliberately exact after generic preparation removal, so unrelated words
# such as ``pineapple`` and ``watercress`` cannot trigger a target.
_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (_exact_patterns("all purpose flour", "plain flour"), "flour"),
    (_exact_patterns("garlic clove", "garlic cloves", "clove of garlic", "cloves of garlic", "garlic paste"), "garlic"),
    (_exact_patterns("sea salt", "kosher salt", "table salt", "coarse salt"), "salt"),
    (_exact_patterns("extra virgin olive oil", "olive oil"), "olive oil"),
    (_exact_patterns("scallion", "scallions", "green onion", "green onions", "spring onion", "spring onions"), "green onion"),
    (_exact_patterns("coriander", "coriander leaf", "coriander leaves", "cilantro", "cilantro leaf", "cilantro leaves"), "cilantro"),
    (_exact_patterns("confectioner sugar", "confectioners sugar", "powdered sugar", "icing sugar", "white sugar", "granulated sugar", "brown sugar", "light brown sugar"), "sugar"),
    (_exact_patterns("chicken breast", "chicken breasts", "chicken thigh", "chicken thighs", "chicken tenderloin", "chicken tenderloins"), "chicken"),
    (_exact_patterns("ground beef", "lean ground beef"), "beef"),
    (_exact_patterns("ground pork", "lean ground pork"), "pork"),
    (_exact_patterns("large egg", "large eggs", "extra large egg", "extra large eggs"), "egg"),
    (_exact_patterns("black pepper", "white pepper", "ground pepper"), "pepper"),
    (_exact_patterns("bay leaves"), "bay leaf"),
    (_exact_patterns("warm water", "cold water"), "water"),
    (_exact_patterns("basil leaves"), "basil"),
    (_exact_patterns("mint leaves"), "mint"),
    (_exact_patterns("thyme leaves"), "thyme"),
    (_exact_patterns("flat leaf parsley"), "parsley"),
    (_exact_patterns("ginger root"), "ginger"),
    (_exact_patterns("purple onion", "red onion"), "red onion"),
    (_exact_patterns("white onion", "yellow onion", "spanish onion", "sweet onion"), "onion"),
    (_exact_patterns("cayenne"), "cayenne pepper"),
    (_exact_patterns("cider vinegar"), "apple cider vinegar"),
    (_exact_patterns("cooking oil"), "oil"),
    (_exact_patterns("whole milk"), "milk"),
    (_exact_patterns("chicken stock", "chicken broth"), "chicken broth"),
    (_exact_patterns("soy sauce"), "soy sauce"),
    (_exact_patterns("sesame oil"), "sesame oil"),
    (_exact_patterns("yogurt", "yoghurt", "plain yogurt", "plain yoghurt", "greek yogurt", "greek yoghurt", "plain greek yogurt", "plain greek yoghurt"), "yogurt"),
    (_exact_patterns("tomato paste", "tomato sauce"), "tomato sauce"),
    (_exact_patterns("red pepper"), "red bell pepper"),
    (_exact_patterns("green pepper"), "green bell pepper"),
)

_PASTA_SHAPES = _exact_patterns(
    "spaghetti", "penne", "fusilli", "rigatoni", "linguine", "tagliatelle", "fettuccine", "macaroni", "farfalle", "lasagna"
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
    "bananas": "banana",
    "almonds": "almond",
    "pecans": "pecan",
    "walnuts": "walnut",
}


@dataclass(frozen=True)
class TargetStandardizationResult:
    targets_by_record: list[list[str]]
    recipe_support: Mapping[str, int]
    retained_records: list[int]


def _normalize_source_text(raw_line: object) -> str | None:
    """Remove mechanical recipe-line syntax while retaining semantic qualifiers."""
    if not isinstance(raw_line, str):
        return None
    ingredient = unicodedata.normalize("NFKD", raw_line).encode("ascii", "ignore").decode("ascii").lower()
    ingredient = _PARENTHETICAL.sub(" ", ingredient)
    ingredient = ingredient.split(",", maxsplit=1)[0]
    ingredient = _TRAILING_PREPARATION.sub("", ingredient)
    ingredient = ingredient.replace("&", " and ")
    for _ in range(3):
        updated = _LEADING_QUANTITY.sub("", ingredient)
        updated = _LEADING_UNIT.sub("", updated)
        if updated == ingredient:
            break
        ingredient = updated
    ingredient = re.sub(r"^of\s+", "", ingredient)
    ingredient = _NON_WORD.sub(" ", ingredient).strip()
    return ingredient or None


def _is_salt_and_pepper_composite(ingredient: str) -> bool:
    """Recognize only bounded salt-and-pepper lists with harmless descriptors."""
    words = ingredient.split()
    return (
        "salt" in words
        and "pepper" in words
        and set(words).issubset(_SALT_AND_PEPPER_WORDS)
    )


def normalize_ingredient_targets(raw_line: object) -> tuple[str, ...]:
    """Return zero, one, or multiple explicit targets for one raw ingredient line."""
    ingredient = _normalize_source_text(raw_line)
    if not ingredient:
        return ()
    if _GREEK_YOGURT_VARIANT.fullmatch(ingredient):
        return ("yogurt",)
    for pattern, replacements in _QUALIFIER_SENSITIVE_ALIASES:
        if pattern.fullmatch(ingredient):
            return replacements

    ingredient = _PREPARATION_WORDS.sub(" ", ingredient)
    ingredient = _NON_WORD.sub(" ", ingredient).strip()
    if not ingredient or ingredient == "sauce":
        return ()
    if _is_salt_and_pepper_composite(ingredient):
        return ("salt", "pepper")
    for pattern, replacement in _ALIASES:
        if pattern.fullmatch(ingredient):
            return (replacement,)
    if _PASTA_SHAPES.fullmatch(ingredient):
        return ("pasta",)
    if re.search(r"\b(?:cheese|queso)\b", ingredient):
        return ("cheese",)
    if ingredient in _SINGULARS:
        return (_SINGULARS[ingredient],)
    return (ingredient,)


def normalize_ingredient_line(raw_line: object) -> str | None:
    """Return the sole target for a line, or ``None`` when it emits zero/many."""
    targets = normalize_ingredient_targets(raw_line)
    return targets[0] if len(targets) == 1 else None


def normalized_recipe_targets(ingredients: object) -> list[str]:
    if not isinstance(ingredients, Sequence) or isinstance(ingredients, str):
        return []
    return sorted({target for line in ingredients for target in normalize_ingredient_targets(line)})


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
