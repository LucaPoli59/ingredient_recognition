# Ingredient target mapping rules

**Created:** 2026-08-04  
**Last updated:** 2026-08-04

## Purpose and authority

This document is the durable registry of custom semantic mappings used to derive the Yummly `ingredients_target` field from the original `ingredients` lines. It records both the rules implemented by the first candidate standardizer and the post-audit rules implemented in `ingredients_target_v4_metadata.json`.

`v4` is a tested, reproducible baseline rather than the selected next benchmark generation. Its frequency filter runs before controlled-concept association and can remove valid ingredients. [Work package 2.2c of the Data plan](../plans/yummly_data_phase.md#work-package-22c--controlled-vocabulary-research) is selecting the replacement vocabulary and Work package 2.2d will implement it. Until then, this registry remains authoritative only for the retained baseline rules; a vocabulary decision must update this document, the standardizer, and its tests together.

This registry remains part of the implementation documentation after the feature plan is completed. Future changes to ingredient mappings must update this document, the standardizer, and the corresponding regression tests in the same change. Execution status belongs in [`../plans/yummly_data_phase.md`](../plans/yummly_data_phase.md); audit evidence and decision rationale belong in [`../project_objective/ingredient_vocabulary_audit.md`](../project_objective/ingredient_vocabulary_audit.md).

The lifecycle labels used below are:

- **Active in `v1`**: implemented by [`ingredient_standardization.py`](../../src/data_processing/ingredient_standardization.py) and used to generate `ingredients_target_v1_metadata.json`.
- **Baseline in `v4`**: implemented by [`ingredient_standardization.py`](../../src/data_processing/ingredient_standardization.py), covered by [`../../tests/test_ingredient_standardization.py`](../../tests/test_ingredient_standardization.py), and used to generate `ingredients_target_v4_metadata.json`; it is not selected for runtime integration.
- **Preserved boundary**: an explicit non-mapping or collision boundary that implementations must retain.

No rule may be inferred from lexical similarity, edit distance, co-occurrence, or embedding similarity. Matching must be deterministic and token- or phrase-bounded.

## Processing contract

For a new metadata generation, the standardizer must:

1. read the original `ingredients` lines without modifying them;
2. normalize case, Unicode, quantities, units, punctuation, and approved non-semantic preparation text;
3. apply the explicit rules in this registry at a stage where meaningful qualifiers are still available;
4. allow a rule to emit zero, one, or multiple targets;
5. deduplicate and deterministically order the targets within each recipe;
6. compute support by distinct source recipes and retain targets with support of at least 500 recipes;
7. retain recipes with at least three supported targets.

Qualifier-sensitive mappings such as `fresh coriander`, `crushed red pepper`, and `toasted sesame oil` must run before generic descriptor removal can erase the evidence needed to choose the correct target.

## Baseline mappings active in `v1`

These mappings describe the first candidate standardizer. They remain part of the next implementation unless a later approved rule below explicitly supersedes their output.

### Phrase aliases and generalizations

| Input phrase or bounded pattern | Output target | Lifecycle | Notes |
| --- | --- | --- | --- |
| `all-purpose flour`, `all purpose flour`, `plain flour` | `flour` | Active in `v1` | Flour subtype is removed. |
| `garlic clove`, `garlic cloves`, `clove of garlic`, `cloves of garlic` | `garlic` | Active in `v1` | Quantity and unit removal may already reduce some forms. |
| `sea salt`, `kosher salt`, `table salt` | `salt` | Active in `v1` | `coarse salt` is added by 2.2b below. |
| `extra-virgin olive oil`, `extra virgin olive oil`, `olive oil` | `olive oil` | Active in `v1` | Does not merge other source oils. |
| `scallion`, `scallions`, `green onion`, `green onions` | `green onion` | Active in `v1` | `spring onion` is added by 2.2b below. |
| `cilantro leaf`, `cilantro leaves` | `cilantro` | Active in `v1` | Leaf wording is canonicalized. |
| `fresh coriander` | intended as `cilantro`, currently becomes `coriander` | Declared but ineffective in `v1`; superseded by 2.2b | The current processing order erases `fresh` before alias matching. |
| `confectioner sugar`, `confectioners sugar`, `powdered sugar`, `icing sugar` | `sugar` | Active in `v1` | Brown and granulated sugar are covered by 2.2b below. |
| `chicken breast(s)`, `chicken thigh(s)`, `chicken tenderloin(s)` | `chicken` | Active in `v1` | Cut distinctions are removed. |
| `ground beef`, `lean ground beef` | `beef` | Active in `v1` | Source meat is retained. |
| `ground pork`, `lean ground pork` | `pork` | Active in `v1` | Source meat is retained. |
| `egg`, `eggs`, `large egg(s)`, `extra large egg(s)` | `egg` | Active in `v1` | Does not merge explicit egg whites or yolks. |
| `black pepper`, `white pepper`, `ground pepper` | `pepper` | Active in `v1` | Red-pepper phrases follow the separate 2.2b rules below. |
| `chile powder`, `chili powder` | `chili powder` | Active in `v1`; superseded by 2.2b | The approved final output becomes `chili`. |
| any bounded normalized phrase containing `cheese` or `queso` | `cheese` | Active in `v1` | Includes grated, shredded, and crumbled forms. |
| `spaghetti`, `penne`, `fusilli`, `rigatoni`, `linguine`, `tagliatelle`, `fettuccine`, `macaroni`, `farfalle`, `lasagna` | `pasta` | Active in `v1` | Shape distinctions are removed. |

### Exact singularization

The `v1` standardizer applies the following mappings only when the whole normalized phrase equals the listed plural.

| Input | Output | Lifecycle |
| --- | --- | --- |
| `apples` | `apple` | Active in `v1` |
| `eggs` | `egg` | Active in `v1` |
| `onions` | `onion` | Active in `v1` |
| `tomatoes` | `tomato` | Active in `v1` |
| `potatoes` | `potato` | Active in `v1` |
| `carrots` | `carrot` | Active in `v1` |
| `strawberries` | `strawberry` | Active in `v1` |
| `blueberries` | `blueberry` | Active in `v1` |
| `raspberries` | `raspberry` | Active in `v1` |
| `mushrooms` | `mushroom` | Active in `v1` |
| `beans` | `bean` | Active in `v1` |
| `peppers` | `pepper` | Active in `v1` |
| `limes` | `lime` | Active in `v1` |
| `lemons` | `lemon` | Active in `v1` |

## Mappings active in `v4` from Work package 2.2b

The following tables define the final approved post-audit behavior. They supersede any conflicting intermediate mapping or label in the `v1` candidate.

### Conservative alias cleanup

| Input target or normalized source form | Final output target | Rationale or boundary |
| --- | --- | --- |
| `bay leaves` | `bay leaf` | Number variant. |
| `warm water`, `cold water` | `water` | Temperature is not an ingredient identity. |
| `basil leaves` | `basil` | Redundant leaf suffix for this target. |
| `mint leaves` | `mint` | Redundant leaf suffix for this target. |
| `thyme leaves` | `thyme` | Redundant leaf suffix for this target. |
| `flat leaf parsley` | `parsley` | Preparation/style distinction is not retained. |
| `ginger root` | `ginger` | Redundant physical-form name. |
| `garlic paste` | `garlic` | Paste is treated as a preparation form. `Garlic powder` remains separate. |
| `spring onion`, `spring onions` | `green onion` | Regional alias. |
| `purple onion` | `red onion` | Regional/colour alias. |
| `cayenne` | `cayenne pepper` | Canonical spelling. |
| `cider vinegar` | `apple cider vinegar` | Canonical spelling. |
| `coarse salt` | `salt` | Grain-size distinction is not retained. |
| `cooking oil` | `oil` | Generic alias only; named source oils remain separate. |
| `whole milk` | `milk` | Fat/style qualifier is not retained here. |
| `bananas` | `banana` | Number variant. |
| `almonds` | `almond` | Number variant. |
| `pecans` | `pecan` | Number variant. |
| `walnuts` | `walnut` | Number variant. |

### Composite and exclusion rules

| Input | Final output | Rule type | Boundary |
| --- | --- | --- | --- |
| `salt and pepper` and bounded variants containing only salt/pepper descriptors (for example `coarse salt and freshly ground black pepper`) | `salt` and `pepper` | Multi-target expansion | Emit both targets, then deduplicate at recipe level. |
| bare generic `sauce` | no target | Exclusion | Specific targets such as `soy sauce`, `fish sauce`, `hot sauce`, and `tomato sauce` remain eligible. |

### Broth, sauce, oil, dairy, and tomato families

| Input target or normalized source form | Final output target | Boundary |
| --- | --- | --- |
| `chicken stock`, `chicken broth`, `low sodium chicken broth` | `chicken broth` | Does not merge other stock or broth source families. |
| `soy sauce`, `light soy sauce`, `dark soy sauce`, `low sodium soy sauce` | `soy sauce` | Other named sauces remain separate. |
| `sesame oil`, `toasted sesame oil` | `sesame oil` | Olive, vegetable, peanut, coconut, and other source oils remain separate. |
| `yogurt`, `yoghurt`, plain/Greek forms, and non-visual fat-style qualifiers on those forms | `yogurt` | Style and spelling distinctions are removed. |
| `tomato paste`, `tomato sauce` | `tomato sauce` | Fresh `tomato` remains a separate target. |

### Spice physical forms

Only the explicitly listed spice families use the ground/powder-to-base policy. Seeds, leaves, and sticks remain distinct targets when they pass the support threshold.

| Input target or bounded form | Final output target | Preserved forms |
| --- | --- | --- |
| `ground cumin`, `cumin powder` | `cumin` | `cumin seed(s)` |
| `ground coriander`, `coriander powder`, bare `coriander`, fresh/leaf `coriander`, `cilantro` | `cilantro` | `coriander seed(s)` |
| `ground turmeric`, `turmeric powder` | `turmeric` | Explicit non-powder physical forms are not generalized automatically. |
| `ground cinnamon`, `cinnamon powder` | `cinnamon` | `cinnamon stick(s)` |
| `ground nutmeg`, `nutmeg powder` | `nutmeg` | Explicit non-powder physical forms are not generalized automatically. |
| `ground ginger`, `ginger powder` | `ginger` | Explicit non-powder physical forms are not generalized automatically. |

This rule does not implicitly merge `garlic powder` with `garlic` or `onion powder` with `onion`.

### Onion taxonomy

| Input target or normalized source form | Final output target | Boundary |
| --- | --- | --- |
| `white onion`, `yellow onion`, `spanish onion`, `sweet onion` | `onion` | These bulb-onion variants collapse into the generic target. |
| `purple onion`, `red onion` | `red onion` | Retained separately from generic bulb onion. |
| `scallion(s)`, `spring onion(s)`, `green onion(s)` | `green onion` | Retained separately from bulb onion. |

### Sugar taxonomy

| Input target or normalized source form | Final output target | Boundary |
| --- | --- | --- |
| `sugar`, `white sugar`, `granulated sugar`, `brown sugar`, `light brown sugar` | `sugar` | Colour and granulation distinctions are not retained. |
| `confectioner(s) sugar`, `powdered sugar`, `icing sugar` | `sugar` | Existing `v1` behavior remains. |

### Pepper and chili taxonomy

| Input target or bounded source form | Final output target | Boundary |
| --- | --- | --- |
| bare `red pepper` | `red bell pepper` | Interpreted as a fresh colour-specific bell pepper. |
| bare `green pepper` | `green bell pepper` | Interpreted as a fresh colour-specific bell pepper. |
| `chile/chili powder` (including explicit `red` or `hot` variants), `ground red pepper`, `crushed red pepper` (including dried/flaked forms), `red pepper flakes`, `chile/chili flakes` | `chili` | Powdered, ground, crushed, dried-crushed, and flaked generic red-pepper seasonings are one target. |

`Red bell pepper` and `green bell pepper` remain separate from `chili` and from each other. The mapping to `chili` applies to the explicit dried, powdered, ground, crushed, or flaked seasoning forms above; it must not absorb a fresh colour-specific bell pepper.

## Preserved collision boundaries

The following cases must never be produced by unbounded substring matching:

| Input | Required normalized identity | Forbidden target inference |
| --- | --- | --- |
| `pineapple` | `pineapple` | `apple` |
| `butternut squash` | `butternut squash` | `butter` |
| `pepperoni` | `pepperoni` | `pepper` |
| `watercress` | `watercress` | `water` |

The same principle applies generally: a target token occurring inside an unrelated word is not evidence for that target.

## Deliberately retained distinctions

The recognizability policy is not permission for unrestricted taxonomy collapse. Unless a later documented decision changes them, preserve these boundaries:

- fresh `tomato` versus processed `tomato sauce`;
- `red bell pepper` versus `green bell pepper` versus dried-seasoning `chili`;
- coriander seeds versus canonical `cilantro`;
- cumin seeds and cinnamon sticks versus their corresponding base spice;
- `red onion`, `green onion`, and generic bulb `onion`;
- named source oils such as olive oil and sesame oil;
- specific sauces rather than the excluded generic `sauce` target;
- garlic and onion powders, which are outside the approved six-family spice collapse;
- visually meaningful subtypes such as baby spinach, romaine lettuce, and French bread until the later ingredient-selection phase makes a different explicit decision.

## Maintenance and verification

Every mapping change must include:

1. an explicit source pattern and final output in this document;
2. an explanation of important exclusions or neighboring labels that must remain distinct;
3. a token- or phrase-bound implementation rule;
4. positive and collision-boundary regression tests;
5. regeneration of a new metadata version rather than modification of an existing generation;
6. comparison of vocabulary size, support, retained recipes, and affected records with the preceding candidate.

Work package 2.2b was implemented in `ingredients_target_v4_metadata.json` on 2026-08-04. The preceding `_ingredients_target_v2_metadata.json` and `_ingredients_target_v3_metadata.json` files are retained as non-selected diagnostic generations: their automatic data checks passed, but review of their comparison output exposed out-of-scope normalizations that were corrected before `v4` was generated. Their leading underscore prevents accidental selection by convention. `v4` must not be selected for new consumers; it remains a reproducible baseline until Work packages 2.2c and 2.2d define and validate its replacement.

## Related documents and code

- [`../../src/data_processing/ingredient_standardization.py`](../../src/data_processing/ingredient_standardization.py) contains the current `v1` implementation.
- [`../project_objective/ingredient_vocabulary_audit.md`](../project_objective/ingredient_vocabulary_audit.md) contains the quantitative evidence and decision history.
- [`../project_objective/benchmark_decisions.md`](../project_objective/benchmark_decisions.md) defines the binding benchmark-level policies.
- [`../plans/yummly_data_phase.md`](../plans/yummly_data_phase.md) tracks implementation progress.
- [`../../src_scratches/data_anlysis/ingredient_vocabulary_audit.py`](../../src_scratches/data_anlysis/ingredient_vocabulary_audit.py) reproduces the aggregate audit and counterfactual packages.
