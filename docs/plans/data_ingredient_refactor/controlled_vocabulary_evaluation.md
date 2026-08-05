# Yummly controlled-vocabulary evaluation

**Created:** 2026-08-04  
**Last updated:** 2026-08-05

**Work package:** Data 2.2c  
**Status:** Research complete; implementation recommendation awaiting approval

## Objective

This study tests whether an external controlled vocabulary can improve Yummly ingredient canonicalization without an online service, fuzzy matching, a manual review workflow, or per-line mapping artifacts. It evaluates the candidates described in the reusable [`vocabulary_catalog.md`](../../research/topics/ingredient_vocabularies/vocabulary_catalog.md) and defines the decision gate before Work package 2.2d changes production code or generates new metadata.

The study does not modify `ingredients_target`, the existing `v1`–`v4` metadata, the split, the DataModule, or historical experiments.

## Materials and reproducibility

The quantitative experiment used:

- the original `metadata.json` records: 54,724 train, 5,210 validation, and 5,212 test recipes;
- 622,564 ingredient-line occurrences and 43,751 distinct mechanically parsed source terms in the legacy train split;
- the pinned [FoodOn v2025-07-31 release](https://github.com/FoodOntology/foodon/releases/tag/v2025-07-31), restricted to descendants of `FOODON_00001002` (`food product`);
- the official LanguaL 2017 XML export, evaluated both as product-type facet A and, diagnostically, across all active facets;
- the USDA Foundation Foods April 2026 JSON release as a database-shaped comparison rather than an ontology candidate.

[`../../../src_scratches/data_anlysis/controlled_vocabulary_research.py`](../../../src_scratches/data_anlysis/controlled_vocabulary_research.py) reproduces the study. It writes only aggregate results and a fixed edge-case report under [`../../../src_scratches/data_anlysis/outputs/controlled_vocabulary_research/`](../../../src_scratches/data_anlysis/outputs/controlled_vocabulary_research/); it does not persist one mapping per source line.

## Association protocol tested

The experiment uses a deliberately conservative lexical protocol:

1. mechanically parse quantities, units, punctuation, parentheticals, and trailing recipe instructions while preserving semantically meaningful qualifiers;
2. match an exact preferred label first;
3. if no unique preferred-label match exists, accept an exact synonym or a suffix-safe FoodOn label variant only when it resolves to one concept;
4. for an unmatched line, apply the existing bounded fallback standardization and retry;
5. permit final-token singularization only when the singular form resolves to exactly one vocabulary concept;
6. retain ambiguous or unmatched outputs as local concepts rather than deleting them or using `<UNK>`.

There is no edit distance, embedding similarity, substring containment, model prediction, or automatic hierarchy ascent.

## Candidate coverage on train

Percentages below use the 622,564 train ingredient-line occurrences. `Local` means that the term remains usable as a project-owned standalone concept; it is not lost.

These values measure deterministic lexical association coverage, not accepted semantic accuracy. A unique ontology label can still be too fine-grained or have the wrong task-specific sense; the edge-case analysis and proposed recognizability precedence address that separate question.

| Candidate index | Direct unique association | Association after fallback | Ambiguous | Local unmatched | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| FoodOn food-product branch | 30.11% | 21.05% | 4.66% | 44.17% | Best external anchor; useful but necessarily hybrid |
| LanguaL all active facets | 18.65% | 20.49% | 0.94% | 59.93% | Higher coverage than facet A, but semantically unsafe because non-identity facets are mixed in |
| LanguaL product-type facet A | 3.79% | 3.34% | 0.62% | 92.25% | Too sparse as a standalone ingredient vocabulary |
| FoodData Central Foundation Foods | 0.00% | 0.00% | 0.00% | 100.00% | Confirms that food-record descriptions are not an ingredient lexicon |

The FoodOn index contains 14,185 labeled concepts in the selected descendant branch. It yielded 1,468 external concepts on train before support filtering, while 34,399 distinct local concepts remained. The large local tail is expected: it includes inflected phrases, recipe-specific products, brand-like descriptions, composite instructions, and rare culturally specific ingredients.

## Important edge cases

| Source form | Conservative result | Train recipe support | Consequence |
| --- | --- | ---: | --- |
| `English muffins` | `FOODON_03305833`, `english muffin`, after vocabulary-validated singularization | 35 | Correctly recovered by FoodOn, but removed by any support threshold above 35 |
| `fish fillets` | `FOODON_00002679`, `fish fillet`, after vocabulary-validated singularization | 50 | Correctly recovered, but removed by the current threshold of 500 |
| `chicken broth` | `FOODON_03303004`, direct | 3,605 | Stable direct product concept |
| `tomato sauce` | `FOODON_03301217`, direct | 1,002 after direct concept association | Stable direct concept, but not automatically merged with tomato paste |
| `garam masala` | Local concept | 1,811 | High-support local concepts can be valuable even when FoodOn has no safe selected-branch match |
| `mirin` | Local concept | 857 | Demonstrates why unmatched concepts must not be discarded |
| `gochujang` | Local concept | 2 | Valid but extremely rare in this corpus |

The motivating recipe is train record `10002`. Its original list contains both `fish fillets` and `English muffins`; the `v4` frequency-first pipeline loses both. FoodOn can identify both after a bounded singularization, proving that they are not inherently unstandardizable. Their supports of 50 and 35 also prove that vocabulary association alone cannot save them while the threshold remains 500.

## Support-threshold sensitivity

This analysis derives support only from the legacy train split. `Assignments retained` counts unique recipe-concept assignments, not raw lines. The results are diagnostic and do not select a final model vocabulary.

| Minimum train recipe support | Concepts retained | Recipes with at least three retained concepts | Assignments retained |
| ---: | ---: | ---: | ---: |
| 1 | 35,867 | 54,724 | 100.00% |
| 10 | 2,850 | 54,129 | 91.24% |
| 25 | 1,537 | 53,783 | 87.95% |
| 50 | 962 | 53,452 | 84.66% |
| 100 | 590 | 52,891 | 80.36% |
| 250 | 290 | 51,712 | 72.61% |
| 500 | 165 | 50,427 | 65.32% |

The current threshold retains only 65.32% of canonical recipe-concept assignments and removes 4,297 of 54,724 train recipes under the minimum-three-target rule. A threshold of 25 is the highest tested value that retains both motivating concepts, but it still discards 12.05% of assignments and creates a provisional 1,537-class space. This is evidence against using support filtering as part of semantic standardization.

### Executed provisional current-normalizer sweep

The requested decision-oriented sweep was executed on 2026-08-05 by [`ingredient_threshold_sweep.py`](../../../src_scratches/data_anlysis/ingredient_threshold_sweep.py). It reads only the original train `ingredients` values, applies the current deterministic normalizer, and writes [`ingredient_threshold_sweep.json`](../../../src_scratches/data_anlysis/outputs/ingredient_threshold_sweep.json). It does not alter metadata or make an ontology association, so its counts are provisional evidence for selecting a support policy; the selected policy must be revalidated after the FoodOn-plus-local association step before a new generation is frozen.

| Minimum train support | Provisional targets | Retained assignments | Recipes with 0 / 1 / 2 / at least 3 targets |
| ---: | ---: | ---: | ---: |
| 100 | 562 | 79.97% | 190 / 565 / 1,150 / 52,819 |
| 200 | 318 | 74.31% | 410 / 814 / 1,571 / 51,929 |
| 250 | 260 | 72.18% | 435 / 885 / 1,789 / 51,615 |
| 300 | 227 | 70.67% | 468 / 942 / 1,890 / 51,424 |
| 400 | 168 | 67.30% | 532 / 1,119 / 2,251 / 50,822 |
| 500 | 143 | 65.44% | 582 / 1,263 / 2,455 / 50,424 |

The complete output records every target removed between adjacent thresholds, rather than a raw-line mapping. For example, increasing 400 to 500 removes 25 targets including `romaine lettuce` (499), `quinoa` (497), `black olives` (491), `ghee` (485), `mango` (475), `eggplant` (468), and `broccoli` (441) recipe occurrences. The selected standard policy is support >= 500 train recipes and at least three retained targets per recipe. Rare labels are therefore not retained merely because they are semantically valid; the exact resulting counts will be revalidated after association.

## Findings

### FoodOn is useful as a reference, not as the complete target vocabulary

FoodOn is the preferred external resource because it has versioned identifiers, food-product coverage, typed synonyms, and a downloadable offline representation. However, 44.17% of train occurrences remain local and a further 4.66% are lexically ambiguous. Forcing those terms into a FoodOn parent or a fuzzy nearest concept would replace transparent incompleteness with silent semantic errors.

### The ontology hierarchy cannot choose visual granularity automatically

FoodOn directly distinguishes `tomato paste` from `tomato sauce` and Greek from generic yogurt. Those distinctions are semantically legitimate but conflict with the project's approved prepared-image recognizability merges. Conversely, a generic surface such as `pepper` can refer to multiple product families. A fixed parent depth or first-parent rule would therefore be inconsistent and order-dependent.

Project-approved recognizability overrides must take precedence over direct ontology matching. Otherwise a successful fine-grained FoodOn match would bypass the accepted merge and reintroduce distinctions already rejected for this task.

Automatic parent traversal is excluded from the standard pipeline, but explicit parent-based abstraction is retained as a deferred experimental option. A future experiment may deliberately map selected fine-grained concepts to named parents to reduce label-space difficulty, provided that every mapping is reviewed, versioned, reproducible, and evaluated against the unchanged standard `ingredients_target` vocabulary. It must never be enabled implicitly by ontology depth or replace the default targets.

### The support threshold is the principal cause of the observed information loss

Moving the threshold after concept association fixes premature fragmentation, but it does not fix rare-concept deletion. The `English muffin` and `fish fillet` examples remain below 500 after correct association. The threshold must therefore be selected from an explicit sensitivity analysis before producing the shared `ingredients_target` vocabulary.

### Unmatched local concepts are necessary

High-support examples such as `garam masala` and `mirin` remain valuable local concepts under the conservative FoodOn branch. An external identifier is helpful metadata, not a condition for an ingredient to exist. `<UNK>` would collapse unrelated meanings and is inappropriate for these multi-label targets.

## Recommended implementation contract for 2.2d

1. Pin FoodOn v2025-07-31 and use only its `food product` descendant branch as the external reference lexicon.
2. Keep output metadata simple: `ingredients_target` remains a list of canonical strings. FoodOn IDs are implementation-time association anchors, not additional per-record fields.
3. Apply the explicit recognizability rules in [`../../implementation_details/ingredient_mapping_rules.md`](../../implementation_details/ingredient_mapping_rules.md) before direct ontology matching whenever a rule intentionally collapses a distinction that FoodOn preserves.
4. For other lines, use exact preferred-label matching, then exact synonyms, then the approved bounded fallback and vocabulary-validated final-token singularization.
5. Accept a vocabulary association only when deterministic precedence produces one concept. Keep an unresolved or ambiguous normalized term as a local concept.
6. Do not use unspecified broad/narrow synonyms, fuzzy matching, embeddings, substring containment, cuisine, recipe title, images, or automatic parent traversal to force an association.
7. Before selecting a support threshold or minimum-target recipe rule, run a reproducible train-only threshold sweep. The provisional current-normalizer sweep now reports retained ingredients, assignments, zero/one/two/at-least-three target recipe buckets, and named targets lost at every adjacent threshold transition. Re-run the same measurement after controlled-vocabulary association to validate the selected threshold numerically.
8. Use that evidence to select one standard `ingredients_target` vocabulary. The selected policy is support >= 500 distinct train recipes and at least three retained targets per recipe. The standard pipeline must not maintain separate semantic and learnable vocabularies.
9. Validation and test data must not influence threshold selection. After a train-only choice is made, the same `ingredients_target` vocabulary is written and used consistently across train, validation, test, and model implementations.
10. Optional ingredient subsets may be introduced later only as explicitly named experimental projections. They must preserve the standard `ingredients_target` generation as the comparison baseline and must not silently redefine the project vocabulary.
11. Preserve `v1`–`v4` unchanged and generate a new version only after this contract is approved and implemented with regression tests.

## Decision gate

Work package 2.2d must not start until the following recommendations are accepted or revised:

- FoodOn is adopted as a pinned reference lexicon, with local concepts retained for incomplete or ambiguous coverage;
- project recognizability overrides run before a conflicting direct FoodOn match;
- automatic ontology parent traversal is rejected from the standard pipeline, while explicit reviewed parent mappings are retained as a deferred option for separately versioned difficulty-reduction experiments;
- the support threshold and minimum-target rule are fixed at >= 500 distinct train recipes and >= 3 retained targets per recipe, selected from train-only evidence and subject to numerical revalidation after association;
- after that revalidation, one shared `ingredients_target` vocabulary is used across all splits and model implementations; optional subsets remain separately named experiments.
