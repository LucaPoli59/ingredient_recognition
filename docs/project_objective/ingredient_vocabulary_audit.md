# Candidate ingredient vocabulary audit

**Created:** 2026-08-04  
**Last updated:** 2026-08-04

## Purpose and boundary

This document records Work package 2.2a: an evidence-based review of the 209-target `ingredients_target_v1_metadata.json` candidate. Its purpose is to identify avoidable vocabulary fragmentation, semantic collisions, and unsupported distinctions before the extractor is strengthened and the benchmark is regenerated.

The audit is diagnostic only. It does not change [`ingredient_standardization.py`](../../src/data_processing/ingredient_standardization.py), any metadata generation, any split, or any legacy artifact. The classifications and counterfactual packages below are proposals for discussion, not approved preprocessing rules.

## Evidence and method

The reproducible analysis is implemented in [`ingredient_vocabulary_audit.py`](../../src_scratches/data_anlysis/ingredient_vocabulary_audit.py). It reads the three candidate metadata files and the immutable source `metadata.json` files, then produces aggregate evidence under `src_scratches/data_anlysis/outputs/ingredient_vocabulary_audit/`:

- `audit.json` contains the structured summary, source traceability, collision checks, ranked pair views, and counterfactual review packages;
- `target_support.csv` contains per-target support before and after record retention and support by split;
- `pair_relationships.csv` contains non-zero co-occurrence, conditional probabilities, Jaccard similarity, and lift;
- `lexical_relationships.csv` contains token-bounded phrase containment, singular/plural, and conservative near-spelling signals.

The audit scanned 60,550 retained recipes and 707,771 raw ingredient lines. Every one of the 475,976 candidate target assignments was traced to at least one raw line that produces that target under the current standardizer. No raw-line-to-target mapping is persisted.

Two consecutive executions on identical inputs produced byte-identical `audit.json` and CSV outputs. This verifies deterministic ordering as well as identical statistics.

## Candidate baseline

| Measure | Result |
| --- | ---: |
| Source recipes | 65,146 |
| Retained candidate recipes | 60,550 |
| Recipes removed by the minimum-three-target rule | 4,596 |
| Train / validation / test records | 48,439 / 6,056 / 6,055 |
| Vocabulary size | 209 |
| Target assignments | 475,976 |
| Targets per recipe, mean / median | 7.86 / 7 |
| Targets per recipe, 10th / 90th percentile | 4 / 13 |
| Minimum / maximum cardinality | 3 / 34 |
| Median target support after retention | 1,109 recipes |
| Targets below 1,000 retained recipes | 91 |
| Targets with at least 5,000 retained recipes | 18 |

All 209 targets occur in train, validation, and test. The lowest source support among retained vocabulary entries is 502 recipes, which confirms that the configured support threshold of 500 was applied correctly before record retention. Five targets fall slightly below 500 after recipes with fewer than three supported targets are removed: `blueberry` (469), `orange` (476), `walnuts` (495), `black peppercorns` (496), and `peanut butter` (499). This is an expected ordering effect, not a threshold failure.

The support threshold is therefore not the main cause of the excessive vocabulary size. The stronger reduction opportunity is inconsistent normalization of semantically equivalent forms.

## Pair relationships do not justify automatic merging

The strongest common pair is `pepper`–`salt`: 17,251 co-occurrences, Jaccard 0.438, and `P(salt | pepper) = 0.779`. The most cuisine-specific high-similarity pair is `mirin`–`sake`: 398 co-occurrences, Jaccard 0.310, and lift 35.61.

These examples demonstrate why co-occurrence must remain diagnostic evidence only. `Mirin` and `sake` are strongly associated but remain different ingredients; the same applies to pairs such as `sesame oil`–`soy sauce`, `baking powder`–`baking soda`, and `garlic powder`–`onion powder`. No candidate merge below is based solely on a relationship score.

## High-confidence normalization findings

The following findings represent the same ingredient under spelling, number, preparation, temperature, regional-alias, or non-semantic product qualifiers. Low co-occurrence between the alternatives is expected because recipes normally use one spelling or form, and strengthens the fragmentation diagnosis when paired with the raw-line evidence.

| Current targets | Retained support | Provisional classification | Proposed canonical behavior |
| --- | ---: | --- | --- |
| `bay leaf`, `bay leaves` | 1,227 + 1,181; only 6 overlap | Candidate merge | `bay leaf` |
| `water`, `warm water`, `cold water` | 8,805 + 681 + 513 | Candidate merge | `water` |
| `basil` / `basil leaves`; `mint` / `mint leaves`; `thyme` / `thyme leaves` | 2,827/1,356; 1,236/550; 2,373/527 | Candidate merge | Remove the redundant `leaves` form |
| `parsley`, `flat leaf parsley` | 3,492 + 1,635 | Candidate merge | `parsley` |
| `ginger`, `ginger root` | 7,018 + 608 | Candidate merge | `ginger` |
| `garlic`, `garlic paste` | 24,801 + 543 | Candidate merge | Treat paste as a preparation form of `garlic` |
| `green onion`, `spring onions` | 8,103 + 752 | Candidate merge | `green onion` |
| `purple onion`, `red onion` | 2,637 + 1,162 | Candidate merge | `red onion` |
| `cayenne`, `cayenne pepper` | 598 + 1,890 | Candidate merge | `cayenne pepper` |
| `apple cider vinegar`, `cider vinegar` | 578 + 537; no overlap | Candidate merge | `apple cider vinegar` |
| `coarse salt`, `salt` | 1,126 + 34,455 | Candidate merge | `salt` |
| `cooking oil`, `oil` | 516 + 3,310 | Candidate merge | `oil` |
| `whole milk`, `milk` | 1,012 + 3,433 | Candidate merge | `milk` |
| `low sodium chicken broth`, `chicken broth` | 654 + 1,942 | Candidate merge | `chicken broth` |
| `low sodium soy sauce`, `soy sauce` | 730 + 5,367 | Candidate merge | `soy sauce` |
| `granulated sugar`, `white sugar`, `sugar` | 1,732 + 894 + 9,681 | Candidate merge | `sugar` |
| `light brown sugar`, `brown sugar` | 751 + 2,788 | Candidate merge | `brown sugar` |
| `plain greek yogurt`, `greek yogurt` | 1,166 + 1,697 | Candidate merge | `greek yogurt` |
| `yoghurt`, `plain yogurt` | 506 + 560 | Candidate merge | Canonical new label `yogurt` |

The singular/plural and alias rules must also standardize the canonical spelling of labels such as `bananas`, `almonds`, `pecans`, and `walnuts`. These are candidate rule replacements but do not reduce vocabulary size because no corresponding singular target currently coexists.

## Composite and semantically mixed targets

### `salt and pepper`

`salt and pepper` is a composite target in 1,470 recipes. It overlaps `salt` in only 170 recipes and `pepper` in only 84, so it withholds one or both elemental labels from most affected recipes. This is not an alias merge. It is a high-confidence candidate rule replacement: one raw line should be allowed to emit both `salt` and `pepper`, and the composite target should disappear.

This change reduces the vocabulary by one but can increase target cardinality because a single composite assignment becomes up to two elemental assignments.

### `coriander`

The current `coriander` target has support 1,151 and mixes regional and physical meanings. Among its producing lines, 419 recipes contain an explicit fresh/leaf form while 741 contain a generic form; a recipe can contribute to both counts. The intended alias for `fresh coriander` is currently ineffective because the word `fresh` is removed before alias matching, leaving `coriander`.

The approved rule retains this ingredient family under canonical `cilantro`: bare `coriander`, `cilantro`, fresh/leaf forms, and ground/powder forms collapse into that target through the base-spice rule. `Coriander seeds` remains separate because seeds are a distinct physical form.

### `red pepper`

The `red pepper` target has support 1,751 but combines at least two meanings: 831 recipes contain explicit ground/crushed/flaked wording, while 929 contain generic wording; a recipe can contribute to both counts. Bare `red pepper` maps to fresh `red bell pepper`. Generic powder, ground, crushed, dried-crushed, and flaked red-pepper forms collapse with `chili powder` into `chili` because the approved policy does not retain powder versus flakes as separate outputs.

`green pepper` (546 recipes) maps to `green bell pepper`. Red and green fresh bell peppers remain separate because their colour is a robust visual distinction.

### Generic `sauce`

The `sauce` target occurs in 558 recipes. Its source evidence is overwhelmingly the bare string `sauce` or section-header variants such as `Sauce:`; it does not identify a stable ingredient. This is a high-confidence candidate exclusion, not a merge with any specific sauce.

## Granularity choices that require an explicit policy

The following groups contain real culinary distinctions. Collapsing them may improve sample support and reduce output size, but the decision cannot be inferred from lexical containment or co-occurrence.

| Group | Union support / current assignments | Provisional classification | Decision required |
| --- | ---: | --- | --- |
| Ground/base spices: cumin, coriander, turmeric, cinnamon, nutmeg, and ginger forms | 17,143 / 24,703 | Approved merge | Ground and powder forms collapse into the base spice; seeds, leaves, and sticks remain distinct |
| `chicken broth`, low-sodium broth, `chicken stock` | 4,293 / 4,301 | Approved merge | Collapse into `chicken broth` because the variants are not reliably separable from a prepared-dish image |
| Base/light/low-sodium/dark soy sauce | 7,056 / 7,503 | Approved merge | Collapse into `soy sauce`; the style differences are too small once used in a cooked product |
| Base/toasted sesame oil | 3,568 / 3,588 | Approved merge | Collapse into `sesame oil` |
| Plain, Greek, and spelling variants of yogurt | 3,874 / 3,929 | Approved merge | Collapse into `yogurt` |
| Bulb-onion colour/type labels | 20,379 / 20,748 | Approved partial merge | Retain red and green onion; collapse white, yellow, Spanish, and sweet onion into `onion` |
| Generic/specific vegetables and staples such as baby spinach, romaine lettuce, and French bread | Depends on selected group | Retain | Keep visually meaningful subtypes unless a later ingredient-selection decision excludes them |

The sesame-oil row contains 2,978 `sesame oil` and 610 `toasted sesame oil` assignments with 20 overlapping recipes, yielding 3,568 unique recipes. These choices belong to target-taxonomy policy, not fuzzy normalization.

## Known collision regression

The token-bounded standardizer still passes all four confirmed legacy substring-collision checks:

| Raw line | Current normalized output | Forbidden legacy collision | Lines observed in retained recipes |
| --- | --- | --- | ---: |
| `pineapple` | `pineapple` | `apple` | 368 |
| `butternut squash` | `butternut squash` | `butter` | 373 |
| `pepperoni` | `pepperoni` | `pepper` | 115 |
| `watercress` | `watercress` | `water` | 172 |

None of these normalized outputs reaches the current vocabulary threshold, but preserving their correct boundaries remains necessary because future data or thresholds could make them targets.

## Counterfactual review packages

The analysis simulates proposed transformations on the candidate target lists without changing the extractor or metadata. These counts are decision aids, not predicted final dataset sizes; the full pipeline must recompute support, recipe retention, SHA groups, and splits after accepted rules are implemented.

| Provisional package | Vocabulary | Affected candidate recipes | Target assignments | Candidate recipes falling below three targets |
| --- | ---: | ---: | ---: | ---: |
| Current candidate | 209 | — | 475,976 | 0 |
| Conservative normalization and `salt and pepper` expansion | 185 | 19,458 | 475,968 | 42 |
| Conservative package plus generic `sauce` exclusion | 184 | 19,826 | 475,410 | 54 |
| Plus ground/form granularity collapse | 175 | 27,439 | 475,571 | 42 |
| Plus broader subtype/taxonomy collapse | 168 | 31,414 | 475,363 | 42 |

The conservative package reduces the vocabulary by 24 labels without materially reducing total assignments because the `salt and pepper` expansion offsets assignments removed by deduplicated aliases. The 42 recipes below the current minimum illustrate why the extractor must be rerun before retaining records; editing the existing metadata in place would be incorrect.

## Decision update: recognizability-led granularity

The 2.2b decision criterion is primarily whether two ingredient labels can be distinguished reliably from the final food image, especially after cooking. Fine product, preparation, or style differences should collapse when they do not create a realistically learnable visual distinction. Variants with a robust visual difference, such as green and red bell peppers, may remain separate.

This criterion applies within a meaningful ingredient family. It does not imply collapsing every liquid, oil, or seasoning into one generic target: different source ingredients such as olive oil and sesame oil may remain different, while an intra-family qualifier such as toasted sesame oil is removed.

The following changes are approved for the future 2.2b implementation:

1. apply the conservative normalization package;
2. exclude the generic `sauce` target;
3. replace `salt and pepper` with the two elemental targets;
4. merge and retain explicit fresh/leaf coriander under `cilantro`;
5. collapse `chicken stock`, `chicken broth`, and low-sodium chicken broth into `chicken broth`;
6. collapse base, light, dark, and low-sodium soy-sauce variants into `soy sauce`;
7. collapse all current plain and Greek yogurt variants and spellings into `yogurt`;
8. collapse `toasted sesame oil` into `sesame oil`;
9. collapse `tomato paste` and `tomato sauce` into one `tomato sauce` target while retaining fresh `tomato` separately;
10. retain red and green bell-pepper targets separately;
11. keep the 500-recipe source-support threshold and minimum-three-target retention rule for the first comparison run.
12. merge ground and powder spice forms into their base spice while retaining seeds, leaves, and sticks as separate physical forms;
13. retain `red onion` and `green onion`, while collapsing white, yellow, Spanish, and sweet onion into `onion`;
14. collapse brown, white, granulated, and light-brown sugar into `sugar`;
15. merge and retain bare, fresh, and leaf `coriander`/`cilantro` under canonical `cilantro`, while explicit ground/powder forms follow the spice rule;
16. map bare `red pepper` and `green pepper` to the corresponding colour-specific bell-pepper target.
17. collapse `chili powder`, `ground red pepper`, `crushed red pepper`, `dried crushed red pepper`, `red pepper flakes`, and equivalent generic dried/powdered forms into `chili`.

The decision gate is complete. Powder and flakes are not retained as separate chili outputs, while fresh red and green bell peppers remain separate from `chili` and from each other.

The durable source-to-target contract, including every approved mapping and preserved boundary, is maintained in [`../implementation_details/ingredient_mapping_rules.md`](../implementation_details/ingredient_mapping_rules.md). This audit remains the evidence and decision-history record rather than a second mapping authority.

After final approval, Work package 2.2b must add a regression test for every accepted behavior, regenerate a new metadata version rather than overwrite `v1`, and compare vocabulary size, recipe retention, target support, and affected records against this audit.

## Limitations

- Co-occurrence measures recipe metadata association, not visual similarity, interchangeability, or causation.
- Lexical containment generates review candidates but cannot distinguish aliases from legitimate subtypes.
- The counterfactual packages operate on already retained candidate records; only a full rebuild can determine the final support and retained record counts.
- Work package 2.2b uses practical recognizability to avoid unsupported fine-grained target distinctions. The later Ingredient selection macro-section in [`../general_plan.md`](../general_plan.md) still performs the formal per-label relevance and visual-distinguishability selection used for thesis evaluation.

## Related documents

- [`yummly_data_audit.md`](yummly_data_audit.md) documents the source data, legacy targets, leakage, and observability limitations.
- [`benchmark_decisions.md`](benchmark_decisions.md) defines the binding benchmark policies.
- [`../implementation_details/ingredient_mapping_rules.md`](../implementation_details/ingredient_mapping_rules.md) is the durable mapping registry.
- [`../plans/yummly_data_phase.md`](../plans/yummly_data_phase.md) defines Work packages 2.2a and 2.2b and their decision gate.
