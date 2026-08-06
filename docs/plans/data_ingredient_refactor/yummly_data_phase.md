# Yummly data-phase implementation plan

**Created:** 2026-08-02  
**Last updated:** 2026-08-06

This plan translates the Data macro-section of [`general_plan.md`](../../general_plan.md) into a deliberately small implementation sequence. It covers the shared-image-store prerequisite, compatibility with historical experiments, generation of `ingredients_target`, deterministic split construction, and runtime integration.

The plan avoids persistent intermediate artifacts that are not consumed by the project. The benchmark outputs remain the common image directory and one selected metadata file in each split.

## Progress tracker

**Overall status:** In progress  
**Current task:** Validate the implemented runtime integration in the repository training environment.
**Next action:** Repair or activate a compatible Torch/NumPy/Lightning environment, then run the training, checkpoint-reload, and dashboard smoke tests required to close Work package 2.4.

| # | Task | Status | Evidence or result |
| --- | --- | --- | --- |
| P0 | Inspect the current data layout, loaders, legacy preprocessing, and historical experiment artifacts | **Done** | [Verified implementation findings](#verified-implementation-findings) |
| P1 | Agree on the simplified benchmark scope | **Done** | [Accepted design](#accepted-design) |
| P2 | Implement and verify the shared-image-store prerequisite | **Done** | `scripts/migrate_yummly_images.py` staged and SHA-256-verified 65,146 files in `imgs/standard`; both legacy generations load through the refactored DataModule. |
| P3 | Implement and verify historical experiment compatibility without rewriting saved artifacts | **Deferred** | Current-style legacy configurations retain `ingredients_ok` and receive `images_subdir` in memory. Complete validation is deferred until the historical experiments worth retaining are selected. |
| P4 | Design and implement the improved `ingredients` to `ingredients_target` standardizer | **Done** | `src/data_processing/ingredient_standardization.py` uses explicit token-bounded rules, recipe support >= 500, and at least three retained targets. |
| P4a | Audit the candidate ingredient vocabulary and present findings | **Done** | [`../../project_objective/ingredient_vocabulary_audit.md`](../../project_objective/ingredient_vocabulary_audit.md) audits 209 targets, 60,550 recipes, 707,771 raw lines, relationships, collisions, and counterfactual review packages without changing data or code. |
| P4b | Strengthen the ingredient extractor from accepted audit findings | **Done** | All approved token-bounded rules and collision boundaries are covered by tests; `ingredients_target_v4_metadata.json` is the reproducible 161-target candidate. |
| P4c | Research and select a controlled ingredient vocabulary | **Done** | [`controlled_vocabulary_evaluation.md`](controlled_vocabulary_evaluation.md) selects pinned FoodOn as the primary association lexicon, retained local concepts, and no automatic hierarchy traversal. Exact association runs before local fallback standardization; bounded fuzzy recovery is rejected after empirical evaluation. The selected standard filtering policy is train support >= 500 and at least three retained targets per recipe. |
| P4d | Implement the controlled-vocabulary target-generation pipeline | **Done** | `scripts/build_yummly_foodon_metadata.py` generated `ingredients_target_v5_metadata.json` with the pinned offline FoodOn index, exact-plus-fallback association, train-only support >= 500, and >= 3 targets per retained recipe. |
| P5 | Implement deterministic exact-duplicate-aware splitting and metadata generation | **Done** | `v4` remains the validated baseline; the same validator also passed the FoodOn-first `v5` generation as 47,965/5,996/5,996 records with no exact-image leakage. |
| P6 | Integrate the new target default and remove `<UNK>` from new multi-label outputs | **In progress** | Code and regression tests are complete: new configurations use `v5`, `ingredients_target`, and a strict 165-class encoder without `<UNK>`; legacy robust encoders retain `<UNK>`. Full runtime smoke tests are blocked locally by missing Lightning and a Torch/NumPy ABI mismatch. |
| P7 | Run all data checks and freeze the first new metadata generation | **Done** | The `v5` apply run passed image decoding, SHA-256, uniqueness, ratio, distribution, vocabulary, and deterministic builder assertions. Runtime DataModule smoke testing remains part of in-progress 2.4 because the available environment lacks Lightning and has an incompatible Torch/NumPy ABI. |

## Accepted design

The implementation follows these decisions:

- keep `feature_label` configurable and change only its default from `ingredients_ok` to `ingredients_target`;
- derive new `ingredients_target` values from the original `ingredients` field through an improved deterministic preprocessing script;
- retain the practical-recognizability mappings as the tested `v4` baseline and as bounded fallback standardization only; FoodOn association has priority in the replacement pipeline, while visual-distinguishability selection is deferred to Macro-section 3;
- retain `ingredients_ok` only as the legacy target used by historical data and experiments;
- do not save one mapping row per raw ingredient line;
- do not introduce manual image-review workflows;
- accept naturally noisy or imperfect images when they pass basic automatic file checks;
- do not create perceptual, name-based, ingredient-based, or manually reviewed recipe families;
- use exact file SHA-256 only to prevent byte-identical images from crossing split boundaries;
- balance the new split by cuisine and ingredient-target distribution;
- derive split membership from the location of the selected metadata file;
- derive the runtime vocabulary deterministically from training metadata and keep it in experiment/checkpoint configuration;
- perform automatic validations without requiring a persistent `validation_report.json`;
- investigate the original purpose and actual usefulness of `<UNK>` before deciding whether it remains a model output.

## Objective

Produce a simple data pipeline in which:

1. all Yummly images are stored once below `data/input/yummly/imgs/standard/`;
2. `train`, `val`, and `test` contain only metadata generations;
3. every new ingredient-recognition metadata generation uses `ingredients_target`;
4. old experiments remain loadable against unchanged legacy metadata and preserve exactly the label meaning and class order on which they were trained;
5. new targets are produced deterministically from `ingredients`;
6. exact duplicate images cannot leak across splits;
7. cuisine and ingredient distributions remain acceptably balanced;
8. the DataModule continues to support alternative target fields through `feature_label`.

## Non-goals

- No manual relabeling or image-quality review pipeline.
- No persistent raw-line mapping trace.
- No runtime ontology service, separate dataset-level vocabulary file, or per-line mapping artifact. A reference vocabulary may be evaluated and embedded deterministically in the target-generation code.
- No fuzzy or perceptual duplicate grouping.
- No standalone split manifest: the three metadata files are the split definition.
- No persistent machine-readable validation report.
- No rewriting or reinterpretation of old checkpoints using newly standardized targets.
- No decision about removing `<UNK>` until its historical and current use has been investigated.

## Target layout and metadata contract

```text
data/input/yummly/
├── imgs/
│   └── standard/
│       └── <image files>
├── train/
│   ├── metadata.json
│   ├── sel_ing_2410_metadata.json
│   └── <new_generation>.json
├── val/
│   ├── metadata.json
│   ├── sel_ing_2410_metadata.json
│   └── <new_generation>.json
└── test/
    ├── metadata.json
    ├── sel_ing_2410_metadata.json
    └── <new_generation>.json
```

All three split directories use the same filename for one generation. The DataModule selects that filename through `metadata_filename` and resolves every record's relative `image` value against `imgs/standard`.

A new ingredient-recognition record must provide at least:

```json
{
  "id": "stable-record-id",
  "image": "relative-image-name.jpg",
  "cuisine": "French",
  "ingredients": ["2 tablespoons unsalted butter"],
  "ingredients_target": ["butter"]
}
```

Additional source fields may be preserved. Split and image-root paths are not repeated inside each record.

## Verified implementation findings

### Current loader coupling

[`../../../src/data_processing/images_recipes.py`](../../../src/data_processing/images_recipes.py) currently uses the same stage directory both to open `metadata_filename` and to resolve `record["image"]`. `ImagesRecipesBaseDataModule` builds `data_dir/<split>` and `images_recipes_processing()` passes it to both operations. The path contract must therefore be separated before moving any image.

The experiment configuration currently persists `data_dir`, `metadata_filename`, and `feature_label`, but it has no common-image-directory setting. A relative `images_subdir`, defaulting to `imgs/standard`, is preferred over another absolute path because `data_dir` already has Windows/WSL remapping logic.

### Current metadata generations

The canonical metadata contains 54,724 train, 5,210 validation, and 5,212 test records. A second generation, `sel_ing_2410_metadata.json`, exists in every split and contains 50,866 train, 4,802 validation, and 4,854 test records. Its image references match the canonical records that it retains.

This confirms that keeping multiple same-named generations across the three split folders is already an active behavior.

`ingredients_target_v1_metadata.json` is the first 209-target candidate. `ingredients_target_v4_metadata.json` is a deterministic, post-audit 161-target baseline with 48,282 train, 6,036 validation, and 6,036 test records: a complete read-only rebuild produced exactly the saved three JSON objects. It must not become the runtime default, because its frequency-first filter removes valid normalized ingredients before vocabulary association. `_ingredients_target_v2_metadata.json` and `_ingredients_target_v3_metadata.json` are retained but non-selected diagnostic generations; their leading underscore prevents accidental selection by convention, after their comparison output identified out-of-scope mappings that were removed before `v4`.

### Legacy target preprocessing lineage

Two candidate preprocessing implementations exist:

- [`../../../prev_attempts/attempt1/preprocessing_v2.py`](../../../prev_attempts/attempt1/preprocessing_v2.py) produces flat string targets in `ingredients_ok`, applies frequency filtering, performs a Levenshtein-based merge, removes recipes below three targets, shuffles with seed 42, and writes `recipes_train.json`, `recipes_val.json`, and `recipes_test.json`;
- [`../../../prev_attempts/attempt2/pre_process.py`](../../../prev_attempts/attempt2/pre_process.py), with [`../../../prev_attempts/attempt2/utils.py`](../../../prev_attempts/attempt2/utils.py), classifies ingredients into `[category, subcategory]` pairs and writes nested list targets.

The current `ingredients_ok` values are flat strings, not category/subcategory pairs. The split sizes produced by attempt 1 match the checked-in historical recipe files, and the existing audit found close—but not exact—target agreement when executing its normalizer. Attempt 1 is therefore the most probable lineage. Attempt 2 was not the generator of the current target representation.

Attempt 1 must not be copied unchanged because:

- its support threshold counts distinct raw strings rather than distinct recipes;
- it performs unbounded substring operations that created confirmed collisions;
- its similarity merge depends on unordered multiprocessing results;
- its set conversion makes target order non-deterministic;
- it reproduces the exact current target set for only 80.42% of processed records before the unavailable historical merge state is considered.

The new standardizer will use attempt 1 as forensic input and a source of useful rules, not as an executable dependency.

### Historical experiment storage

`experiments/basic` currently contains five experiment groups, 324 checkpoints totaling approximately 28.16 GB, 106 JSON configuration files, and 111 Lightning YAML files.

The relevant configurations fall into three generations:

1. **104 JSON data configurations** contain `data_dir`, `metadata_filename`, `feature_label="ingredients_ok"`, and serialized label-encoder data.
2. **292 light checkpoints** contain no embedded data configuration. They rely on the JSON configuration in their trial or experiment directory and do not need checkpoint rewriting.
3. **32 full checkpoints** embed data hyperparameters:
   - 23 use the current-style `data_dir`, `metadata_filename`, and `feature_label` keys;
   - 9 older DenseNet checkpoints use `global_images_dir`, `recipes_dir`, and `recipe_feature_label`.

The YAML files describe model hyperparameters and do not contain DataModule fields relevant to compatibility. CSV result files also require no change.

### Historical target compatibility

Old checkpoints must continue to see exactly the metadata field and target values used during training. The existing `metadata.json` and `sel_ing_2410_metadata.json` generations therefore retain `ingredients_ok` unchanged. New standardized `ingredients_target` values are written only to new metadata generations and never replace the targets behind an old checkpoint.

The compatibility layer resolves the new common image directory at load time without changing saved label encoders, class order, output dimensions, `<UNK>` behavior, JSON configurations, or checkpoint bytes.

## Work package 2.1b — shared image store and metadata decoupling

**Status:** Done

### Required implementation

1. Add `images_subdir`, defaulting to `imgs/standard`, to `ImagesRecipesBaseDataModule` and its serialized configuration.
2. Change `images_recipes_processing()` to receive separate metadata and image directories.
3. Keep `data_dir` as the dataset root and preserve its Windows/WSL resolution behavior.
4. Validate that `metadata_filename` exists in every requested split.
5. Validate relative image references and missing files before worker processes start.
6. Add temporary-fixture tests for multiple metadata generations using one common image directory.
7. Implement a dry-run-first migration that stages the current images in `imgs/standard` without changing their bytes or filenames.
8. Verify file counts and checksums before retiring split-local image copies.

The migration must preserve image filenames. SHA-256 is used for verification and later duplicate grouping, not as a required storage filename.

### Compatibility boundary

This work package does not rename `ingredients_ok`, regenerate targets, rewrite experiment artifacts, or decide the role of `<UNK>`.

### Completion gate

Both current metadata generations load through the common image directory; no Yummly loader derives image paths from a split directory; migration checks pass; and the old split-local copies can be removed safely.

## Work package 2.1c — historical experiment compatibility

**Status:** Deferred

### Purpose

Keep the historical experiments selected for retention loadable after the common-image and target-field refactors without modifying their saved configurations or checkpoints and without changing what any historical model predicts.

### Compatibility implementation

Implement compatibility in configuration loading and add a dedicated verification script, preferably `scripts/validate_legacy_experiments.py`. The script audits and smoke-tests historical artifacts; it does not rewrite them.

The implementation must:

1. retain the legacy `metadata.json` and `sel_ing_2410_metadata.json` fields and values unchanged, including `ingredients_ok`;
2. recognize current-style saved configurations with `data_dir`, `metadata_filename`, and `feature_label="ingredients_ok"`;
3. supply `images_subdir="imgs/standard"` in memory when an old configuration does not contain it;
4. translate the nine older DenseNet checkpoint schemas from `global_images_dir`, `recipes_dir`, and `recipe_feature_label` into the current DataModule arguments in memory;
5. preserve the explicitly saved `feature_label="ingredients_ok"` instead of replacing it with the new default;
6. leave the 104 JSON configurations, 324 checkpoints, label encoders, model/optimizer state, output dimensions, and `<UNK>` untouched on disk;
7. ignore YAML and result CSV files that do not contain DataModule configuration;
8. fail clearly on an unknown historical schema instead of guessing;
9. keep the compatibility translations isolated and tested so they can be removed only when the legacy experiments are intentionally retired.

### Verification

- Confirm that legacy metadata files remain byte-unchanged.
- Confirm that saved JSON and checkpoint files remain byte-unchanged.
- Confirm that encoder classes and indices are unchanged after in-memory configuration translation.
- Load one representative experiment from each configuration generation.
- Resolve at least one image and one target batch for `metadata.json` and `sel_ing_2410_metadata.json`.
- Run the verification script repeatedly and confirm that it performs no writes.

### Completion gate

Every selected retained experiment loads unchanged legacy metadata through the new image layout while retaining `ingredients_ok`, its original target values, and its original model semantics.

## Work package 2.2 — improved ingredient-target standardization

**Status:** In progress

### Purpose

Create a new deterministic script that derives `ingredients_target` from each recipe's original `ingredients` lines. This is the preprocessing used by new metadata generations.

### Preliminary lineage conclusion

Attempt 1 is the correct historical starting point; attempt 2 is a separate category/subcategory experiment. The exact historical mapping is unavailable, so the new script must prioritize correctness and determinism rather than exact reproduction of `ingredients_ok`.

### Required implementation

1. Extract useful normalization behavior from attempt 1 into small pure functions.
2. Define the new normalization, frequency, alias, generalization, and record-retention rules before implementation.
3. Replace substring deletion with token- or phrase-bounded operations.
4. Compute support using distinct recipes, not distinct raw strings.
5. Remove unordered similarity merging; every merge must be an explicit deterministic rule.
6. Preserve the original `ingredients` field unchanged.
7. Write a deterministic, duplicate-free `ingredients_target` list.
8. Do not generate a per-line mapping artifact or require manual mapping review.
9. Add regression tests for the confirmed legacy collisions and for every retained rule borrowed from attempt 1.
10. Emit concise aggregate console statistics so the effect of a generation can be inspected without another permanent artifact.

The exact standardization rules, support threshold, minimum retained targets, and desired level of ingredient generalization are intentionally left for a focused follow-up discussion.

### Completion gate

Repeated runs on identical input produce identical `ingredients_target` values and ordering; known collisions are covered by tests; and the aggregate effect of the preprocessing is understood before it is used for a split.

## Work package 2.2a — ingredient vocabulary audit

**Status:** Done — discussion checkpoint reached

### Purpose

Review the candidate vocabulary produced by Work package 2.2 before it is treated as the benchmark target. The aim is to reduce avoidable fragmentation and unsupported target distinctions without recreating the legacy pipeline's fuzzy or order-dependent merges.

### Required investigation

1. Extract the unique `ingredients_target` values, per-target recipe support, target cardinality, and train/validation/test support from the candidate generation.
2. Compute interpretable target-pair relationships: co-occurrence counts, conditional probabilities, and Jaccard similarity.
3. Inspect lexical relationships using token-aware phrase containment and the raw ingredient lines that produced each candidate. Flag possible singular/plural variants, aliases, overly specific variants, and semantically misleading generalizations.
4. Re-run the known legacy substring-collision audit against the new targets and add targeted checks for any new candidate collision discovered during the review.
5. Classify each finding provisionally as: retain, candidate merge, candidate rule replacement, candidate exclusion, or unresolved. Do not apply these classifications to the extractor.
6. Record strong findings in temporary research notes during the analysis and present the principal evidence, candidate changes, trade-offs, and unresolved cases to the user for discussion. Do not create or retain a per-raw-line mapping artifact.
7. Stop after the discussion-ready report. Applying any rule is Work package 2.2b, not part of this audit.

### Non-goals

- No automatic merge based only on co-occurrence, similarity score, embedding distance, or lexical containment.
- No manual relabeling of individual recipes.
- No change to legacy `ingredients_ok` or historical experiment artifacts.
- No use of cuisine, image similarity, recipe names, or model predictions as a target-normalization rule.

### Completion gate

The vocabulary size and support profile are understood; the main findings and candidate refinements are recorded in [`../../project_objective/ingredient_vocabulary_audit.md`](../../project_objective/ingredient_vocabulary_audit.md) for discussion; and no extractor, metadata, or split was changed by this work package. This gate is satisfied.

## Work package 2.2b — ingredient extractor strengthening

**Status:** Done

### Purpose

Turn only the explicitly accepted findings from Work package 2.2a into tested, deterministic extractor rules.

### Approved recognizability policy and scope

Target granularity is determined primarily by whether the distinction remains realistically recognizable in a prepared-dish image. The rule applies within ingredient families: merge fine preparation, product-style, or naming variants that are not visually separable, but retain variants with a robust visual distinction and retain different source-ingredient families.

The durable, explicit source-to-target contract is maintained in [`../../implementation_details/ingredient_mapping_rules.md`](../../implementation_details/ingredient_mapping_rules.md). The summary below defines this work package's approved scope but must not become a separate mapping authority.

The following scope is approved:

1. apply the conservative normalization package documented by Work package 2.2a;
2. exclude generic `sauce`;
3. expand `salt and pepper` into `salt` and `pepper`;
4. map explicit fresh/leaf coriander to `cilantro`;
5. merge `chicken stock`, `chicken broth`, and low-sodium chicken broth into `chicken broth`;
6. merge base, light, dark, and low-sodium soy sauce into `soy sauce`;
7. merge all current plain and Greek yogurt spellings and variants into `yogurt`;
8. merge `toasted sesame oil` into `sesame oil`;
9. merge `tomato paste` and `tomato sauce` into `tomato sauce`, while retaining fresh `tomato`;
10. retain red and green bell-pepper targets separately;
11. retain the source-support threshold of 500 recipes and the minimum-three-target retention rule for the first comparison build.
12. merge ground and powder spice forms into the base spice while keeping seeds, leaves, and sticks separate;
13. retain `red onion` and `green onion`, while collapsing white, yellow, Spanish, and sweet onion into `onion`;
14. collapse brown, white, granulated, and light-brown sugar into `sugar`;
15. merge and retain all bare, fresh, and leaf `coriander`/`cilantro` forms under the canonical `cilantro` target, while explicit ground/powder forms follow the base-spice rule;
16. map bare `red pepper` and `green pepper` to the corresponding red and green bell-pepper targets, keeping the colours separate.
17. collapse `chili powder`, `ground red pepper`, `crushed red pepper`, `dried crushed red pepper`, `red pepper flakes`, and equivalent generic dried/powdered forms into `chili`.

The family boundary prevents an uncontrolled collapse: for example, sesame oil and olive oil remain distinct source ingredients even though toasted and untoasted sesame oil merge.

### Required implementation

1. Keep the final approved target changes and boundaries synchronized with [`../../implementation_details/ingredient_mapping_rules.md`](../../implementation_details/ingredient_mapping_rules.md). **Done.**
2. Implement only the agreed token- or phrase-bounded rules. **Done.**
3. Add a regression test for every accepted rule and its relevant collision boundary. **Done.**
4. Compare vocabulary size, target support, record retention, and affected recipes with the `v1` candidate. **Done:** `v4` retains 60,354 records (196 fewer than `v1`) and 161 targets (48 fewer), with 40,525 changed shared records; the six new canonical labels are `almond`, `banana`, `chili`, `pecan`, `walnut`, and `yogurt`.
5. Regenerate the exact-SHA-safe metadata generation through Work package 2.3. **Done:** `ingredients_target_v4_metadata.json`.

### Completion gate

Every extractor change is traceable to an explicit decision made after the 2.2a discussion, passes regression tests, and is reflected in the selected `v4` metadata generation. This gate is satisfied.

## Work package 2.2c — controlled vocabulary research

**Status:** Done

### Purpose

Determine whether a controlled external vocabulary can provide the canonical ingredient concepts for new Yummly targets without adding a runtime service, duplicate metadata artifacts, or a manual review workflow. This work package standardizes semantic concepts; the later Ingredient selection macro-section will determine which concepts are sufficiently visually distinguishable for particular experiments.

This work package extends the completed 2.2a audit and 2.2b implementation. The existing `v4` generation remains reproducible comparison evidence, but must not become the runtime default while this research is in progress.

### Pre-research pipeline hypothesis

Work package 2.2c was initially scoped against the following provisional order. It is retained here as planning history, not as the approved 2.2d contract. The project uses one shared `ingredients_target` vocabulary; a later Ingredient selection phase may define explicitly named subsets for visual-distinguishability experiments.

The provisional order was:

1. Preserve the original line and attempt a deterministic association directly to a concept in the selected vocabulary.
2. When no association is found, apply the small, explicit standardization round needed for that vocabulary, using only tested phrase- or token-bounded rules.
3. Attempt the vocabulary association again on the standardized result.
4. If it still has no vocabulary concept, retain the standardized result as its own canonical concept rather than silently deleting it or assigning `<UNK>`.
5. Deduplicate and deterministically order the recipe concepts; only then apply the current support threshold and minimum-target retention rule.

The provisional comparison retained support 500 and a minimum of three targets, applied to final concepts rather than raw normalized strings. The completed analysis demonstrates unacceptable information loss at those values. A replacement threshold and minimum-target rule now require an explicit train-only sweep before the single shared `ingredients_target` vocabulary is generated.

The mapping is implemented as part of the deterministic standardizer. It does not require a `vocabulary.json`, a raw-line mapping file, or an online lookup at data-loading or inference time.

### Research questions

1. Which candidate vocabulary has sufficient coverage and stable identifiers for recipe ingredients, including processed products and culturally specific items?
2. Does it offer stable lexical concepts and synonyms without requiring automatic hierarchy traversal?
3. Can direct matching plus the small fallback standardization round cover the corpus deterministically and transparently, and is a bounded fuzzy recovery step justified for the remaining cases?
4. After association, do the threshold and minimum-target rules retain useful recipe content and a viable multi-label learning problem?

### Required investigation

1. Identify and document candidate vocabularies, beginning with FoodOn and using alternatives only where they meet a distinct need.
2. Inspect their licence, release format, identifier stability, hierarchy, food/ingredient coverage, synonym support, and suitability for offline deterministic use.
3. Run reproducible corpus-level coverage experiments on raw lines, directly matched concepts, fallback-standardized concepts, and remaining standalone concepts. Do not use test data to choose a final support threshold or visual target level.
4. Analyse representative edge cases, including `fish fillets`, `English muffins`, sauces, stock/broth, tomato products, peppers, and cuisine-specific ingredients.
5. Compare candidate concept levels for semantic correctness, recognizability in prepared-dish images, support after association, and the effect on recipe retention.
6. Present the recommendation, quantified trade-offs, unresolved ambiguities, and a proposed minimal implementation contract before modifying the production standardizer or generating another metadata version.

All six investigation items are complete. The reusable resource catalog is in [`../../research/topics/ingredient_vocabularies/vocabulary_catalog.md`](../../research/topics/ingredient_vocabularies/vocabulary_catalog.md); the corpus experiment, threshold analysis, edge cases, and proposed contract are in [`controlled_vocabulary_evaluation.md`](controlled_vocabulary_evaluation.md).

### Research outcome

- FoodOn v2025-07-31 is the preferred external reference, but it is not a complete or automatically usable target space. On the legacy train split, 30.11% of ingredient-line occurrences associate directly and 21.05% after the bounded fallback; 4.66% remain ambiguous and 44.17% remain transparent local concepts.
- FoodOn is the primary association authority: first attempt direct lexical association on the mechanically cleaned line, then use the existing bounded standardization only for a line that FoodOn did not associate, and retry the association. The 2.2b recognizability rules remain a conservative fallback, not a pre-association override; visual label selection belongs to Macro-section 3.
- A bounded fuzzy evaluation is complete and rejected. A strict one-edit typo rule recovers only 99 of 34,369 local train terms and still produces semantic collisions such as `fish stock` -> `fish stick`; a two-edit rule produces much more severe collisions. Fuzzy association is excluded from 2.2d.
- Explicit reviewed mappings to FoodOn parents are retained only as a deferred, separately versioned experiment for reducing problem difficulty; they are not active in the standard pipeline and must be compared against the unchanged `ingredients_target` baseline.
- Correct vocabulary association does not solve the observed deletion: `english muffin` and `fish fillet` have train support 35 and 50, so the threshold of 500 still removes them.
- The reproducible train-only sweep selected support >= 500 distinct train recipes and at least three retained targets per recipe. The values must be revalidated numerically after the approved association pipeline runs.
- The selected threshold will define one shared `ingredients_target` vocabulary used across all splits and model implementations. Optional subsets may exist later only as explicitly named experimental projections, not as a second default vocabulary.
- No production standardizer, metadata generation, split, runtime default, or legacy artifact was changed by the research.

### Completion gate

The research records an evidence-backed vocabulary decision, defines the association and fallback contract, and demonstrates its expected effect on coverage and retention. The exact-plus-fallback protocol is complete, fuzzy recovery is explicitly rejected, and the contract is implemented in `v5`.

## Work package 2.2d — controlled-vocabulary target-generation implementation

**Status:** Done

### Purpose

Implement only the vocabulary, concept level, and deterministic contract approved after Work package 2.2c. This is a new implementation phase; it does not rewrite the completed 2.2b rules, metadata, or tests.

### Implemented contract and results

The approved contract is implemented by the offline FoodOn index and the
dedicated builder. The legacy metadata generations remain immutable and are
not used as runtime inputs for the new generation.

1. [x] Add the selected vocabulary as the compact offline index `src/data_processing/resources/foodon_food_product_v2025_07_31.json`.
2. [x] Associate each raw ingredient line directly to a selected concept before applying fallback normalization.
3. [x] Apply only the approved phrase- or token-bounded fallback standardization to unmatched lines, then retry exact association; fuzzy recovery remains disabled.
4. [x] Retain an unmatched or unresolved standardized term as its own canonical concept; never use `<UNK>` as a multi-label target.
5. [x] Extend `ingredient_threshold_sweep.py` and persist the post-association train-only threshold comparison.
6. [x] Revalidate support >= 500 and minimum-three-target filtering using only the source training partition; share the resulting vocabulary across all three output splits.
7. [x] Extend the regression suite with FoodOn precedence, plural retry, local residual, and train-only support cases.
8. [x] Produce `ingredients_target_v5_metadata.json` without modifying legacy generations; compare it with `v4` and pass the Work package 2.3 integrity validator.

### Implementation artifacts and results

- [`../../../src/data_processing/foodon_lexicon.py`](../../../src/data_processing/foodon_lexicon.py) loads the pinned offline index and performs exact lexical association plus vocabulary-validated final-token singularization.
- [`../../../src/data_processing/ingredient_target_generation.py`](../../../src/data_processing/ingredient_target_generation.py) implements exact FoodOn, fallback, retry, and local-concept retention.
- [`../../../scripts/build_yummly_foodon_metadata.py`](../../../scripts/build_yummly_foodon_metadata.py) generates the new metadata and reuses the exact-image-aware split validator.
- [`../../../src_scratches/data_anlysis/outputs/controlled_target_generation/report.json`](../../../src_scratches/data_anlysis/outputs/controlled_target_generation/report.json) records the aggregate generation result.
- [`../../../src_scratches/data_anlysis/outputs/controlled_target_generation/threshold_sweep.json`](../../../src_scratches/data_anlysis/outputs/controlled_target_generation/threshold_sweep.json) records the post-association threshold comparison.

The generated `v5` candidate contains 47,965/5,996/5,996 train/validation/test
records and 165 train-supported canonical targets. It retains 59,957 records
before the 80/10/10 allocation, compared with 60,354 records and 161 targets
in the `v4` baseline. The builder passed automatic image decoding, exact
SHA-256 leakage, uniqueness, ratio, distribution, target-vocabulary, and
minimum-cardinality assertions. The motivating `English muffins` and `fish
fillets` lines are now associated to FoodOn concepts before filtering, but
remain below the deliberately retained support threshold of 500 in this
standard generation.

Against `v4`, `v5` shares 59,825 record IDs, removes 529 records, adds 132
records, changes the target list for 49,214 shared records, and changes the
deterministic split assignment for 20,326 shared records. The full target-name
comparison is retained in the aggregate report.

A clean rebuild produced byte-identical JSON files: train SHA-256
`30fa8d1aec04ca951c5a498f4c08c7f458b8bdfa754dd3a1fb839c9d898c221b`,
validation `67ea919607e34ef249553b4fcf9c2851e4dc341a064a23bd405175c83f8ce3f6`,
and test `f48a4f8cae649dfe994c1bd8181782af2b107e74a18746dfbb0c1f9dc420cbf5`.

### Completion gate

The approved pipeline is deterministic and tested; its replacement metadata
generation passes the existing split and integrity checks; and its coverage,
support, vocabulary, and record-retention changes versus `v4` are recorded.
Work package 2.4 can now resume without changing the new data generation.

## Work package 2.3 — deterministic metadata generation and split

**Status:** Done

### Required implementation

1. Start from the available Yummly records and common images.
2. Generate `ingredients_target` with Work package 2.2.
3. Apply only automatic image existence and decoding checks.
4. Compute image SHA-256 temporarily and treat records with identical bytes as one allocation group.
5. Do not use perceptual hashes, names, ingredients, or manual decisions to create groups.
6. Assign exact-image groups to a deterministic 80/10/10 train/validation/test split.
7. Balance cuisine and `ingredients_target` distributions within documented tolerances.
8. Write one chosen metadata filename under each split and preserve the same record schema.
9. Derive split membership exclusively from those three files.
10. Run automatic assertions and stop on failure rather than writing a standalone validation report.

### Mandatory assertions

- every record ID is unique and appears in exactly one split;
- every referenced image exists below `imgs/standard` and decodes;
- no exact SHA-256 image group crosses splits;
- the requested split ratios and cuisine/target tolerances pass;
- every record has a valid `ingredients_target` list according to the chosen retention rule;
- the same metadata filename exists in all three splits;
- a rerun with the same input and seed produces identical record assignments and metadata content.

### Completion gate

The selected `v4` metadata files pass all automatic assertions, have no exact-image leakage, and rebuild identically from the source metadata and rules. This gate is satisfied.

## Work package 2.4 — runtime target integration and `<UNK>` decision

**Status:** In progress

### Required implementation

1. Keep `feature_label` configurable and set its default to `ingredients_target`.
2. Fit or reconstruct the vocabulary deterministically from training metadata and preserve it in experiment/checkpoint configuration.
3. Remove `<UNK>` from the vocabulary and output space of new multi-label experiments: it has no positive target in their training data.
4. Preserve the saved `<UNK>` class and output dimension for any legacy experiment selected for retention; do not rewrite its artifacts.
5. Treat any future ingestion or sequence-token use as a separate, explicitly documented decision rather than retaining it in the multi-label output by default.
6. Add tests for the selected behavior before changing existing encoders.
7. Update image-statistics and dashboard consumers to use the common image root and selected target field.
8. Run a minimal training, checkpoint reload, and dashboard smoke test.

### Implementation result

- `YUMMLY_TARGET_METADATA_FILENAME` and new experiment configurations now select `ingredients_target_v5_metadata.json`.
- New `ingredients_target` datasets use the strict `MultiLabelBinarizer`; their training-derived vocabulary has 165 classes and no `<UNK>` output.
- Alternative and legacy fields continue to default to `MultiLabelBinarizerRobust`, while serialized legacy encoder configurations reconstruct their saved classes, `<UNK>` index, and output dimension.
- Unknown labels presented to the strict encoder now fail explicitly instead of being silently assigned to the last real class.
- `compute_img_stats.py` now defaults to the selected Yummly metadata and resolves images through `imgs/standard`; its dataset root, metadata, target field, and image subdirectory remain configurable.
- Sixteen unit and data-contract tests pass, including full train/validation/test transformation against `v5`.
- The full training/checkpoint/dashboard smoke test remains pending because the available Windows environment lacks `lightning` and its Torch build targets NumPy 1.x while NumPy 2.1 is installed. This is an environment verification dependency, not an unresolved target-policy decision.

### Completion gate

New experiments default to `ingredients_target` and omit `<UNK>` from their multi-label outputs, alternative feature labels remain supported, retained historical experiments preserve their semantics, and the policy is covered by tests.

## Ordered delivery sequence

```text
2.1b shared image store and loader path separation
  -> 2.2 baseline ingredients_target standardizer
  -> 2.2a ingredient vocabulary audit and discussion
  -> 2.2b accepted extractor strengthening
  -> 2.2c controlled vocabulary research and decision gate
  -> 2.2d controlled-vocabulary target-generation implementation
  -> replacement exact-duplicate-aware stratified metadata generation
  -> 2.4 runtime default and <UNK> decision
  -> freeze the first selected generation

2.1c legacy experiment compatibility (deferred; resume after selecting retained experiments)
```

Work package 2.2a may proceed while Work package 2.1c is deferred, but legacy metadata must remain isolated from the new standardizer.

## Risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Shared-image migration is partial | Some metadata loads only because stale split images remain | Stage and verify the common directory before removing split copies |
| Old experiment is pointed at newly standardized targets | Output indices no longer match training semantics | Preserve legacy metadata and its explicit `ingredients_ok` field |
| Historical schemas no longer match the current loader | Old experiments cannot be inspected | Translate known schema generations in memory and test representative artifacts |
| Attempt 1 is treated as exact ground truth | Known collisions and nondeterminism are reintroduced | Use it only as lineage evidence and test every adopted rule |
| Fuzzy duplicate grouping introduces subjective bias | Valid records are coupled by arbitrary thresholds | Group only byte-identical SHA-256 images |
| Cuisine-only splitting produces label drift | Rare targets become unreliable in evaluation | Balance cuisine and ingredient targets together |
| `<UNK>` removal breaks a retained legacy output shape | A saved checkpoint cannot be loaded | Omit it only from new multi-label vocabularies; preserve saved legacy behavior for selected retained experiments |
| Frequency filtering precedes semantic association | Valid ingredients disappear before they can be generalized to an appropriate concept | Associate to the selected controlled vocabulary first; apply support only to final concepts and reassess the provisional threshold from evidence |

## Completion criteria for this plan

The plan is complete when:

- all current images are served from the common directory;
- historical metadata and experiments remain loadable without modifying saved artifacts;
- the controlled-vocabulary pipeline deterministically produces `ingredients_target` from `ingredients`;
- the vocabulary decision and final ingredient vocabulary have passed the evidence-backed Work package 2.2c audit;
- the new split has no exact-image leakage and acceptable cuisine/target balance;
- the DataModule defaults to `ingredients_target` while retaining `feature_label`;
- the `<UNK>` decision has been investigated, documented, implemented, and tested;
- the first new metadata generation passes all automatic checks and a minimal runtime smoke test.

## Decision log

| Date | Decision or change | Rationale |
| --- | --- | --- |
| 2026-08-02 | Created the initial Data implementation plan | Data layout and benchmark construction required one coordinated plan |
| 2026-08-02 | Simplified outputs to shared images plus split metadata generations | The earlier manifest-heavy design duplicated information not consumed by this thesis project |
| 2026-08-02 | Standardized future targets on configurable `feature_label="ingredients_target"` | One field name supports all new ingredient-recognition models without removing loader flexibility |
| 2026-08-02 | Identified attempt 1 as the probable `ingredients_ok` lineage | It produces flat targets and matching historical split sizes; attempt 2 produces incompatible nested categories |
| 2026-08-02 | Replaced saved-file migration with a legacy compatibility layer | Keeping legacy metadata and translating paths in memory is safer than rewriting 28 GB of checkpoints across multiple schemas |
| 2026-08-02 | Limited duplicate grouping to exact SHA-256 identity | Exact equality prevents direct leakage without similarity thresholds or manual bias |
| 2026-08-02 | Reopened the `<UNK>` decision | It may still be useful for ingestion, filtered vocabularies, or sequence models even if it is questionable as a multi-label output |
| 2026-08-03 | Created the shared image store and first target generation | The migration is additive and checksum-verified; `ingredients_target_v1_metadata.json` is the first deterministic 209-label generation, while legacy artifacts remain unchanged |
| 2026-08-03 | Added Work packages 2.2a–2.2b for vocabulary work | The 209-label `v1` generation is a validated candidate; 2.2a is analysis and discussion only, while extractor changes require subsequent explicit approval in 2.2b |
| 2026-08-04 | Deferred historical experiment compatibility | Select the historical experiments to retain before investing in schema-specific compatibility and smoke tests; this does not block the vocabulary audit |
| 2026-08-04 | Accepted `<UNK>` removal for new multi-label outputs | It receives no positive training targets; any selected legacy artifact retains its saved behavior |
| 2026-08-04 | Completed the 2.2a candidate vocabulary audit without changing the extractor | The 209-target candidate is split-stable but contains mechanical fragmentation, composite and mixed targets, and unresolved granularity choices; 2.2b remains gated on explicit approval |
| 2026-08-04 | Adopted recognizability-led target granularity for 2.2b and approved the first merge scope | Broth/stock, all soy-sauce styles, yogurt variants, toasted/base sesame oil, and tomato paste/sauce are not reliably separable in prepared images; red and green bell peppers remain distinct |
| 2026-08-04 | Approved the remaining spice, onion, sugar, coriander, and fresh-pepper rules | Ground/powder spices merge into their base while seeds/leaves/sticks remain separate; red/green onion distinctions are retained as specified; sugars collapse; coriander/cilantro is retained as one target; bare red/green pepper maps by colour to bell pepper |
| 2026-08-04 | Finalized the chili-family rule and closed the 2.2b decision gate | Generic chili powder and crushed/flaked dried red pepper all collapse into `chili`; fresh red and green bell peppers remain separate |
| 2026-08-04 | Implemented and selected the post-audit target generation | `ingredient_standardization.py` and its regression tests implement the approved mappings; `v4` is a deterministic 161-target, 60,354-record candidate. `v2` and `v3` remain non-selected diagnostic generations after their comparison output exposed out-of-scope intermediate mappings. |
| 2026-08-04 | Created a durable ingredient mapping registry | [`../../implementation_details/ingredient_mapping_rules.md`](../../implementation_details/ingredient_mapping_rules.md) is the long-term source of truth for active and approved mappings, expansions, exclusions, and collision boundaries; the feature plan retains only scope and execution status |
| 2026-08-04 | Added Work packages 2.2c–2.2d for controlled-vocabulary target generation | The completed 2.2a audit and 2.2b implementation remain historical evidence. Work package 2.2c evaluates the vocabulary and association contract; 2.2d implements it only after approval. |
| 2026-08-04 | Completed the 2.2c controlled-vocabulary research checkpoint | FoodOn was recommended as a pinned reference lexicon with retained local concepts and no automatic parent traversal. The original recommendation separated label selection from semantic metadata; the later decision recorded below supersedes that part with one shared `ingredients_target` vocabulary. |
| 2026-08-04 | Deferred the final filtering policy to a threshold-sweep review | Before choosing a support threshold or minimum-target rule, a reproducible train-only script must compare vocabulary size, assignment and recipe retention, and concrete ingredients gained or lost across candidate thresholds. |
| 2026-08-04 | Kept one shared `ingredients_target` vocabulary | The standard pipeline will not split semantic and learnable vocabularies. Optional subsets must be explicitly named experiments, while all standard models use the same target vocabulary. |
| 2026-08-04 | Retained explicit parent abstraction as a deferred experiment | Automatic FoodOn parent traversal remains prohibited in the standard pipeline, but reviewed and versioned child-to-parent mappings may later be tested to reduce label-space difficulty without replacing the baseline. |
| 2026-08-05 | Executed the provisional train-only threshold sweep | The current normalizer yields 143, 260, and 562 targets at support 500, 250, and 100 respectively; the durable output also records recipe-cardinality buckets and named target losses at every threshold transition. The result informs the policy decision but must be revalidated after FoodOn association before freezing a replacement generation. |
| 2026-08-05 | Selected the standard target-filter policy | Retain only concepts appearing in at least 500 train recipes and retain recipes with at least three resulting `ingredients_target` values. This preserves the established filter contract, gives every standard output class at least 500 positive train examples, and will be revalidated after association. |
| 2026-08-05 | Revised the FoodOn association order | Attempt exact FoodOn association on the mechanically cleaned source line first; use the existing local standardization only for an unassociated line, then retry FoodOn. The `v4` recognizability mappings do not override a direct FoodOn concept, and visual distinguishability is deferred to Macro-section 3. A bounded fuzzy recovery after fallback remains to be evaluated. |
| 2026-08-05 | Rejected bounded fuzzy FoodOn recovery | The strict one-edit evaluator recovered only 99 of 34,369 local train terms and still produced wrong semantic associations; the broader two-edit variant produced severe collisions. The approved 2.2d contract remains exact FoodOn, local fallback, exact retry, then local concept. |
| 2026-08-05 | Completed Work package 2.2d | The compact pinned FoodOn index, FoodOn-first generator, post-association threshold sweep, regression tests, and `v5` metadata are complete. A clean rebuild reproduced all three saved files byte-for-byte; runtime integration remains deferred to 2.4. |
| 2026-08-06 | Implemented Work package 2.4 runtime policy | New configurations select `v5` and a strict 165-class encoder without `<UNK>`; legacy robust configurations preserve `<UNK>`. Regression and split-contract tests pass. Training, checkpoint-reload, and dashboard smoke execution remains pending because the available ML environment is incompatible. |
