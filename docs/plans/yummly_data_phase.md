# Yummly data-phase implementation plan

**Created:** 2026-08-02  
**Last updated:** 2026-08-02

This plan translates the Data macro-section of [`general_plan.md`](../general_plan.md) into a deliberately small implementation sequence. It covers the shared-image-store prerequisite, compatibility with historical experiments, generation of `ingredients_target`, deterministic split construction, and runtime integration.

The plan avoids persistent intermediate artifacts that are not consumed by the project. The benchmark outputs remain the common image directory and one selected metadata file in each split.

## Progress tracker

**Overall status:** In progress  
**Current task:** Finalize the narrow Work package 2.1b contract.
**Next action:** Implement path-resolution tests and the shared-image-store refactor without changing target semantics.

| # | Task | Status | Evidence or result |
| --- | --- | --- | --- |
| P0 | Inspect the current data layout, loaders, legacy preprocessing, and historical experiment artifacts | **Done** | [Verified implementation findings](#verified-implementation-findings) |
| P1 | Agree on the simplified benchmark scope | **Done** | [Accepted design](#accepted-design) |
| P2 | Implement and verify the shared-image-store prerequisite | **Pending** | Work package 2.1b |
| P3 | Implement and verify historical experiment compatibility without rewriting saved artifacts | **Pending** | Work package 2.1c |
| P4 | Design and implement the improved `ingredients` to `ingredients_target` standardizer | **Pending** | Work package 2.2 |
| P5 | Implement deterministic exact-duplicate-aware splitting and metadata generation | **Pending** | Work package 2.3 |
| P6 | Integrate the new target default and resolve the role of `<UNK>` | **Pending** | Work package 2.4 |
| P7 | Run all data checks and freeze the first new metadata generation | **Pending** | Data completion gate |

## Accepted design

The implementation follows these decisions:

- keep `feature_label` configurable and change only its default from `ingredients_ok` to `ingredients_target`;
- derive new `ingredients_target` values from the original `ingredients` field through an improved deterministic preprocessing script;
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
- No ontology service or separate vocabulary artifact.
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

[`../../src/data_processing/images_recipes.py`](../../src/data_processing/images_recipes.py) currently uses the same stage directory both to open `metadata_filename` and to resolve `record["image"]`. `ImagesRecipesBaseDataModule` builds `data_dir/<split>` and `images_recipes_processing()` passes it to both operations. The path contract must therefore be separated before moving any image.

The experiment configuration currently persists `data_dir`, `metadata_filename`, and `feature_label`, but it has no common-image-directory setting. A relative `images_subdir`, defaulting to `imgs/standard`, is preferred over another absolute path because `data_dir` already has Windows/WSL remapping logic.

### Current metadata generations

The canonical metadata contains 54,724 train, 5,210 validation, and 5,212 test records. A second generation, `sel_ing_2410_metadata.json`, exists in every split and contains 50,866 train, 4,802 validation, and 4,854 test records. Its image references match the canonical records that it retains.

This confirms that keeping multiple same-named generations across the three split folders is already an active behavior.

### Legacy target preprocessing lineage

Two candidate preprocessing implementations exist:

- [`../../prev_attempts/attempt1/preprocessing_v2.py`](../../prev_attempts/attempt1/preprocessing_v2.py) produces flat string targets in `ingredients_ok`, applies frequency filtering, performs a Levenshtein-based merge, removes recipes below three targets, shuffles with seed 42, and writes `recipes_train.json`, `recipes_val.json`, and `recipes_test.json`;
- [`../../prev_attempts/attempt2/pre_process.py`](../../prev_attempts/attempt2/pre_process.py), with [`../../prev_attempts/attempt2/utils.py`](../../prev_attempts/attempt2/utils.py), classifies ingredients into `[category, subcategory]` pairs and writes nested list targets.

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

**Status:** In progress

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

**Status:** Pending

### Purpose

Keep all experiments below `experiments/basic` loadable after the common-image and target-field refactors without modifying their saved configurations or checkpoints and without changing what any historical model predicts.

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

Every recognized experiment under `experiments/basic` loads unchanged legacy metadata through the new image layout while retaining `ingredients_ok`, its original target values, and its original model semantics.

## Work package 2.2 — improved ingredient-target standardization

**Status:** Pending

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

## Work package 2.3 — deterministic metadata generation and split

**Status:** Pending

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

The new train, validation, and test metadata pass all assertions, have no exact-image leakage, and can be regenerated deterministically without auxiliary manifests.

## Work package 2.4 — runtime target integration and `<UNK>` decision

**Status:** Pending

### Required implementation

1. Keep `feature_label` configurable and set its default to `ingredients_target`.
2. Fit or reconstruct the vocabulary deterministically from training metadata and preserve it in experiment/checkpoint configuration.
3. Verify how validation/test-only labels, cuisine filtering, selected-ingredient generations, and sequence encoders currently use `<UNK>`.
4. Distinguish an ingestion fallback or sequence-input token from a trainable multi-label output class.
5. Decide whether `<UNK>` should remain an output, become diagnostic-only, or have different behavior by encoder type.
6. Add tests for the selected behavior before changing existing encoders.
7. Update image-statistics and dashboard consumers to use the common image root and selected target field.
8. Run a minimal training, checkpoint reload, and dashboard smoke test.

### Completion gate

New experiments default to `ingredients_target`, alternative feature labels remain supported, historical experiments remain loadable, and the role of `<UNK>` is explicit and covered by tests.

## Ordered delivery sequence

```text
2.1b shared image store and loader path separation
  -> 2.1c legacy experiment compatibility
  -> 2.2 ingredients_target standardizer
  -> 2.3 exact-duplicate-aware stratified metadata generation
  -> 2.4 runtime default and <UNK> decision
  -> freeze the first new generation
```

Work package 2.2 may be designed while 2.1b–2.1c are implemented, but legacy metadata must remain isolated from the new standardizer.

## Risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Shared-image migration is partial | Some metadata loads only because stale split images remain | Stage and verify the common directory before removing split copies |
| Old experiment is pointed at newly standardized targets | Output indices no longer match training semantics | Preserve legacy metadata and its explicit `ingredients_ok` field |
| Historical schemas no longer match the current loader | Old experiments cannot be inspected | Translate known schema generations in memory and test representative artifacts |
| Attempt 1 is treated as exact ground truth | Known collisions and nondeterminism are reintroduced | Use it only as lineage evidence and test every adopted rule |
| Fuzzy duplicate grouping introduces subjective bias | Valid records are coupled by arbitrary thresholds | Group only byte-identical SHA-256 images |
| Cuisine-only splitting produces label drift | Rare targets become unreliable in evaluation | Balance cuisine and ingredient targets together |
| `<UNK>` is removed before its uses are understood | Filtered or sequence workflows regress | Defer the decision to Work package 2.4 and preserve historical behavior during migration |

## Completion criteria for this plan

The plan is complete when:

- all current images are served from the common directory;
- historical metadata and experiments remain loadable without modifying saved artifacts;
- the improved standardizer deterministically produces `ingredients_target` from `ingredients`;
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
