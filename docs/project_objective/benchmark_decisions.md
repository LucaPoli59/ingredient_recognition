# Yummly data and benchmark decisions

**Created:** 2026-08-02  
**Last updated:** 2026-09-24
**Status:** Active and binding

## Purpose

This document records the decisions that govern the Yummly data pipeline and the ingredient-recognition benchmark. The detailed evidence remains in [`yummly_data_audit.md`](yummly_data_audit.md), while implementation order and acceptance checks remain in [`../plans/data_ingredient_refactor/yummly_data_phase.md`](../plans/data_ingredient_refactor/yummly_data_phase.md).

The existing 65,146-record metadata remains valid for historical experiments. It is a legacy artifact, not the target representation for new experiments. New generations use a simpler pipeline that avoids duplicated manifests and subjective review stages.

## Decision summary

| ID | Question | Binding decision |
| --- | --- | --- |
| D1 | Which target field do models use? | Keep `feature_label` configurable. Its default for new configurations becomes `ingredients_target`; historical configurations retain their explicit `ingredients_ok`. |
| D2 | How is `ingredients_target` produced? | Derive it deterministically from the original `ingredients` lines by associating them to a selected controlled vocabulary, with a small tested standardization fallback. Choose the concept level primarily by practical recognizability in the prepared-dish image, and preserve `ingredients` unchanged. |
| D3 | Which persistent artifacts are required? | Store the common images and one selected metadata file per split. Do not create separate mapping, review, family, split, vocabulary, or validation-report artifacts. |
| D4 | How is image quality handled? | Apply automatic existence and decoding checks. Do not add a manual image-review or adjudication workflow; models must tolerate remaining noise. |
| D5 | How are leakage groups and splits built? | Group byte-identical images by SHA-256 only, then create and freeze one reproducible 80/10/10 multi-label-stratified split balanced for cuisine and ingredient targets. Do not use pure random splitting or fuzzy recipe families. |
| D6 | How is the vocabulary represented? | Derive it deterministically from training metadata and save its class order with each experiment or checkpoint. Do not maintain a separate dataset-level vocabulary file. |
| D7 | How are historical experiments kept compatible? | Do not rewrite their metadata, configurations, or checkpoints. Preserve the accepted minimum November 2024 ResNet selection evidence, adapt selected checkpoint anchors in memory, and keep the passing read-only manifest/validator as the gate for any later cleanup. |
| D8 | What happens to `<UNK>`? | Remove it from new multi-label vocabularies and outputs because it has no positive training target. Preserve saved behavior for any legacy experiment selected for retention. |
| D9 | Which primary metrics are used? | Report macro mean average precision and micro F1 together; neither is sufficient alone. |
| D10 | Where are thresholds and calibration selected? | Fit thresholds, calibration, and other selection-time parameters on validation data only. Keep the test split unavailable to selection decisions. |
| D11 | How are ingredient selection and model comparison coupled? | Subphase 4A chooses the experiment model categories, Subphase 4B independently freezes the reference selector, and Macro-section 3 then produces one shared selected vocabulary. Compare model categories on common full and selected tasks; test vocabulary reduction through transferred-hyperparameter and support-matched random-vocabulary controls; keep selected-task local adaptation separate. |
| D12 | Which model-side protocol defines reference-selector learnability? | Use the 4B-D1 EfficientNetV2-S protocol: exact supervised ImageNet-1K weights, full-backbone adaptation, full-frame 384-pixel preprocessing, an independent pooled 165-logit head, and train-only positive-weighted BCE. |
| D13 | Which campaign measures reference-selector learnability? | Use the Phase 3-D1 single-run protocol: seed 42, AdamW with a fixed warm-up/cosine schedule, 20 epochs, deterministic audits every two epochs, AP trajectories, fixed-0.5 F1 diagnostics, finite-validation bootstrap uncertainty, low-cost non-visual controls, and a support-stratified blind pilot cohort. No extra robustness configuration or second pilot training is required. |

## D1: target-field contract

The DataModule continues to accept a configurable `feature_label`. This is required for alternative targets and for compatibility with historical experiments.

- New configurations default to `ingredients_target`.
- Existing metadata files keep `ingredients_ok` unchanged.
- Existing configurations that explicitly select `ingredients_ok` must continue to select it.
- New metadata generations store `ingredients_target` and preserve the original `ingredients` field.
- A legacy checkpoint must never be paired silently with newly standardized targets because its saved class indices and output semantics refer to `ingredients_ok`.

## D2: controlled-vocabulary target generation

The new target-generation pipeline starts from the original ingredient lines, not from `ingredients_ok`. Work packages 2.2c–2.2d selected and implemented the pinned FoodOn-first `v5` pipeline with retained local concepts, exact association before and after the bounded fallback, and no fuzzy recovery. `ingredients_target_v5_metadata.json` is the standard runtime generation.

The most probable historical lineage is [`../../prev_attempts/attempt1/preprocessing_v2.py`](../../prev_attempts/attempt1/preprocessing_v2.py): it produces flat string labels and its split sizes match the historical files. [`../../prev_attempts/attempt2/pre_process.py`](../../prev_attempts/attempt2/pre_process.py) produces nested category/subcategory pairs and was not the generator of the current representation.

Attempt 1 is evidence, not an executable dependency. Any fallback standardization must correct its known defects:

- support is counted across distinct recipes, not distinct raw strings;
- text operations respect token or phrase boundaries;
- aliases and generalizations are explicit and deterministic;
- unordered similarity merging is removed;
- target lists have deterministic, duplicate-free ordering;
- confirmed legacy collisions are regression-tested.

Fine-grained target distinctions are governed primarily by practical recognizability from the prepared-dish image. Within one meaningful ingredient family, preparation or product-style variants that are not realistically separable are collapsed; robust visual distinctions may remain separate. This rule does not collapse different source-ingredient families automatically merely because both can become visually subtle after cooking.

The pipeline will first associate a raw line directly to a selected vocabulary concept. Only an unmatched line receives the small, explicit, phrase- or token-bounded standardization fallback; it is then associated again. If association still fails, the standardized term is retained as its own concept rather than silently removed or mapped to `<UNK>`. Recipe concepts are deduplicated and ordered deterministically before support filtering.

The standard source-support threshold is 500 distinct train recipes and the minimum retained target count is three. Both are applied after vocabulary association to final concepts and produce the shared 165-label `v5` vocabulary. Any alternative threshold or selected ingredient subset must be a separately named experiment rather than a silent replacement default. No per-line mapping table, manual mapping review, or runtime ontology service is required.

The previous 2.2b rules and `ingredients_target_v4_metadata.json` remain tested baseline evidence, not the selected runtime benchmark. [`../implementation_details/ingredient_mapping_rules.md`](../implementation_details/ingredient_mapping_rules.md) remains the registry for the bounded fallback rules used by the current pipeline.

## D3: minimal persistent outputs

The data layout is:

```text
data/input/yummly/
├── imgs/
│   └── standard/
│       └── <image files>
├── train/
│   └── <metadata generation>.json
├── val/
│   └── <metadata generation>.json
└── test/
    └── <metadata generation>.json
```

The same metadata filename identifies one generation across train, validation, and test. Each record contains its relative image filename and selected target field; the loader resolves the image against `imgs/standard`.

Split membership is already encoded by the three metadata files. Target mappings are encoded by the deterministic standardizer. Vocabulary order belongs with an experiment. Validation results are enforced by assertions and concise run output. Persisting the same information in additional JSONL or JSON artifacts would add synchronization risk without a current consumer.

## D4: automatic image checks and accepted noise

Benchmark construction verifies that every referenced image exists and decodes. It does not introduce human exclusion decisions or a permanent review queue.

The audit's suspicious-image and perceptual-similarity outputs remain useful descriptive evidence, but they do not become benchmark-control files. This is deliberate: the thesis evaluates models under realistic noisy metadata and avoids subjective image curation.

## D5: exact-duplicate-safe deterministic split

The audit found 1,113 exact SHA-256 duplicate groups covering 2,406 records. Under the legacy split, 355 exact groups cross split boundaries; 185 validation records and 171 test records have an exact-image connection to training. Exact grouping therefore prevents measurable direct leakage without introducing a similarity threshold.

The split builder must:

1. compute image SHA-256 values during the build;
2. allocate every byte-identical group wholly to one split;
3. use a fully specified deterministic allocation; if randomized tie-breaking is introduced, declare and freeze its seed;
4. target an 80/10/10 record allocation;
5. balance cuisine and `ingredients_target` marginals within documented tolerances;
6. assert that no record or exact-image group crosses splits;
7. produce identical metadata content for identical input, configuration, and seed.

Perceptual hashes, recipe names, ingredient similarity, and manual decisions must not create allocation groups. They are too dependent on arbitrary thresholds or interpretation for the chosen project scope.

The frozen split is shared by every standard model so model comparisons use
identical examples. It is a controlled in-distribution benchmark, not an
out-of-distribution evaluation. The test split is unavailable to model,
threshold, or hyperparameter selection. The general research rationale is
recorded in [`../research/topics/dataset_splitting/split_strategy.md`](../research/topics/dataset_splitting/split_strategy.md);
the current Yummly implementation and validation contract are in
[`../technical_details/data/yummly_benchmark_split/explaination.md`](../technical_details/data/yummly_benchmark_split/explaination.md).

## D6: vocabulary ownership

The output vocabulary is derived from training metadata after the split. Its exact ordered classes must be serialized with the experiment configuration or checkpoint so predictions remain interpretable and reloadable.

Validation and test must never expand or reorder that vocabulary. A separate `vocabulary.json` beside the dataset is unnecessary because it would duplicate experiment state and could drift from the selected `feature_label` or cuisine filter.

## D7: immutable legacy artifacts

The repository contains multiple historical storage generations: JSON-driven experiments, light checkpoints that depend on nearby JSON configuration, current-style full checkpoints, and older DenseNet checkpoints with different DataModule key names.

The minimum retained set for the November 2024 ResNet ingredient-selection process is now selected in [Data Work package 2.1c](../plans/data_ingredient_refactor/yummly_data_phase.md#work-package-21c--historical-experiment-compatibility). It preserves the H1–H4 aggregate evidence and selected trial configurations, all three `sel_ing_2410_metadata.json` files, the analysis and launcher provenance, and bounded executable anchors for full-vocabulary and selected-vocabulary checkpoint loading.

Compatibility for those selected anchors is implemented during loading:

- supply `images_subdir="imgs/standard"` when an older configuration lacks it;
- preserve explicit `feature_label="ingredients_ok"`;
- retain saved label encoders, class order, output dimensions, model state, and `<UNK>` behavior;
- fail clearly on unknown schemas.

The legacy `metadata.json` and `sel_ing_2410_metadata.json` files, retained configurations, metrics, and checkpoints are not rewritten. [`scripts/validate_legacy_experiments.py`](../../scripts/validate_legacy_experiments.py) and its 72-entry [`retention_manifest.json`](../../src_scratches/ingredient_selection_reconstruction/retention_manifest.json) reproduce the exact 40-label intersection, verify artifact hashes, and prove that the selected anchors still load after the image move; this gate passed on 2026-08-12. No deletion is authorized by this decision. The previously proposed old DenseNet-schema translation is outside this bounded retention set and requires a separate selection decision if needed later.

## D8: `<UNK>` is removed from new multi-label outputs

The current multi-label encoder appends `<UNK>`. Across the current full Yummly data, every observed target label already appears in train, validation, and test, so this extra output has no positive examples in that setting. New multi-label vocabularies and output layers must therefore omit it.

This decision does not reinterpret existing artifacts. For every legacy experiment selected for retention, preserve its serialized label encoder, class order, output dimension, and `<UNK>` behavior. If a future sequence or ingestion workflow needs an unknown-token mechanism, specify and test it separately rather than carrying `<UNK>` into the multi-label output by default.

The implementation must add regression tests for the new multi-label encoder behavior and must not remove `<UNK>` globally from legacy configurations or checkpoints.

## D9–D10: evaluation and selection

Macro mean average precision exposes performance across ingredient labels, including weaker labels. Micro F1 summarizes pooled decisions and remains easier to compare with historical experiments. Headline comparisons report both.

Also report per-label support and per-label metrics, with explicit treatment of labels that have insufficient evaluation positives. Thresholds, calibration, early stopping, hyperparameters, target rules, and ingredient selection must use training and validation data only. The frozen test split is used for final comparison.

## D11: comparative training and vocabulary-reduction methodology

The project uses one shared, versioned selected vocabulary rather than a
different learned vocabulary per model category. Subphase 4B has frozen the
justified reference selector under D12, while Subphase 4A independently chooses
the experiment model categories; Macro-section 3 then owns the selection
workflow and produces the shared projection. Macro-section 6 tunes every model
category on the full v5 task, uses its unchanged full-task configuration for the
selected-vocabulary ablation, and runs support-matched random-vocabulary
controls with the reference selector. A small selected-task adaptation panel,
if used, is reported separately because it changes hyperparameters as well as
the vocabulary.

The binding design, rationale, formulae, single-run resource limitation, and
ownership boundaries are in
[`model_comparison_methodology.md`](model_comparison_methodology.md). That
document is authoritative for this decision; the active Macro-section 3 plan
owns the selection implementation and the later Macro-sections 6 and 7 own
training and final-comparison execution.

## D12: frozen reference-selector boundary

The reference selector is Torchvision EfficientNetV2-S initialized with the
exact `EfficientNet_V2_S_Weights.IMAGENET1K_V1` artifact and trained end to end
from the first step. It consumes the frozen ordered 165-label `v5` task,
preserves the complete image through the declared 384-pixel aspect-preserving
fit/pad transform, and emits independent raw logits through the stock global
average pool, stock classifier dropout, and a new biased `Linear(1280, 165)`
layer. The loss family is mean-reduced `BCEWithLogitsLoss` with per-label
`pos_weight` computed from train positives only.

The exact artifact hash, resize/padding arithmetic, augmentation boundary,
head initialization, FP32 physical-batch-8 execution boundary, interpretation,
and Phase 3 provenance requirements are authoritative in
[`model_comparison_methodology.md`](model_comparison_methodology.md#4b-d1--frozen-reference-selector-protocol).
This decision fixes a measurement instrument, not the final benchmark winner.
It neither proves that an ingredient is directly visible nor establishes that
EfficientNetV2-S is the best-performing `v5` model.

The repository does not yet implement this complete protocol. Phase 3-D1 now
freezes the complementary campaign under D13; P2 must implement the wrapper,
transforms, AP logging, manifest, and validation. Any change to D12 after label
outcomes are inspected is a new versioned methodology decision, not an
implementation convenience.

## D13: frozen reference-selector campaign boundary

The selector campaign uses exactly one primary configuration and one declared
seed. It runs the 4B-D1 model for 20 complete epochs with physical batch 8,
train-only positive-weighted BCE, AdamW at `1e-4`, a two-epoch linear warm-up,
and cosine decay to `1e-6`. There is no early stopping, scheduler reaction to
validation, repeated seed, or additional training-time robustness
configuration.

Full deterministic train and validation audits occur before optimization and
after every second epoch. Per-label AP is the primary trajectory measurement;
F1 uses one global fixed `0.5` threshold and remains diagnostic. A deterministic
24-label support-stratified cohort is the only outcome visible while P3 freezes
the numerical gates. P4 then applies the frozen rule to the remaining labels
from the same sealed run, avoiding a second full selector training.

The exact optimizer arguments, schedule, randomness controls, audit epochs,
trajectory windows, bootstrap policy, non-visual controls, pilot construction,
and output manifest are authoritative in
[`model_comparison_methodology.md`](model_comparison_methodology.md#phase-3-d1--frozen-selector-campaign-and-measurement-protocol).
P2 must pass the complete instrumented resource and reproducibility gates before
the run is accepted. Any campaign-side change after non-pilot label outcomes are
visible creates a new versioned methodology decision.

## Automatic readiness checklist

A new metadata generation is ready only when:

- [ ] every record has a unique identifier and appears in one split;
- [ ] every referenced image exists under `imgs/standard` and decodes;
- [ ] `ingredients` is preserved and every retained record has a valid `ingredients_target` list;
- [ ] target generation and ordering are deterministic;
- [ ] no exact SHA-256 image group crosses splits;
- [ ] split ratios and cuisine/target distribution tolerances pass;
- [ ] the same metadata filename exists in train, validation, and test;
- [ ] a clean rerun produces identical assignments and metadata content;
- [ ] the DataModule loads every split with `feature_label="ingredients_target"`;
- [ ] selected retained legacy experiments load without any saved-file modification;
- [ ] `<UNK>` removal from new multi-label outputs is implemented and regression-tested before new encoder semantics are frozen.

## Superseded proposals

Earlier planning proposed per-line ingredient mappings, manual image reviews, perceptual and semantic recipe families, a separate split manifest, a dataset vocabulary file, and a persistent validation report. These proposals are superseded. They duplicated metadata, required subjective review, or introduced grouping bias without a demonstrated runtime consumer.

## Evidence and related documents

- [`yummly_data_audit.md`](yummly_data_audit.md)
- [`ingredient_vocabulary_audit.md`](ingredient_vocabulary_audit.md)
- [`../implementation_details/ingredient_mapping_rules.md`](../implementation_details/ingredient_mapping_rules.md)
- [`problem_definition.md`](problem_definition.md)
- [`model_comparison_methodology.md`](model_comparison_methodology.md)
- [`../plans/data_ingredient_refactor/yummly_data_phase.md`](../plans/data_ingredient_refactor/yummly_data_phase.md)
- [`../plans/recognizable_ingredient_selection.md`](../plans/recognizable_ingredient_selection.md)
- [`../general_plan.md`](../general_plan.md)
- [`../research/topics/dataset_splitting/split_strategy.md`](../research/topics/dataset_splitting/split_strategy.md)
- [`../technical_details/data/yummly_benchmark_split/explaination.md`](../technical_details/data/yummly_benchmark_split/explaination.md)
- [`../../src_scratches/data_anlysis/README.md`](../../src_scratches/data_anlysis/README.md)
