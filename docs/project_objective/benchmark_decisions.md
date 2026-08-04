# Yummly data and benchmark decisions

**Created:** 2026-08-02  
**Last updated:** 2026-08-04
**Status:** Active and binding

## Purpose

This document records the decisions that govern the Yummly data pipeline and the ingredient-recognition benchmark. The detailed evidence remains in [`yummly_data_audit.md`](yummly_data_audit.md), while implementation order and acceptance checks remain in [`../plans/yummly_data_phase.md`](../plans/yummly_data_phase.md).

The existing 65,146-record metadata remains valid for historical experiments. It is a legacy artifact, not the target representation for new experiments. New generations use a simpler pipeline that avoids duplicated manifests and subjective review stages.

## Decision summary

| ID | Question | Binding decision |
| --- | --- | --- |
| D1 | Which target field do models use? | Keep `feature_label` configurable. Its default for new configurations becomes `ingredients_target`; historical configurations retain their explicit `ingredients_ok`. |
| D2 | How is `ingredients_target` produced? | Derive it deterministically from the original `ingredients` lines with a tested standardizer. Choose fine-grained distinctions primarily by practical recognizability in the prepared-dish image, and preserve `ingredients` unchanged. |
| D3 | Which persistent artifacts are required? | Store the common images and one selected metadata file per split. Do not create separate mapping, review, family, split, vocabulary, or validation-report artifacts. |
| D4 | How is image quality handled? | Apply automatic existence and decoding checks. Do not add a manual image-review or adjudication workflow; models must tolerate remaining noise. |
| D5 | How are leakage groups and splits built? | Group byte-identical images by SHA-256 only, then create a deterministic 80/10/10 split balanced for cuisine and ingredient targets. Do not use fuzzy recipe families. |
| D6 | How is the vocabulary represented? | Derive it deterministically from training metadata and save its class order with each experiment or checkpoint. Do not maintain a separate dataset-level vocabulary file. |
| D7 | How are historical experiments kept compatible? | Do not rewrite their metadata, configurations, or checkpoints. Compatibility work is deferred until the historical experiments to retain are selected; then adapt their paths and known schemas in memory when loading. |
| D8 | What happens to `<UNK>`? | Remove it from new multi-label vocabularies and outputs because it has no positive training target. Preserve saved behavior for any legacy experiment selected for retention. |
| D9 | Which primary metrics are used? | Report macro mean average precision and micro F1 together; neither is sufficient alone. |
| D10 | Where are thresholds and calibration selected? | Fit thresholds, calibration, and other selection-time parameters on validation data only. Keep the test split unavailable to selection decisions. |

## D1: target-field contract

The DataModule continues to accept a configurable `feature_label`. This is required for alternative targets and for compatibility with historical experiments.

- New configurations default to `ingredients_target`.
- Existing metadata files keep `ingredients_ok` unchanged.
- Existing configurations that explicitly select `ingredients_ok` must continue to select it.
- New metadata generations store `ingredients_target` and preserve the original `ingredients` field.
- A legacy checkpoint must never be paired silently with newly standardized targets because its saved class indices and output semantics refer to `ingredients_ok`.

## D2: improved target standardization

The new standardizer starts from the original ingredient lines, not from `ingredients_ok`.

The most probable historical lineage is [`../../prev_attempts/attempt1/preprocessing_v2.py`](../../prev_attempts/attempt1/preprocessing_v2.py): it produces flat string labels and its split sizes match the historical files. [`../../prev_attempts/attempt2/pre_process.py`](../../prev_attempts/attempt2/pre_process.py) produces nested category/subcategory pairs and was not the generator of the current representation.

Attempt 1 is evidence, not an executable dependency. The new implementation must correct its known defects:

- support is counted across distinct recipes, not distinct raw strings;
- text operations respect token or phrase boundaries;
- aliases and generalizations are explicit and deterministic;
- unordered similarity merging is removed;
- target lists have deterministic, duplicate-free ordering;
- confirmed legacy collisions are regression-tested.

Fine-grained target distinctions are governed primarily by practical recognizability from the prepared-dish image. Within one meaningful ingredient family, preparation or product-style variants that are not realistically separable are collapsed; robust visual distinctions may remain separate. This rule does not collapse different source-ingredient families automatically merely because both can become visually subtle after cooking.

The approved 2.2b scope includes the conservative alias cleanup, removal of generic `sauce`, expansion of `salt and pepper`, and recognizability-based merges for chicken broth/stock, soy-sauce styles, yogurt variants, toasted sesame oil, and tomato paste/sauce. Ground/powder spices merge into the base while seeds/leaves/sticks remain separate; white/yellow/Spanish/sweet onion collapse into `onion` while red and green onion remain; sugars collapse into `sugar`; and coriander/cilantro is retained as one canonical target. Bare red/green pepper maps to the corresponding colour-specific bell pepper. Generic chili powder and ground/crushed/dried/flaked red-pepper forms collapse into `chili`. Fresh tomato and red/green bell-pepper distinctions remain separate. The decision gate is complete; [`../implementation_details/ingredient_mapping_rules.md`](../implementation_details/ingredient_mapping_rules.md) is the durable source of truth for the complete mapping contract.

The source-support threshold remains 500 recipes and the minimum retained target count remains three for the first comparison build. No per-line mapping table or manual mapping review is required.

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
3. use a fixed seed and deterministic ordering;
4. target an 80/10/10 record allocation;
5. balance cuisine and `ingredients_target` marginals within documented tolerances;
6. assert that no record or exact-image group crosses splits;
7. produce identical metadata content for identical input, configuration, and seed.

Perceptual hashes, recipe names, ingredient similarity, and manual decisions must not create allocation groups. They are too dependent on arbitrary thresholds or interpretation for the chosen project scope.

## D6: vocabulary ownership

The output vocabulary is derived from training metadata after the split. Its exact ordered classes must be serialized with the experiment configuration or checkpoint so predictions remain interpretable and reloadable.

Validation and test must never expand or reorder that vocabulary. A separate `vocabulary.json` beside the dataset is unnecessary because it would duplicate experiment state and could drift from the selected `feature_label` or cuisine filter.

## D7: immutable legacy artifacts

The repository contains multiple historical storage generations: JSON-driven experiments, light checkpoints that depend on nearby JSON configuration, current-style full checkpoints, and older DenseNet checkpoints with different DataModule key names.

Compatibility, when resumed for selected retained experiments, is implemented during loading:

- supply `images_subdir="imgs/standard"` when an older configuration lacks it;
- preserve explicit `feature_label="ingredients_ok"`;
- translate known older DenseNet keys into current DataModule arguments in memory;
- retain saved label encoders, class order, output dimensions, model state, and `<UNK>` behavior;
- fail clearly on unknown schemas.

The legacy `metadata.json` and `sel_ing_2410_metadata.json` files, saved JSON configurations, YAML files, and checkpoints are not rewritten. A read-only validation script must prove that the selected representative experiments still load after the image move. This work is deferred until those experiments are chosen.

## D8: `<UNK>` is removed from new multi-label outputs

The current multi-label encoder appends `<UNK>`. Across the current full Yummly data, every observed target label already appears in train, validation, and test, so this extra output has no positive examples in that setting. New multi-label vocabularies and output layers must therefore omit it.

This decision does not reinterpret existing artifacts. For every legacy experiment selected for retention, preserve its serialized label encoder, class order, output dimension, and `<UNK>` behavior. If a future sequence or ingestion workflow needs an unknown-token mechanism, specify and test it separately rather than carrying `<UNK>` into the multi-label output by default.

The implementation must add regression tests for the new multi-label encoder behavior and must not remove `<UNK>` globally from legacy configurations or checkpoints.

## D9–D10: evaluation and selection

Macro mean average precision exposes performance across ingredient labels, including weaker labels. Micro F1 summarizes pooled decisions and remains easier to compare with historical experiments. Headline comparisons report both.

Also report per-label support and per-label metrics, with explicit treatment of labels that have insufficient evaluation positives. Thresholds, calibration, early stopping, hyperparameters, target rules, and ingredient selection must use training and validation data only. The frozen test split is used for final comparison.

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
- [`../plans/yummly_data_phase.md`](../plans/yummly_data_phase.md)
- [`../general_plan.md`](../general_plan.md)
- [`../../src_scratches/data_anlysis/README.md`](../../src_scratches/data_anlysis/README.md)
