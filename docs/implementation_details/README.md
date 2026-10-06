# Implementation details

**Created:** 2026-08-06  
**Last updated:** 2026-10-06

This directory contains durable documentation of the repository's current implementation contracts. It explains what the code supports, how components integrate, which configuration defaults and invariants are relied on, and where the behavior is verified.

## Scope and boundaries

Use this category for code-facing facts that must remain aligned with the implementation, such as data contracts, model interfaces, configuration behavior, supported variants, persistence formats, and operational constraints. Keep broad literature research in [`../research/`](../research/README.md), architecture-focused explanations in [`../models_deepdive/`](../models_deepdive/README.md), cross-cutting conceptual analyses in [`../technical_details/`](../technical_details/README.md), and execution state in [`../plans/`](../plans/README.md).

Implementation-detail documents describe verified current behavior. They must identify the relevant source files and tests, distinguish legacy behavior from new defaults, and be updated in the same change as the code or contract they document.

## Files

- [`models.md`](models.md) describes the vision-model implementations available under `src/models` and their training-pipeline contracts.
- [`experimental_model_contract.md`](experimental_model_contract.md) defines opt-in Phase 5 preprocessing/identity, initialization, exact batching and engineering policy, implemented EfficientNet/MaxViT adapters, MaxViT artifact and normalization provenance, and strict experimental Lightning/full-light restoration. P2 and measured CUDA/real-consumer qualification are separate gates.
- [`ingredient_mapping_rules.md`](ingredient_mapping_rules.md) is the long-term authority for custom `ingredients` to `ingredients_target` mappings, exclusions, multi-target expansions, retained distinctions, and collision boundaries.
- [`experiment_artifacts.md`](experiment_artifacts.md) records the audited target-v5 experiment artifacts, scalar/histogram semantics, restart and checkpoint-selection behavior, and offline analysis limits.
- [`experiment_comparison.md`](experiment_comparison.md) defines the optional Lightning-model per-ingredient logging contract and the maintained local N-experiment JSON/HTML comparison command.
- [`image_data_loading.md`](image_data_loading.md) defines shared-image and target loading, saved encoder compatibility, platform-aware pinned memory, model-specific dashboard preprocessing, and the rerunnable real CUDA/checkpoint/dashboard smoke that closed Data 2.4.
- [`ingredient_selection.md`](ingredient_selection.md) defines the maintained Phase 3 selector, blind-pilot/D4 analysis, D6 saved-score inclusion review, P6 frozen projection/export and P7 opt-in runtime/checkpoint/analysis contract, parity checks and historical retention dispositions, alongside the rerunnable launcher, effective batching and measured capacity.

When a new implementation contract is added, use a focused descriptive filename, add it to this index, and link it from the relevant plan or project-objective document when it changes a tracked decision or completion gate.
