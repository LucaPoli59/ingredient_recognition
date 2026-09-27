# Implementation details

**Created:** 2026-08-06  
**Last updated:** 2026-09-27

This directory contains durable documentation of the repository's current implementation contracts. It explains what the code supports, how components integrate, which configuration defaults and invariants are relied on, and where the behavior is verified.

## Scope and boundaries

Use this category for code-facing facts that must remain aligned with the implementation, such as data contracts, model interfaces, configuration behavior, supported variants, persistence formats, and operational constraints. Keep broad literature research in [`../research/`](../research/README.md), architecture-focused explanations in [`../models_deepdive/`](../models_deepdive/README.md), cross-cutting conceptual analyses in [`../technical_details/`](../technical_details/README.md), and execution state in [`../plans/`](../plans/README.md).

Implementation-detail documents describe verified current behavior. They must identify the relevant source files and tests, distinguish legacy behavior from new defaults, and be updated in the same change as the code or contract they document.

## Files

- [`models.md`](models.md) describes the vision-model implementations available under `src/models` and their training-pipeline contracts.
- [`ingredient_mapping_rules.md`](ingredient_mapping_rules.md) is the long-term authority for custom `ingredients` to `ingredients_target` mappings, exclusions, multi-target expansions, retained distinctions, and collision boundaries.
- [`experiment_artifacts.md`](experiment_artifacts.md) records the audited target-v5 experiment artifacts, scalar/histogram semantics, restart and checkpoint-selection behavior, and offline analysis limits.
- [`experiment_comparison.md`](experiment_comparison.md) defines the optional Lightning-model per-ingredient logging contract and the maintained local N-experiment JSON/HTML comparison command.
- [`image_data_loading.md`](image_data_loading.md) defines the image DataModule's platform-aware pinned-memory policy, configuration persistence, and worker-setting boundaries.
- [`ingredient_selection.md`](ingredient_selection.md) defines the maintained Phase 3 selector model/data/training/audit/analysis boundary, blind-pilot gate, rerunnable launcher, effective-batch planning, source-snapshot provenance, historical reproduction, and measured 8 GB capacity.

When a new implementation contract is added, use a focused descriptive filename, add it to this index, and link it from the relevant plan or project-objective document when it changes a tracked decision or completion gate.
