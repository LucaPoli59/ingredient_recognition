# Project objective

**Created:** 2026-08-02  
**Last updated:** 2026-09-24

This directory contains the documents that formalize the research problem addressed by the Ingredient Recognition project. These documents establish the context and boundaries that guide topic research, discovery work, technical decisions, and evaluation.

```text
project_objective/
├── README.md
├── benchmark_decisions.md
├── experimental_model_portfolio.md
├── ingredient_vocabulary_audit.md
├── model_comparison_methodology.md
├── problem_definition.md
└── yummly_data_audit.md
```

Documents in this directory should define, as applicable:

- the problem statement and its motivation;
- the primary research objective and supporting questions;
- the intended inputs, outputs, and users;
- the scope and explicit non-goals;
- assumptions, constraints, and dependencies;
- measurable success criteria and evaluation principles;
- unresolved questions that require research or validation.

Keep each file focused on a clearly identified aspect of the objective. Update this README when files are added, moved, or renamed so that it remains an accurate index of the directory.

## Current documents

- [`experimental_model_portfolio.md`](experimental_model_portfolio.md) owns the adopted 4A established pair (4A-D1) and custom P2-S query/context topology (4A-D2), their protocols/fallbacks, rationale and implementation gates; all three are selected designs, not verified implementations.
- [`problem_definition.md`](problem_definition.md) defines the research problem, scope, research questions, evaluation principles, and completion gates.
- [`yummly_data_audit.md`](yummly_data_audit.md) documents the processing lineage, schema, distributions, quality defects, leakage, and implications of the Yummly data used by the project.
- [`ingredient_vocabulary_audit.md`](ingredient_vocabulary_audit.md) audits the 209-target candidate generation, quantifies fragmentation and semantic collisions, and defines the discussion gate before extractor changes.
- [`benchmark_decisions.md`](benchmark_decisions.md) defines the target-field contract, deterministic target generation, minimal outputs, automatic image checks, exact-duplicate split policy, legacy compatibility, evaluation rules, `<UNK>` removal, and the frozen model- and campaign-side reference-selector boundaries.
- [`model_comparison_methodology.md`](model_comparison_methodology.md) defines the binding research design that separates Subphase 4A experiment-model selection from Subphase 4B reference-selector selection, freezes both the exact EfficientNetV2-S model-side protocol and the Phase 3 campaign/measurement protocol, and governs the shared selected vocabulary, fair model-category comparisons, support-matched random vocabulary controls, transferred-hyperparameter ablations, and optional local adaptation.

Read the data audit first, then the problem definition, benchmark decisions, comparative methodology, and candidate vocabulary audit. Discovery and model research must use these documents as the current statement of project scope. Existing results on the 182-label `ingredients_ok` split remain valid historical experiments, while new comparative claims use a deterministic `ingredients_target` generation after the readiness checklist passes.

Project progress and the ordered research plan are maintained in [`../general_plan.md`](../general_plan.md). Detailed execution plans for concrete implementations belong in [`../plans/`](../plans/README.md); the Yummly-specific controlled-vocabulary evaluation is kept with its active plan in [`../plans/data_ingredient_refactor/`](../plans/data_ingredient_refactor/README.md).
