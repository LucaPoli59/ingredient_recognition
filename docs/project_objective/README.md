# Project objective

**Created:** 2026-08-02  
**Last updated:** 2026-08-02

This directory contains the documents that formalize the research problem addressed by the Ingredient Recognition project. These documents establish the context and boundaries that guide topic research, discovery work, technical decisions, and evaluation.

```text
project_objective/
├── README.md
├── benchmark_decisions.md
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

- [`problem_definition.md`](problem_definition.md) defines the research problem, scope, research questions, evaluation principles, and completion gates.
- [`yummly_data_audit.md`](yummly_data_audit.md) documents the processing lineage, schema, distributions, quality defects, leakage, and implications of the Yummly data used by the project.
- [`benchmark_decisions.md`](benchmark_decisions.md) defines the target-field contract, deterministic target generation, minimal outputs, automatic image checks, exact-duplicate split policy, legacy compatibility, evaluation rules, and pending `<UNK>` investigation.

Read the data audit first, then the problem definition and benchmark decisions. Discovery and model research must use all three as the current statement of project scope. Existing results on the 182-label `ingredients_ok` split remain valid historical experiments, while new comparative claims use a deterministic `ingredients_target` generation after the readiness checklist passes.

Project progress and the ordered research plan are maintained in [`../general_plan.md`](../general_plan.md). Detailed execution plans for concrete implementations belong in [`../plans/`](../plans/README.md).
