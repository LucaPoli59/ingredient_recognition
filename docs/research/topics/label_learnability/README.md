# Label learnability research

**Created:** 2026-08-12
**Last updated:** 2026-08-12

## Context

This topic records general research on assessing whether an output label can be
learned by a supervised classifier. It applies to multi-label image
classification, but separates findings that are general to supervised learning
from adaptations that need validation in a multi-label setting. It is not a
claim that learnability is an intrinsic or permanent property of a label.

## Scope

The research covers metric choice, training dynamics, class imbalance,
held-out validation, diagnostic controls, and how to express the result without
reducing it to one misleading scalar. It deliberately does not choose a dataset,
model, threshold, numerical cut-off, or production vocabulary.

## Files

- [`learnability_assessment.md`](learnability_assessment.md) synthesizes the
  evidence and proposes a reusable assessment protocol.

## Related work

- [`../../../plans/recognizable_ingredient_selection.md`](../../../plans/recognizable_ingredient_selection.md)
  is the project-specific plan that may adopt a version of this protocol after
  its experimental contract is frozen.
- [`../../../project_objective/problem_definition.md`](../../../project_objective/problem_definition.md)
  defines the weakly supervised multi-label image-prediction setting in which
  the ingredient use case belongs.
- [`../../../project_objective/benchmark_decisions.md`](../../../project_objective/benchmark_decisions.md)
  remains authoritative for the project's frozen evaluation decisions.
