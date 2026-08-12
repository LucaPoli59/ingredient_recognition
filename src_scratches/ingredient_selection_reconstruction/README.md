# Ingredient-selection reconstruction workspace

**Created:** 2026-08-10

**Last updated:** 2026-08-10

## Purpose

This temporary workspace reconstructs the November 2024 ingredient-selection experiment that predates the current project plans. It is exploratory evidence for defining Macro-section 3, not an authoritative methodology, implementation contract, or feature plan.

**Review status:** Accepted on 2026-08-10; Data 2.1c closed on 2026-08-12. The durable retention decision is now in [`../../docs/plans/data_ingredient_refactor/yummly_data_phase.md`](../../docs/plans/data_ingredient_refactor/yummly_data_phase.md), and the discrepancy resolutions and replacement workflow are owned by [`../../docs/plans/recognizable_ingredient_selection.md`](../../docs/plans/recognizable_ingredient_selection.md).

## Files

- [`historical_logic_reconstruction.md`](historical_logic_reconstruction.md) records the reconstructed experiment stages, exact selection rule, involved artifacts, verified contradictions, reproducibility limits, and implications for the future plan.
- [`retention_manifest.json`](retention_manifest.json) is the generated 72-entry, repository-relative retention manifest used by the read-only compatibility gate.
- [`../../scripts/validate_legacy_experiments.py`](../../scripts/validate_legacy_experiments.py) verifies hashes, historical selection reproduction, metadata/image compatibility, checkpoint anchors, and saved H2 configuration state without rewriting artifacts.
- `working/` is ignored disposable QA material used to render and inspect the external Word source; it is not part of the reconstruction deliverable.

## Working rules

- Keep historical facts separate from interpretations and proposed improvements.
- Treat saved experiment configurations, metrics, and metadata as stronger evidence than notebook prose or stale notebook outputs.
- Do not change or delete legacy experiments, checkpoints, metadata, or the external professor-communication material from this workspace.
- Keep this workspace as forensic support. Update the durable Data and Macro-section 3 plans rather than treating this scratch report as the final project record.

## Related project records

- [`../../docs/general_plan.md`](../../docs/general_plan.md)
- [`../../docs/plans/data_ingredient_refactor/yummly_data_phase.md`](../../docs/plans/data_ingredient_refactor/yummly_data_phase.md)
- [`../../docs/plans/recognizable_ingredient_selection.md`](../../docs/plans/recognizable_ingredient_selection.md)
- [`../../README_PROJECT_KNOWLEDGE.md`](../../README_PROJECT_KNOWLEDGE.md)
