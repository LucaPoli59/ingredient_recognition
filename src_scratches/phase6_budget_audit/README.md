# Phase 6 retained-cost audit

**Created:** 2026-10-13
**Last updated:** 2026-10-13

Read-only reproducibility helper for the [reviewed budget baseline](../../docs/experiment_results/phase6_budget_baseline.md). This is not a training launcher, new qualification stage or benchmark.

`audit.py` checks the historical journal and all 200 trial configurations against the retained comparison inventory, aggregates recorded durations, and reads six Phase 5 smoke results plus the completed selector's timestamps. It saves exact input hashes in its output. It does not replay the journal, read predictions, instantiate models or access dataset images.

From the repository root:

```bash
python src_scratches/phase6_budget_audit/audit.py
```

For a new JSON output, use `--output NEW_PATH`; its parent must exist and existing files are rejected. The checked-in `summary.json` is the reviewed 2026-10-13 output, not a live budget contract. The helper uses the standard library and expects the retained artifacts in their canonical locations; it deliberately rejects changed historical evidence rather than silently updating the cohort.

The last-two-update extrapolation is explicit sensitivity arithmetic, not a measured epoch rate, lower/upper bound or campaign ETA. Adopted scientific choices remain in the [methodology](../../docs/project_objective/model_comparison_methodology.md); execution and unresolved resource choices remain in the [Phase 6 plan](../../docs/plans/training_hyperparameter_tuning.md).
