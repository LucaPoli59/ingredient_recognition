# Ingredient-selection inclusion-policy audit

**Created:** 2026-10-04
**Last updated:** 2026-10-04

Read-only, standard-library audit of the completed `phase3-d1-v3` campaign.
It verifies the frozen rule and campaign hashes, exact 165-class order, full
21-audit train/validation metric inventory, and an independent replay of all
D4 outcomes. It then reports overlapping gate failures, nested counterfactual
policies, numerical floor sensitivity, and support/uncertainty diagnostics.

From the main WSL repository root:

```bash
/usr/bin/python3 src_scratches/ingredient_selection_policy_audit/audit_inclusion.py \
  analysis_outputs/ingredient_selection/phase3-d1-v3
```

The command writes deterministic JSON to standard output and changes no file.
It imports neither Torch nor the training package, reads no images or test data,
and runs no training. Input hashes are embedded in the output. Assertions bind
this audit to the reviewed v3 cohort; do not run Python with `-O`.

All policies are **post-P4 diagnostic counterfactuals**, not a frozen vocabulary.
The held-out screens retain the existing late-window AP and epoch-40 bootstrap
lower bound. That interval is not an interval for the late-window statistic.
The normalized excess-AP values and baseline comparisons are illustrative;
they do not introduce an adopted threshold or a matched bootstrap difference.
See the [reviewed P4 record](../../docs/experiment_results/phase3_d1_v3_full_profile.md)
and [binding methodology](../../docs/project_objective/model_comparison_methodology.md)
for interpretation and decision ownership.
