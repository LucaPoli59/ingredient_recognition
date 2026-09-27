# Ingredient-selection workflow

**Created:** 2026-09-24
**Last updated:** 2026-09-24

## Purpose and scope

This document describes the maintained implementation of the Phase 3
ingredient-learnability campaign. The binding scientific choices remain owned
by [4B-D1 and Phase 3-D1](../project_objective/model_comparison_methodology.md#4b-d1--frozen-reference-selector-protocol),
while the operational sequence and current status remain owned by the
[ingredient-selection plan](../plans/recognizable_ingredient_selection.md).

The implementation prepares the selector campaign and its deterministic
analysis. It does not contain a `v5` label outcome, numerical P3 profile gates,
or a selected vocabulary.

## Maintained components

| Component | Current responsibility |
| --- | --- |
| `src/models/efficientnet.py` | Builds the exact pretrained EfficientNetV2-S adapter, preserves stock pooling/dropout, installs the biased 165-logit head, and asserts full trainability. |
| `src/data_processing/transformations.py` | Implements RGB conversion, exact long-side-384 round-half-up resize, ImageNet-mean center padding, train-only horizontal flip, and ImageNet normalization. |
| `src/ingredient_selection/data.py` | Loads only the frozen train and validation metadata, establishes the ordered train class space, encodes targets, and provides shuffled training plus ordered deterministic audit loaders. It has no test-split path. |
| `src/ingredient_selection/protocol.py` | Owns frozen constants, deterministic runtime setup, hashes, train-only positive weights, the support-stratified pilot rule, and the clean-worktree gate. |
| `src/training/ingredient_selection.py` | Owns weighted BCE, the one-group AdamW optimizer, linear-warm-up/cosine scheduler, fixed-state audit callback, and real-data CUDA resource gate. |
| `src/ingredient_selection/metrics.py` | Computes per-label AP, fixed-0.5 precision/recall/F1, micro F1, trajectory-window summaries, deterministic AP bootstrap intervals, and application of later P3 gates. |
| `src/ingredient_selection/artifacts.py` | Writes the manifest, cohort, tidy metrics, compressed validation scores, and bootstrap output atomically while rejecting duplicate audit keys. |
| `src/ingredient_selection/analysis.py` | Validates provenance and audit cadence, derives controls and profile evidence, exposes only the 24-label pilot before a hashed rule exists, and writes `validation_summary.json`. |
| `scripts/ingredient_selection/` | Provides thin campaign, analysis, and historical-reproduction commands. |

## Campaign execution boundary

Run commands from the WSL repository with the `wsl_image_pytorch` interpreter:

```bash
/root/miniconda3/envs/wsl_image_pytorch/bin/python \
  scripts/ingredient_selection/run_campaign.py --resource-gate-only

/root/miniconda3/envs/wsl_image_pytorch/bin/python \
  scripts/ingredient_selection/run_campaign.py

/root/miniconda3/envs/wsl_image_pytorch/bin/python \
  scripts/ingredient_selection/analyze_campaign.py \
  analysis_outputs/ingredient_selection/phase3-d1-v1
```

The resource gate is mandatory and is bound to the protocol ID, ordered-class
hash, train/validation metadata hashes, and exact pretrained-weight identity.
The full launcher then requires a clean tracked worktree. It writes
`pilot_cohort.json` before constructing the campaign model, verifies the gate,
and records the initial head hash and every frozen configuration field in
`campaign_manifest.json`.

The training loader shuffles records without replacement using seed 42. Audit
loaders are ordered, use the validation transform, and run the module in
evaluation mode. Full train and validation audits occur at epochs
`0,2,...,20`; train AP therefore never aggregates predictions emitted while
weights are changing. Only validation record IDs, targets, and logits are
persisted at each audit point.

Before P3 creates a valid `profile_rule.json`, the analysis command writes only
the deterministic 24-label pilot to `profile_evidence.csv`. A full analysis is
unlocked only when the rule's own hash, campaign identity hash, and pilot hash
all match. Neither path opens test metadata.

## Historical reproduction

`scripts/ingredient_selection/reproduce_historical.py` delegates to the
maintained read-only retention audit. It regenerates the four 46-label Q3 sets
and their exact 40-label intersection, verifies the three projected metadata
hashes and the executable checkpoint anchors, and reports the historical
weighting/augmentation discrepancies.

The two Optuna journals are declared append-only artifacts. A journal may grow
only when its entire originally hashed byte prefix remains unchanged; modifying
that prefix or any other retained artifact still fails verification. This
preserves the historical evidence while allowing later trials to append to the
shared journal.

## Verified resource and test evidence

On 2026-09-24 the real-data gate used an RTX 4060 with 8,187.375 MiB reported
memory and a physical `[8,3,384,384]` batch. The exact forward, weighted BCE
backward, and AdamW step passed in true FP32 with 3,632.20 MiB peak allocated
and 5,362.00 MiB peak reserved. This is a feasibility result, not throughput or
accuracy evidence.

The repository suite passes 64 tests after the integration. The Phase 3 tests
cover resize/padding/RGB behavior, head construction and trainability,
positive-weight arithmetic, blind cohort reproduction and tamper rejection,
AP/F1 separation, bootstrap determinism, trajectory edge cases, profile
assignment, the complete 20-epoch learning-rate sequence, optimizer/scheduler
resume, eval-mode auditing, test-split isolation, rule-gated label exposure,
and append-only historical retention.

## Limitations and next action

- No selector campaign outcome has been inspected; P3 remains responsible for
  running the sealed campaign and freezing numerical profile gates from only
  the 24-label pilot.
- The single campaign does not estimate seed or configuration stability.
- Bootstrap intervals describe validation-record uncertainty only.
- The cuisine prior is a mechanism diagnostic, not an image-model competitor.
- Checkpoints and generated outputs remain outside durable documentation.

P3 must rerun the resource gate from the clean committed P2 revision so that
the campaign is bound to an immutable code and data identity, then execute the
full 20-epoch campaign.
