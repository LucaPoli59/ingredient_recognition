# Recognizable ingredient selection plan

**Created:** 2026-08-10
**Last updated:** 2026-08-10

This plan is the operational source of truth for Macro-section 3, **Ingredient selection**, in [`general_plan.md`](../general_plan.md). It preserves the November 2024 ResNet selection as a historical baseline and replaces its exploratory workflow with a reproducible selection study over the frozen FoodOn-first `v5` vocabulary.

## Progress tracker

**Overall status:** In progress
**Current task:** Define the `v5` experimental contract and pilot the learning-dynamics criteria.
**Next action:** Freeze the ResNet configuration panel, seeds, budgets, logged per-label statistics, candidate trajectory criteria, controls, and output manifest before implementing or launching new experiments.

| # | Task | Status | Evidence or result |
| --- | --- | --- | --- |
| P0 | Reconstruct the historical 2024 selection process and resolve its discrepancies | **Done** | Saved configurations, metrics, metadata, checkpoints, notebooks, launchers, journals, and the external communication archive establish the four historical stages and the exact 40-label rule. The accepted resolutions are recorded in this plan. |
| P1 | Freeze the new selection question and experimental contract | **In progress** | Train F1 is accepted as a convergence signal, but the final rule must also measure trajectory, stability, support, and evidence beyond a transient maximum. |
| P2 | Implement deterministic historical reproduction and reusable analysis | **Pending** | Create a dedicated source package and command-line entry points; notebooks become optional views, not execution state. |
| P3 | Run and validate a bounded `v5` pilot | **Pending** | Compare candidate learning-dynamics criteria and controls without test access; use the pilot to freeze the final rule. |
| P4 | Run the full `v5` ResNet learnability campaign | **Pending** | Execute the frozen configuration panel across the required seeds and preserve a complete provenance manifest. |
| P5 | Combine learnability with relevance and visual-observability evidence | **Pending** | Apply the semantic, support, and annotation protocol; distinguish direct visual evidence from contextual predictability. |
| P6 | Freeze named headline and exploratory ingredient tiers | **Pending** | Publish versioned projections of the shared `v5` vocabulary, with explicit inclusion evidence and uncertainty. |
| P7 | Integrate the workflow and retire superseded scripts safely | **Pending** | Connect training and analysis to canonical APIs, verify parity, document current behavior, then clean legacy notebooks and launchers only after retention gates pass. |

## Objective

Determine which ingredients in the standard `ingredients_target_v5_metadata.json` vocabulary provide a meaningful and reproducible learning target for image-based models. The workflow must identify labels whose training signal is actually learned, separate that evidence from validation generalization and human visual observability, and produce named experimental projections without creating a second implicit default vocabulary.

The result is not a claim that every retained ingredient is literally visible. A label may be directly visible, inferable from dish context, or learnable mainly through dataset priors. Those cases must remain distinguishable in the evidence and final tiers.

## Scope

- Reproduce the historical 40-label selection exactly from retained aggregate artifacts as a read-only regression baseline.
- Design and implement a reusable, configuration-driven per-label training analysis for the 165-label `v5` target space.
- Use per-label train F1 as an optimization-learnability signal and retain validation metrics as separate generalization evidence.
- Evaluate learning trajectories, absolute signal, support effects, and stability across declared seeds and a small fixed ResNet configuration panel.
- Add controls that can test whether label-space reduction, visual input, or prevalence explains an observed result.
- Generate deterministic tables, plots, manifests, and versioned ingredient-tier definitions.
- Integrate the workflow with the canonical APIs under `src/training/` instead of creating another training pipeline.

## Non-goals

- Reusing the historical 40-label list as the selected `v5` vocabulary.
- Treating train F1 as evidence of generalization or literal visual observability.
- Selecting model hyperparameters, thresholds, or ingredients from the test split.
- Training one independently tuned model per ingredient by default.
- Making weighted loss, augmentation, or any other ablation a selection requirement before the pilot justifies it.
- Rewriting legacy metadata, configurations, checkpoints, or exported results.
- Deleting historical experiments or scripts before the retention manifest, compatibility smoke tests, and replacement parity checks are complete.
- Replacing the shared `ingredients_target` default. Any smaller vocabulary is an explicitly named projection for a declared experiment or evaluation tier.

## Historical baseline

The 2024 workflow used four experimental stages:

1. tune ResNet-like configurations on the 183-output legacy `ingredients_ok` vocabulary using validation loss;
2. train four 40-epoch runs with per-label F1, take each label's maximum **train F1**, retain the top quartile within each run, and intersect the four 46-label sets;
3. project legacy metadata onto the resulting 40 labels and tune again;
4. validate three selected-vocabulary checkpoints and produce an exploratory old/new comparison.

The exact 40-label intersection is reproducible from `experiments/basic/resnets_training_BM_F1_INGS/full_f1_train.csv` and the saved encoder classes. It remains a regression fixture and historical comparison only. The new `v5` vocabulary, split, target semantics, and output contract differ, so the selection must be rerun.

The minimum retained historical artifacts and the compatibility anchors are owned by [Data Work package 2.1c](data_ingredient_refactor/yummly_data_phase.md#work-package-21c--historical-experiment-compatibility).

## Accepted discrepancy resolutions

| Historical discrepancy | Resolution for the maintained workflow |
| --- | --- |
| Selection used train F1 rather than validation F1 | **Accepted as intentional historical behavior.** Train F1 is a valid signal for the narrower question “does optimization begin to learn this label?”. It must be named as such and kept separate from validation generalization and observability evidence. |
| The rule used only the maximum F1 relative to other labels | **Improve, do not reproduce as the final criterion.** Preserve the max-Q3 intersection as a historical baseline, then pilot trajectory-aware and stability-aware criteria before freezing the `v5` rule. |
| The intended weighted-loss control saved `weighted_loss: false` | **Treat as a nonessential historical launcher defect.** Run 3 is an unweighted stochastic replica. Do not infer whether weighting helped, and do not reproduce the defective control. |
| Exported `image_augmentation` values were inverted | **Treat as a reporting defect.** Future reports derive the actual transform and boolean state from the saved run configuration and validate them against the manifest. |
| The old/new plot mixed mean historical train F1 with final new validation F1 | **Retire as comparative evidence.** Future comparisons use the same split, cohort, metric definition, epoch aggregation, configuration-selection rule, and seed aggregation. |
| The exact 2024 source snapshot is absent from Git | **Accept the forensic evidence hierarchy.** Saved configurations, checkpoints, metrics, metadata, journals, and exports are primary evidence for executed behavior. Every future run records the code revision, environment, data hashes, configuration, and seeds. |

## Selection framework to pilot

### Evidence dimensions

The final inclusion decision must not collapse these dimensions into one unexplained score:

1. **Optimization learnability:** whether per-label train F1 rises meaningfully from its early-run baseline and remains elevated.
2. **Generalization:** whether validation ranking and set metrics support the learned signal without selecting on test.
3. **Stability:** whether the conclusion persists across seeds, the fixed configuration panel, and nearby epoch windows.
4. **Support and imbalance:** whether the signal is explained primarily by prevalence, cardinality, or the loss configuration.
5. **Visual evidence type:** whether audited examples are `direct`, `contextual`, `not_inferable`, or `uncertain`.
6. **Research relevance:** whether the label helps answer the thesis question and has enough evaluation support for its intended tier.

### Candidate trajectory statistics

P1 and P3 will evaluate, rather than prematurely fix, a compact set of robust statistics:

- early-to-late change in train F1;
- a robust late-window level, such as a median or upper quantile, instead of a single-epoch maximum;
- positive slope or area above the early baseline over a declared epoch window;
- time to a declared absolute F1 floor, when the floor is meaningful for the support tier;
- cross-seed and cross-configuration pass rate;
- uncertainty or dispersion around every aggregated statistic.

The pilot must reject criteria that merely guarantee a fixed quota, are dominated by a one-epoch spike, or change substantially under a nearby reasonable epoch window. A final rule may use more than one statistic and may define an `uncertain` band instead of forcing every label into a binary decision.

### Controls

The protocol must include the least expensive controls that answer the relevant causal questions:

- label prevalence and support correlations for every reported statistic;
- an untrained or early-epoch reference for learning-delta calculations;
- a non-visual prevalence baseline, and a cuisine-prior diagnostic where appropriate;
- at least one matched-size random or support-matched vocabulary projection when claiming that vocabulary reduction improves training or validation behavior;
- identical-metric full-vocabulary versus selected-projection comparisons when assessing the effect of label-space reduction.

A shuffled-label control is optional at pilot time and becomes mandatory only if the cheaper controls cannot distinguish optimization artifacts from a learned signal.

## Experimental contract to freeze in P1

1. Use the frozen `v5` train and validation metadata and their saved class order; do not access test outcomes.
2. Select ResNet configurations globally using validation objectives, never a different configuration chosen after inspecting each label.
3. Use one primary configuration and a bounded robustness panel; run the final campaign with at least three declared seeds per required configuration.
4. Keep batch sampling, epoch budget, transforms, loss, threshold, logging cadence, and early-stopping behavior explicit and comparable.
5. Log per-label train and validation metrics at every declared evaluation point, including support and class index.
6. Persist the exact model, optimizer, scheduler, loss-weighting state, transform identity, seed, code revision, environment, metadata hashes, and encoder classes in a run manifest.
7. Validate every report column against saved configurations; do not relabel booleans manually in analysis code.
8. Keep model and hyperparameter selection on validation. Train F1 is used only for the predeclared optimization-learnability analysis.

## Planned architecture and artifacts

The exact module names are frozen during P1, but the implementation boundary is fixed now:

- `src/ingredient_selection/` will contain reusable ingestion, trajectory metrics, selection rules, validation, and plotting data preparation;
- `scripts/ingredient_selection/` will contain thin command-line entry points for historical reproduction, campaign analysis, and report generation;
- canonical training remains under `src/training/`, with only the reusable logging/configuration extensions required by this study;
- notebooks may consume generated tables for exploration, but no selection decision may depend on notebook execution order or hidden state.

Each analysis execution will create one versioned report directory containing:

- an input manifest with experiment groups, run identifiers, class order, configuration, seeds, code revision, environment, and data hashes;
- a tidy per-label, per-epoch metrics table;
- per-label summary statistics with uncertainty and decision reasons;
- the selected, rejected, and uncertain ingredient sets as named projections of `v5`;
- plots for trajectories, stability, support relationships, and control comparisons;
- a machine-readable validation summary proving schema, class-order, and provenance checks.

Large checkpoints and raw training logs remain in `experiments/`; the report links them rather than copying them.

## Plot requirements

The replacement report must make the selection logic inspectable rather than merely attractive:

- small-multiple or filtered per-label train/validation trajectories with seed dispersion;
- early-versus-late change and robust late-window distributions;
- stability heatmaps across seeds and configurations;
- learnability statistics versus train support and prevalence;
- selected/rejected/uncertain decision plots with thresholds and reasons visible;
- like-for-like full-vocabulary, selected-projection, and matched-control comparisons when reduction claims are made.

Plots must label the split, statistic, aggregation window, seed count, vocabulary version, and actual transform/loss state. The invalid historical old/new plot is retained only as history and is not regenerated as evidence.

## Validation

### Historical reproduction

- Read retained artifacts without modifying them.
- Regenerate the four 46-label Q3 sets and exact 40-label intersection.
- Assert label names and indices against the saved legacy encoder.
- Verify the current three `sel_ing_2410_metadata.json` hashes before any compatibility smoke test.
- Make the weighted-loss and augmentation discrepancies explicit in the generated historical report.

### New workflow

- Unit-test trajectory statistics on hand-verifiable arrays, including flat, improving, noisy-spike, degrading, and missing-epoch cases.
- Reject inconsistent class order, duplicated run IDs, mixed metadata hashes, missing seeds, or contradictory configuration fields.
- Re-run analysis on the same inputs and require identical machine-readable outputs.
- Verify that changing plot styling cannot change the selected ingredient sets.
- Prove that no test metrics are read by selection commands.
- Compare the frozen rule with the historical max-Q3 baseline and the required controls on the pilot before the full campaign.

## Completion criteria

This plan is complete only when:

- the historical 40-label result is reproduced by maintained read-only code;
- the `v5` selection protocol, controls, seeds, and decision rule are frozen before the full analysis;
- the full campaign produces deterministic reports with complete provenance;
- every selected, rejected, or uncertain label has explicit evidence across learnability, generalization, support, stability, observability, and relevance as applicable;
- headline and exploratory tiers are versioned named projections, not a replacement default vocabulary;
- the test split was not used for ingredient, model, threshold, or hyperparameter selection;
- required compatibility and replacement-parity checks pass before any legacy cleanup;
- the final decisions are promoted to the appropriate objective and implementation documentation.

## Risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Train F1 is interpreted as visual generalization | Frequent or memorized labels are called recognizable | Name it optimization learnability and require separate validation and observability evidence. |
| A relative rule retains labels even when all runs are poor | The selected vocabulary has an arbitrary fixed size | Use absolute, trajectory, stability, and uncertainty criteria; allow an uncertain tier. |
| Per-label tuning overfits the selection process | Each label benefits from a different post-hoc configuration | Freeze one global configuration panel before inspecting selection outcomes. |
| Vocabulary reduction appears beneficial because metrics or cohorts changed | The improvement claim is invalid | Compare identical metrics, records, budgets, and seed aggregation; add matched-size controls. |
| Support dominates F1 trajectories | The study selects only frequent labels | Report support dependence, stratify evidence, and combine it with relevance and observability rather than hiding it. |
| Notebook state or mislabeled configuration fields corrupt the report | The selection cannot be reproduced | Use deterministic source modules, manifests, schema validation, and generated plots. |
| Cleanup removes unique forensic evidence | Historical claims can no longer be verified | Apply the 2.1c retention manifest and hashes before deleting anything. |

## Decision log

| Date | Decision or change | Rationale |
| --- | --- | --- |
| 2026-08-10 | Accepted the reconstructed four-stage 2024 workflow as the historical baseline | Saved experiment artifacts establish the executed logic more reliably than the later-committed notebooks and launchers. |
| 2026-08-10 | Accepted train F1 as the intended convergence signal | The historical question was whether the network began to learn a label; train F1 is suitable for that narrow purpose when it is not presented as generalization. |
| 2026-08-10 | Retained max-Q3 intersection only as a baseline and regression | A transient maximum and relative quota do not measure sustained improvement or an absolute learnability level. |
| 2026-08-10 | Classified the fourth H2 run as an unweighted replica | Its saved configuration has `weighted_loss: false`; the intended control is neither required nor reproduced. |
| 2026-08-10 | Classified the augmentation inversion as a reporting defect | Future analysis derives transform state directly from the saved configuration. |
| 2026-08-10 | Retired the historical old/new plot as comparative evidence | It compares different splits of evidence, epoch aggregation, cohorts, and selection conditions. |
| 2026-08-10 | Required a new selection campaign on `v5` | The 40 legacy labels and their indices do not represent the current 165-label FoodOn-first vocabulary. |
| 2026-08-10 | Made saved run artifacts the primary evidence for 2024 and strengthened future provenance | The exact 2024 code snapshot was never committed; future manifests must remove this ambiguity. |

## Related documentation

- [`general_plan.md`](../general_plan.md)
- [`data_ingredient_refactor/yummly_data_phase.md`](data_ingredient_refactor/yummly_data_phase.md)
- [`../project_objective/problem_definition.md`](../project_objective/problem_definition.md)
- [`../project_objective/benchmark_decisions.md`](../project_objective/benchmark_decisions.md)
- [`../project_objective/ingredient_vocabulary_audit.md`](../project_objective/ingredient_vocabulary_audit.md)
- [`../../README_PROJECT_KNOWLEDGE.md`](../../README_PROJECT_KNOWLEDGE.md)
