# Recognizable ingredient selection plan

**Created:** 2026-08-10
**Last updated:** 2026-08-12

This plan is the operational source of truth for Macro-section 3, **Ingredient selection**, in [`general_plan.md`](../general_plan.md). It preserves the November 2024 ResNet selection as a historical baseline and replaces its exploratory workflow with a reproducible, research-informed decision-profile study over the frozen FoodOn-first `v5` vocabulary. Macro-section 3 owns the resulting selected vocabulary, but its new execution is deferred until Macro-section 4 chooses the justified reference selector.

## Progress tracker

**Overall status:** Deferred
**Current task:** Await the Macro-section 4 decision that identifies the reference selector (`M_ref`) and the comparison model categories.
**Next action:** After `M_ref` is approved, freeze its `v5` configuration panel, one declared seed per configuration, budgets, AP trajectories, fixed-policy F1 diagnostics, controls, profile criteria, and output manifest before implementing or launching new experiments.

| # | Task | Status | Evidence or result |
| --- | --- | --- | --- |
| P0 | Reconstruct the historical 2024 selection process and resolve its discrepancies | **Done** | Saved configurations, metrics, metadata, checkpoints, notebooks, launchers, journals, and the external communication archive establish the four historical stages and the exact 40-label rule. The accepted resolutions are recorded in this plan. |
| P1 | Freeze the new selection question and experimental contract | **Deferred** | The plan adopts a profile that separates optimization, held-out generalization, and validity/mechanism evidence. Macro-section 4 must first identify `M_ref`; P1 then freezes its panel and fixed F1 policy before the pilot. P3 will use the bounded pilot to freeze numerical promotion gates. |
| P2 | Implement deterministic historical reproduction and reusable analysis | **Deferred** | Create a dedicated source package and command-line entry points after P1; notebooks become optional views, not execution state. |
| P3 | Run and validate a bounded `v5` pilot | **Deferred** | Compare candidate learning-dynamics criteria and controls without test access; use the pilot to freeze the final rule. |
| P4 | Run the full `v5` reference-selector learnability campaign | **Deferred** | Execute each frozen `M_ref` configuration once with its declared seed, preserve a complete provenance manifest, and report the resulting single-run limitation. |
| P5 | Combine learnability with relevance and visual-observability evidence | **Deferred** | Apply the semantic, support, and annotation protocol; distinguish direct visual evidence from contextual predictability. |
| P6 | Freeze named headline and exploratory ingredient tiers | **Deferred** | Publish versioned projections of the shared `v5` vocabulary, with explicit inclusion evidence and uncertainty. |
| P7 | Integrate the workflow and retire superseded scripts safely | **Deferred** | Connect training and analysis to canonical APIs, verify parity, document current behavior, then clean legacy notebooks and launchers only after retention gates pass. |

## Objective

Determine which ingredients in the standard `ingredients_target_v5_metadata.json` vocabulary provide a meaningful and reproducible learning target for image-based models. After Macro-section 4 selects `M_ref`, the workflow must identify labels whose signal it learns, separate that evidence from validation generalization and human visual observability, and produce named experimental projections without creating a second implicit default vocabulary.

The result is not a claim that every retained ingredient is literally visible. A label may be directly visible, inferable from dish context, or learnable mainly through dataset priors. Those cases must remain distinguishable in the evidence and final tiers.

## Scope

- Reproduce the historical 40-label selection exactly from retained aggregate artifacts as a read-only regression baseline.
- Design and implement a reusable, configuration-driven per-label training analysis for the 165-label `v5` target space using the Macro-section 4-approved `M_ref`.
- Use per-label train AP trajectories as the optimization-learnability signal and validation AP as the primary held-out generalization evidence. Retain fixed-policy F1 only as a secondary diagnostic.
- Evaluate sustained optimization, held-out ranking quality, support effects, within-run temporal stability, configuration sensitivity where the bounded panel is run, and plausible signal mechanism. Repeated seeds for an identical configuration are out of scope; conclusions must not claim seed-level stability.
- Add controls that can test whether label-space reduction, visual input, or prevalence explains an observed result.
- Generate deterministic tables, plots, manifests, and versioned ingredient-tier definitions.
- Integrate the workflow with the canonical APIs under `src/training/` instead of creating another training pipeline.

## Non-goals

- Reusing the historical 40-label list as the selected `v5` vocabulary.
- Treating train F1 as evidence of generalization or literal visual observability.
- Selecting model hyperparameters, thresholds, or ingredients from the test split.
- Training one independently tuned model per ingredient by default.
- Repeating an identical configuration across several random seeds to estimate run-to-run stability. The study has a declared single-seed-per-configuration resource limit.
- Selecting the new vocabulary with an arbitrary historical ResNet before Macro-section 4 reviews and freezes `M_ref`.
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
| Selection used train F1 rather than validation F1 | **Accepted as intentional historical behavior.** Train F1 is a valid signal for the narrower question “does optimization begin to learn this label?”. The new workflow instead uses train AP trajectories for that evidence and retains fixed-policy train F1 only as a secondary continuity diagnostic. Neither is validation generalization or observability evidence. |
| The rule used only the maximum F1 relative to other labels | **Improve, do not reproduce as the final criterion.** Preserve the max-Q3 intersection as a historical baseline, then pilot trajectory-aware and stability-aware criteria before freezing the `v5` rule. |
| The intended weighted-loss control saved `weighted_loss: false` | **Treat as a nonessential historical launcher defect.** Run 3 is an unweighted stochastic replica. Do not infer whether weighting helped, and do not reproduce the defective control. |
| Exported `image_augmentation` values were inverted | **Treat as a reporting defect.** Future reports derive the actual transform and boolean state from the saved run configuration and validate them against the manifest. |
| The old/new plot mixed mean historical train F1 with final new validation F1 | **Retire as comparative evidence.** Future comparisons use the same split, cohort, metric definition, epoch aggregation, and configuration-selection rule; the new study reports its single-run limitation rather than implying seed aggregation. |
| The exact 2024 source snapshot is absent from Git | **Accept the forensic evidence hierarchy.** Saved configurations, checkpoints, metrics, metadata, journals, and exports are primary evidence for executed behavior. Every future run records the code revision, environment, data hashes, configuration, and declared random seed. |

## Adopted decision-profile framework

**Status:** Adopted for Phase 3 planning on 2026-08-12. Its execution is
deferred until Macro-section 4 selects `M_ref`. P1 must then freeze the exact
configuration panel and measurement policy; P3 will freeze numerical promotion
gates from the bounded pilot.

The framework applies the reusable findings in
[`label_learnability/learnability_assessment.md`](../research/topics/label_learnability/learnability_assessment.md)
to this project. It does not treat learnability as an intrinsic label property:
every conclusion is conditional on the frozen `v5` split, declared `M_ref`
protocol, training budget, and annotation regime. The profile replaces the
legacy maximum-train-F1 rule; it does not yet freeze a final ingredient tier.

### Evidence dimensions

The final inclusion decision must not collapse these dimensions into one unexplained score:

1. **Optimization:** whether class-wise train AP shows a sustained signal during the declared budget. A fixed-policy train F1 trajectory is supplementary continuity evidence, not the selection score.
2. **Generalization:** whether validation AP shows reproducible held-out ranking quality without test access. Fixed-policy validation F1, precision, and recall are supplementary only when the declared output policy needs them.
3. **Temporal stability and uncertainty:** whether the optimization and validation conclusions persist across nearby declared epoch windows and, where the bounded panel is run, remain qualitatively consistent across configurations. A single run cannot establish seed-level stability.
4. **Validity and mechanism:** whether support, prevalence, co-occurrence, a non-visual/context baseline, and an image-model advantage support the claimed signal rather than a shortcut or artefact.
5. **Visual evidence type:** whether audited examples are `direct`, `contextual`, `not_inferable`, or `uncertain`. Model metrics alone cannot establish direct visibility.
6. **Research relevance and evaluation support:** whether the label helps answer the thesis question and has enough support for its intended tier.

### Operational profile outputs

The analysis must assign evidence and a reasoned provisional outcome rather
than a percentile rank or a forced binary class. P5 combines these outcomes
with semantic relevance and human observability; it does not silently turn a
predictive metric into a visibility claim.

| Provisional outcome | Minimum evidence pattern | Required action |
| --- | --- | --- |
| `no_sustained_optimization` | No stable improvement in train AP under the declared budget. | Inspect support and annotations; do not call the label intrinsically impossible. |
| `optimization_only` | Sustained train AP but weak or unstable validation AP. | Investigate overfit, split, support, regularisation, and label ambiguity. |
| `generalizable_candidate` | Sustained train and validation AP, sufficient support, image-model advantage, and no contradiction from the declared temporal or configuration checks. | Send to semantic and observability review before inclusion in any direct-visual tier; report that it is not seed-validated. |
| `context_predictable` | Stable validation AP but a strong contextual/non-visual baseline or contextual evidence. | Keep distinct from direct visual-recognition claims; decide its research use explicitly. |
| `uncertain` | Low support, unstable late-window behaviour, configuration sensitivity, conflicting controls, or incomplete evidence. | Gather evidence, report uncertainty, or defer the decision. |

The profile labels are evidence summaries, not permanent metadata fields. A
label may move when the declared learner, data, support, or annotation evidence
changes.

### Candidate trajectory statistics

P1 will define the compact set of robust statistics and their aggregation
policy; P3 will evaluate them without prematurely fixing numerical promotion
gates:

- early-to-late change in train AP, using declared robust early and late windows;
- robust late-window train AP rather than a single-epoch maximum;
- validation AP for every declared configuration and its train-to-validation gap;
- late-window dispersion, nearby-window sensitivity, and configuration sensitivity where the bounded panel is run;
- validation-score resampling intervals where feasible, explicitly labelled as finite-validation-sample uncertainty rather than run-to-run uncertainty;
- a predeclared fixed-threshold F1 trajectory, only where a binary-output policy is relevant; and
- optional positive example-label confidence, variability, or forgetting diagnostics for ambiguous candidates. These are an exploratory multi-label adaptation, not a mandatory standard metric.

The pilot must reject criteria that merely guarantee a fixed quota, are dominated by a one-epoch spike, change substantially under a nearby reasonable epoch window, or let a per-label threshold maximise F1 retrospectively. It must choose explicit numerical profile gates from the fixed pilot evidence and may retain an `uncertain` band instead of forcing every label into a binary decision.

### Controls

The protocol must include the least expensive controls that answer the relevant causal questions:

- label prevalence and support correlations for every reported statistic;
- an untrained or early-epoch reference for learning-delta calculations;
- a non-visual prevalence baseline, and a cuisine-prior diagnostic where appropriate;
- an image-model-versus-non-visual baseline comparison before calling a signal plausibly visual;
- at least one matched-size random or support-matched vocabulary projection when claiming that vocabulary reduction improves training or validation behavior;
- identical-metric full-vocabulary versus selected-projection comparisons when assessing the effect of label-space reduction.

A shuffled-label control is optional at pilot time and becomes mandatory only if the cheaper controls cannot distinguish optimization artifacts from a learned signal.

## Experimental contract to freeze in P1

1. Use the frozen `v5` train and validation metadata and their saved class order; do not access test outcomes.
2. Select `M_ref` configurations globally using validation objectives, never a different configuration chosen after inspecting each label.
3. Use one primary `M_ref` configuration and an optional bounded robustness panel. Run each configuration once with one declared seed; do not present this as evidence of seed-level stability or of reproducibility across stochastic training runs.
4. Keep batch sampling, epoch budget, transforms, loss, threshold, logging cadence, and early-stopping behavior explicit and comparable.
5. Log per-label train and validation AP at every declared evaluation point, including support, prevalence, and class index; persist raw or regenerable scores sufficient to rebuild validation precision-recall summaries.
6. If F1 is reported, declare one threshold-selection and decision policy before analysis. Log its train and validation trajectory separately; never maximise it per epoch or label for selection.
7. Persist the exact model, optimizer, scheduler, loss-weighting state, transform identity, seed, code revision, environment, metadata hashes, and encoder classes in a run manifest.
8. Validate every report column against saved configurations; do not relabel booleans manually in analysis code.
9. Keep model and hyperparameter selection on validation. Train AP is used only for the predeclared optimization evidence; validation AP is the primary held-out learnability evidence; no test metric is read by selection commands.

## Planned architecture and artifacts

The exact module names are frozen during P1, but the implementation boundary is fixed now:

- `src/ingredient_selection/` will contain reusable ingestion, trajectory metrics, selection rules, validation, and plotting data preparation;
- `scripts/ingredient_selection/` will contain thin command-line entry points for historical reproduction, campaign analysis, and report generation;
- canonical training remains under `src/training/`, with only the reusable logging/configuration extensions required by this study;
- notebooks may consume generated tables for exploration, but no selection decision may depend on notebook execution order or hidden state.

Each analysis execution will create one versioned report directory containing:

- an input manifest with experiment groups, run identifiers, class order, configuration, declared random seed, code revision, environment, and data hashes;
- a tidy per-label, per-epoch metrics table;
- per-label optimization, generalization, temporal/configuration sensitivity, validity/mechanism, and observability evidence with uncertainty and profile reasons;
- provisional profile outcomes and the selected, rejected, and uncertain named projections of `v5`;
- plots for trajectories, temporal/configuration sensitivity, support relationships, and control comparisons;
- a machine-readable validation summary proving schema, class-order, and provenance checks.

Large checkpoints and raw training logs remain in `experiments/`; the report links them rather than copying them.

## Plot requirements

The replacement report must make the selection logic inspectable rather than merely attractive:

- small-multiple or filtered per-label train/validation AP trajectories with late-window dispersion;
- early-versus-late train-AP change, robust late-window distributions, and validation-AP summaries;
- precision-recall views or regenerable score references for representative and borderline labels;
- temporal-window sensitivity summaries and, when the bounded panel is run, configuration-comparison views;
- learnability statistics versus train support and prevalence;
- decision-profile plots with provisional outcomes, numerical gates, uncertainty, and reasons visible;
- like-for-like full-vocabulary, selected-projection, and matched-control comparisons when reduction claims are made.

Plots must label the split, statistic, aggregation window, declared random seed, vocabulary version, and actual transform/loss state. They must state that no repeated-seed estimate is available. The invalid historical old/new plot is retained only as history and is not regenerated as evidence.

## Validation

### Historical reproduction

- Read retained artifacts without modifying them.
- Regenerate the four 46-label Q3 sets and exact 40-label intersection.
- Assert label names and indices against the saved legacy encoder.
- Verify the current three `sel_ing_2410_metadata.json` hashes before any compatibility smoke test.
- Make the weighted-loss and augmentation discrepancies explicit in the generated historical report.

### New workflow

- Unit-test train-AP trajectory summaries and profile assignment on hand-verifiable flat, improving, noisy-spike, degrading, low-support, and missing-epoch cases.
- Verify that fixed-policy F1 is a reproducible diagnostic and cannot alter a profile through an undeclared threshold search.
- Reject inconsistent class order, duplicated run IDs, mixed metadata hashes, a missing declared seed, or contradictory configuration fields.
- Re-run analysis on the same inputs and require identical machine-readable outputs.
- Verify that changing plot styling cannot change the selected ingredient sets.
- Prove that no test metrics are read by selection commands.
- Compare the frozen rule with the historical max-Q3 baseline and the required controls on the pilot before the full campaign.

## Completion criteria

This plan is complete only when:

- the historical 40-label result is reproduced by maintained read-only code;
- the `v5` selection protocol, controls, declared seed, numerical profile gates, and decision rule are frozen before the full analysis;
- the full campaign produces deterministic reports with complete provenance;
- every provisional profile outcome and final tier has explicit evidence across optimization, generalization, support, temporal/configuration sensitivity, validity/mechanism, observability, and relevance as applicable, together with the single-run limitation;
- headline and exploratory tiers are versioned named projections, not a replacement default vocabulary;
- the test split was not used for ingredient, model, threshold, or hyperparameter selection;
- required compatibility and replacement-parity checks pass before any legacy cleanup;
- the final decisions are promoted to the appropriate objective and implementation documentation.

## Risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| A training metric is interpreted as visual generalization | Frequent or memorized labels are called recognizable | Use train AP only for optimization, require separate validation AP and observability evidence, and keep F1 diagnostic. |
| A relative rule retains labels even when all runs are poor | The selected vocabulary has an arbitrary fixed size | Use a profile with sustained optimization, validation, temporal/configuration checks, validity, and uncertainty evidence; allow an uncertain tier. |
| Per-label tuning overfits the selection process | Each label benefits from a different post-hoc configuration | Freeze one global configuration panel before inspecting selection outcomes. |
| Vocabulary reduction appears beneficial because metrics or cohorts changed | The improvement claim is invalid | Compare identical metrics, records, budgets, and single-run constraints; add matched-size controls. |
| One stochastic run is mistaken for stable evidence | A label is promoted or rejected due to seed-specific variation | Record the declared seed, use temporal windows and validation-score resampling for limited uncertainty checks, classify borderline labels as `uncertain`, and make no seed-stability claim. |
| Support dominates AP or F1 diagnostics | The study selects only frequent labels | Report support dependence, stratify evidence, and combine it with relevance and observability rather than hiding it. |
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
| 2026-08-12 | Adopted a research-informed learnability decision profile for Phase 3 | The selection must separately assess train-AP optimization trajectories, validation-AP generalization, and validity/mechanism evidence. F1 is retained only under a P1-predeclared fixed policy as a secondary diagnostic; P3 must still freeze numerical gates from a bounded pilot. |
| 2026-08-12 | Removed repeated-seed training from the `v5` selection protocol | Available time does not permit multiple runs with identical hyperparameters. Each configuration will use one declared seed; the analysis substitutes temporal/configuration checks and finite-validation-sample uncertainty where feasible, and does not claim seed-level stability. |
| 2026-08-12 | Deferred new vocabulary-selection execution until Macro-section 4 chooses `M_ref` | Learnability is conditional on the selector. The historical ResNet remains a baseline, not an automatic selector; Phase 3 retains ownership of producing the shared selected vocabulary after the model-research decision. |

## Related documentation

- [`general_plan.md`](../general_plan.md)
- [`data_ingredient_refactor/yummly_data_phase.md`](data_ingredient_refactor/yummly_data_phase.md)
- [`../project_objective/problem_definition.md`](../project_objective/problem_definition.md)
- [`../project_objective/benchmark_decisions.md`](../project_objective/benchmark_decisions.md)
- [`../project_objective/ingredient_vocabulary_audit.md`](../project_objective/ingredient_vocabulary_audit.md)
- [`../project_objective/model_comparison_methodology.md`](../project_objective/model_comparison_methodology.md)
- [`../research/topics/label_learnability/learnability_assessment.md`](../research/topics/label_learnability/learnability_assessment.md)
- [`../../README_PROJECT_KNOWLEDGE.md`](../../README_PROJECT_KNOWLEDGE.md)
