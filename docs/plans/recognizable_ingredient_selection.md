# Recognizable ingredient selection plan

**Created:** 2026-08-10
**Last updated:** 2026-09-24

This plan is the operational source of truth for Macro-section 3, **Ingredient selection**, in [`general_plan.md`](../general_plan.md). It preserves the November 2024 ResNet selection as a historical baseline and replaces its exploratory workflow with a reproducible, research-informed decision-profile study over the frozen FoodOn-first `v5` vocabulary. Macro-section 3 owns the resulting selected vocabulary and now executes against the frozen Subphase 4B reference selector. The independent Subphase 4A experiment-model shortlist is not a Phase 3 gate.

Subphase 4B completed that dependency on 2026-09-15 by freezing the 4B-D1
EfficientNetV2-S model-side protocol. P1 completed on 2026-09-24 by freezing the
complementary Phase 3-D1 campaign and measurement contract. Macro-section 3 is
now ready for P2 implementation. The independent Subphase 4A experiment-model
portfolio remains outside this plan's selector decision.

## Progress tracker

**Overall status:** In progress
**Current task:** P2 — implement the frozen 4B-D1 model-side and Phase 3-D1 campaign-side contracts.
**Next action:** Add the maintained EfficientNetV2-S wrapper and full-frame transform, selector-specific weighted loss and AdamW schedule, deterministic audit path, AP/F1 logging, sealed pilot cohort, provenance manifest, and schema/resource tests before any campaign execution.

| # | Task | Status | Evidence or result |
| --- | --- | --- | --- |
| P0 | Reconstruct the historical 2024 selection process and resolve its discrepancies | **Done** | Saved configurations, metrics, metadata, checkpoints, notebooks, launchers, journals, and the external communication archive establish the four historical stages and the exact 40-label rule. The accepted resolutions are recorded in this plan. |
| P1 | Freeze the new selection question and experimental contract | **Done** | [Phase 3-D1](../project_objective/model_comparison_methodology.md#phase-3-d1--frozen-selector-campaign-and-measurement-protocol) fixes one seed-42 AdamW/warm-up/cosine configuration, 20 epochs, deterministic two-epoch audits, AP windows, fixed-0.5 F1 diagnostics, bootstrap uncertainty, low-cost controls, a sealed support-stratified pilot cohort, and the output boundary. No new selector outcome informed the decision. |
| P2 | Implement the selector integration, deterministic historical reproduction, and reusable analysis | **Pending** | Add the frozen model/transform, loss, optimizer/scheduler, deterministic audit, AP/provenance, blind-cohort, historical reproduction, and reporting paths through canonical APIs. Notebooks become optional views, not execution state. |
| P3 | Run and validate a bounded `v5` pilot | **Deferred** | Execute the sealed single campaign after P2 gates pass, expose only the deterministic 24-label pilot cohort, and freeze absolute numerical profile gates without choosing a fixed retained count. |
| P4 | Apply the frozen profile to the full `v5` campaign evidence | **Deferred** | Unlock the remaining 141 labels from the same sealed 20-epoch run, apply the immutable P3 rule, preserve complete provenance, and report the single-run limitation. No second selector training is required unless a pre-outcome protocol failure invalidates the run. |
| P5 | Combine learnability with relevance and visual-observability evidence | **Deferred** | Apply the semantic, support, and annotation protocol; distinguish direct visual evidence from contextual predictability. |
| P6 | Freeze named headline and exploratory ingredient tiers | **Deferred** | Publish versioned projections of the shared `v5` vocabulary, with explicit inclusion evidence and uncertainty. |
| P7 | Integrate the workflow and retire superseded scripts safely | **Deferred** | Connect training and analysis to canonical APIs, verify parity, document current behavior, then clean legacy notebooks and launchers only after retention gates pass. |

## Objective

Determine which ingredients in the standard `ingredients_target_v5_metadata.json` vocabulary provide a meaningful and reproducible learning target for image-based models. Using the frozen 4B-D1 `M_ref`, the workflow must identify labels whose signal it learns, separate that evidence from validation generalization and human visual observability, and produce named experimental projections without creating a second implicit default vocabulary.

The result is not a claim that every retained ingredient is literally visible. A label may be directly visible, inferable from dish context, or learnable mainly through dataset priors. Those cases must remain distinguishable in the evidence and final tiers.

## Scope

- Reproduce the historical 40-label selection exactly from retained aggregate artifacts as a read-only regression baseline.
- Design and implement a reusable, configuration-driven per-label training analysis for the 165-label `v5` target space using the Subphase 4B-approved `M_ref`.
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
- Replacing the frozen 4B-D1 selector with an arbitrary historical ResNet or revising it after inspecting label-selection outcomes.
- Adding loss variants, stronger augmentation, or other ablations beyond the frozen 4B-D1 primary protocol as selection requirements before the pilot justifies them.
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

**Status:** Adopted for Phase 3 planning on 2026-08-12, released after Subphase
4B froze `M_ref`, and completed at the P1 contract level on 2026-09-24. P3 will
freeze only the numerical promotion gates from the isolated pilot cohort.

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

Phase 3-D1 fixes the compact statistics and aggregation windows; P3 will select
numerical promotion gates from the isolated pilot cohort without changing the
statistics:

- initialization-to-late and early-to-late changes in train AP, using the frozen audit epochs and robust windows;
- robust late-window train and validation AP rather than a single-epoch maximum;
- validation AP for every declared configuration and its train-to-validation gap;
- late-window dispersion and nearby-window sensitivity; configuration sensitivity is explicitly unavailable because P1 adopted no second training configuration;
- a deterministic final-checkpoint validation AP bootstrap, explicitly labelled as finite-validation-sample uncertainty rather than run-to-run uncertainty; and
- a global fixed-0.5 F1 trajectory as diagnostic evidence only.

The pilot must reject criteria that merely guarantee a fixed quota, are dominated by a one-epoch spike, change substantially under a nearby reasonable epoch window, or let a per-label threshold maximise F1 retrospectively. It must choose explicit numerical profile gates from the fixed pilot evidence and may retain an `uncertain` band instead of forcing every label into a binary decision.

### Controls

The protocol must include the least expensive controls that answer the relevant causal questions:

- label prevalence and support correlations for every reported statistic;
- an untrained or early-epoch reference for learning-delta calculations;
- a non-visual prevalence baseline, and a cuisine-prior diagnostic where appropriate;
- an image-model-versus-non-visual baseline comparison before calling a signal plausibly visual;
- support/prevalence fields sufficient for the later matched-size support-matched vocabulary controls; and
- identical-metric full-vocabulary versus selected-projection comparisons only when Macro-section 6 assesses the effect of label-space reduction.

A shuffled-label control is optional at pilot time and requires a recorded
addendum before the non-pilot labels are exposed if the cheaper controls cannot
distinguish optimization artifacts from a learned signal. Reduced-vocabulary
training and matched-random runs remain owned by Macro-section 6, not P1–P4.

## Frozen `M_ref` handoff from Subphase 4B

The binding [4B-D1 decision](../project_objective/model_comparison_methodology.md#4b-d1--frozen-reference-selector-protocol)
is an incoming constraint, not a P1 search space. It fixes:

- Torchvision EfficientNetV2-S with exact
  `EfficientNet_V2_S_Weights.IMAGENET1K_V1` initialization and full-backbone
  training from the first optimizer step;
- RGB full-frame 384×384 aspect-preserving fit/pad preprocessing, bilinear
  antialiasing, ImageNet normalization, a primary train-only horizontal flip,
  and deterministic validation preprocessing;
- stock global average pooling and classifier dropout followed by a new biased
  `Linear(1280, 165)` layer initialized after the declared seed;
- mean-reduced positive-weighted BCE, with the ordered `pos_weight` vector
  computed only from train support; and
- true FP32, physical batch 8, no gradient accumulation as the initial 8 GB
  execution contract, subject to a recorded pre-campaign methodology revision
  if the complete instrumented implementation cannot pass its resource gate.

The exact checkpoint hash, resize arithmetic, padding rule, head initialization
boundary, limitations, and provenance requirements remain authoritative in
4B-D1. The current repository does not yet implement the complete protocol.
P1 has frozen the campaign settings below; P2 implements and tests the missing
wrapper, transform, AP instrumentation, and manifest.

## Frozen P1 campaign contract

The binding [Phase 3-D1 decision](../project_objective/model_comparison_methodology.md#phase-3-d1--frozen-selector-campaign-and-measurement-protocol)
now owns the exact campaign values. In summary, P2 must implement one seed-42
configuration with AdamW, a two-epoch linear warm-up followed by cosine decay,
20 complete epochs, physical batch 8 in true FP32, deterministic train and
validation audits before training and every two epochs, fixed-0.5 F1 diagnostic
trajectories, final-checkpoint validation AP bootstrap intervals, and no second
training-time robustness configuration.

The selection workflow must preserve the following execution boundaries:

1. Use only the frozen `v5` train and validation metadata and their saved class order. Selector commands must not open the test metadata.
2. Compute audit train AP in a separate deterministic evaluation pass from one fixed model state; never aggregate predictions from training batches whose weights changed during the epoch.
3. Generate and hash the 24-label support-stratified pilot cohort before model construction or outcome inspection. P3 may expose only that cohort until `profile_rule.json` is frozen.
4. Reuse the same sealed 20-epoch run when P4 applies the rule to the remaining 141 labels. A second selector training is unnecessary unless an implementation or resource gate invalidated the first run before non-pilot inspection.
5. Derive every report field from validated configuration or run state; do not relabel transform, loss, weighting, or augmentation booleans in analysis code.
6. Keep train AP as optimization evidence and validation AP as held-out evidence. F1, bootstrap intervals, cuisine priors, and support relationships retain their declared diagnostic boundaries.
7. Treat the absence of repeated seeds and a configuration panel as missing stability evidence, not as agreement. Borderline labels remain `uncertain`.

P3 still owns the numerical values that map these frozen statistics to the five
provisional profile outcomes. It may choose simple absolute gates and an
uncertain band from the pilot cohort, but may not change the learner, metric,
windows, cohort, budget, or force a fixed number of selected labels.

## Planned architecture and artifacts

The exact module names may be finalized during P2, but the implementation boundary is fixed now:

- `src/ingredient_selection/` will contain reusable ingestion, trajectory metrics, selection rules, validation, and plotting data preparation;
- `scripts/ingredient_selection/` will contain thin command-line entry points for historical reproduction, campaign analysis, and report generation;
- canonical training remains under `src/training/`, with only the reusable logging/configuration extensions required by this study;
- notebooks may consume generated tables for exploration, but no selection decision may depend on notebook execution order or hidden state.

Each analysis execution will create one versioned
`analysis_outputs/ingredient_selection/<protocol_id>/` report directory
containing:

- `campaign_manifest.json` with run identity, clean code revision, ordered class and data hashes, exact model/weight/head/loss/optimizer/scheduler/transform state, seed, deterministic-runtime state, environment, and device evidence;
- `pilot_cohort.json` before training and `profile_rule.json` after P3, both carrying the source hashes needed to enforce the analysis gate;
- `metrics_per_label_epoch.csv`, the tidy per-label and per-audit-epoch metrics table;
- compressed validation record identifiers, targets, and logits for every audit point, sufficient to regenerate precision-recall summaries and the final bootstrap intervals;
- per-label optimization, generalization, temporal/configuration sensitivity, validity/mechanism, and observability evidence with uncertainty and profile reasons;
- provisional profile outcomes and the selected, rejected, and uncertain named projections of `v5`;
- plots for trajectories, temporal/configuration sensitivity, support relationships, and control comparisons;
- `validation_summary.json`, proving schema, class-order, provenance, blind-pilot, deterministic-analysis, test-isolation, and resource checks.

Large checkpoints and raw training logs remain in `experiments/`; the report links them rather than copying them.

## Plot requirements

The replacement report must make the selection logic inspectable rather than merely attractive:

- small-multiple or filtered per-label train/validation AP trajectories with late-window dispersion;
- early-versus-late train-AP change, robust late-window distributions, and validation-AP summaries;
- precision-recall views or regenerable score references for representative and borderline labels;
- temporal-window sensitivity summaries; the report must mark configuration sensitivity unavailable rather than inventing a comparison;
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
- Prove that audit train AP comes from a deterministic evaluation pass at one model state rather than the stochastic training loop.
- Fix the complete 20-epoch learning-rate sequence in a unit test, including the warm-up/cosine transition at epoch 2 and resume at the next epoch boundary.
- Verify that fixed-policy F1 is a reproducible diagnostic and cannot alter a profile through an undeclared threshold search.
- Reject inconsistent class order, duplicated run IDs, mixed metadata hashes, a missing declared seed, or contradictory configuration fields.
- Reject a pilot cohort that does not reproduce the Phase 3-D1 support-stratified SHA-256 rule, and prevent all-label analysis until the hashed rule file exists.
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
| 2026-08-12 | Deferred new vocabulary-selection execution until the then-undivided Macro-section 4 chose `M_ref` (now owned by Subphase 4B) | Learnability is conditional on the selector. The historical ResNet remains a baseline, not an automatic selector; Phase 3 retains ownership of producing the shared selected vocabulary after the model-research decision. |
| 2026-08-27 | Narrowed the resume dependency to Subphase 4B | Macro-section 4 now separates experiment-model research (4A) from reference-selector research (4B). Phase 3 needs the frozen `M_ref`, not completion of the independent experiment-model shortlist. |
| 2026-09-15 | Accepted the completed 4B-D1 EfficientNetV2-S handoff and resumed Macro-section 3 at P1 | The selector's exact model-side protocol and interpretation boundary are now binding. P1 owns only the remaining campaign settings and measurement contract; implementation and pilot execution remain P2 and P3. |
| 2026-09-24 | Completed P1 and adopted Phase 3-D1 | One seed-42 AdamW/warm-up/cosine configuration, 20 epochs, deterministic two-epoch audits, AP/F1/uncertainty controls, the sealed 24-label pilot, one-run P3/P4 reuse, and the output boundary are frozen. No new selector outcome was inspected; P2 implementation is next. |

## Related documentation

- [`general_plan.md`](../general_plan.md)
- [`data_ingredient_refactor/yummly_data_phase.md`](data_ingredient_refactor/yummly_data_phase.md)
- [`../project_objective/problem_definition.md`](../project_objective/problem_definition.md)
- [`../project_objective/benchmark_decisions.md`](../project_objective/benchmark_decisions.md)
- [`../project_objective/ingredient_vocabulary_audit.md`](../project_objective/ingredient_vocabulary_audit.md)
- [`../project_objective/model_comparison_methodology.md`](../project_objective/model_comparison_methodology.md)
- [`reference_selector_research.md`](reference_selector_research.md)
- [`../research/topics/label_learnability/learnability_assessment.md`](../research/topics/label_learnability/learnability_assessment.md)
- [`../../README_PROJECT_KNOWLEDGE.md`](../../README_PROJECT_KNOWLEDGE.md)
