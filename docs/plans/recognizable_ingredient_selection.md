# Recognizable ingredient selection plan

**Created:** 2026-08-10
**Last updated:** 2026-10-05

This plan is the operational source of truth for Macro-section 3, **Ingredient selection**, in [`general_plan.md`](../general_plan.md). It preserves the November 2024 ResNet selection as a historical baseline and replaces its exploratory workflow with a reproducible, research-informed decision-profile study over the frozen FoodOn-first `v5` vocabulary. Macro-section 3 owns the resulting selected vocabulary and now executes against the frozen Subphase 4B reference selector. The independent Subphase 4A experiment-model shortlist is not a Phase 3 gate.

Subphase 4B completed that dependency on 2026-09-15 by freezing the 4B-D1
EfficientNetV2-S model-side protocol. P1 completed on 2026-09-24 by freezing the
complementary Phase 3-D1 campaign and measurement contract. P2 completed on
2026-09-24 with the maintained selector, deterministic audit/analysis paths,
historical regression, and real-data resource gate. P3 completed the sealed
40-epoch v3 campaign and froze the numerical profile from only the blind pilot
on 2026-09-28. P4 then applied the unchanged rule to all 165 labels from the
same campaign, without retraining or test access. The independent
Subphase 4A experiment-model portfolio remains outside this plan's selector
decision.

## Progress tracker

**Overall status:** In progress
**Current task:** P6 complete: the shared D6 projection is published with original indices, hashes and independent exclusion reasons. Original D4 evidence remains unchanged and mandatory P5 remains superseded.
**Next action:** P7: integrate the explicit projection with canonical training/analysis configuration, verify full/default and legacy parity, then assess retention-gated cleanup. Do not retrain the selector, change membership, restore manual review or replace the full-vocabulary default.

| # | Task | Status | Evidence or result |
| --- | --- | --- | --- |
| P0 | Reconstruct the historical 2024 selection process and resolve its discrepancies | **Done** | Saved configurations, metrics, metadata, checkpoints, notebooks, launchers, journals, and the external communication archive establish the four historical stages and the exact 40-label rule. The accepted resolutions are recorded in this plan. |
| P1 | Freeze the new selection question and experimental contract | **Done** | [Phase 3-D1](../project_objective/model_comparison_methodology.md#phase-3-d1--frozen-selector-campaign-and-measurement-protocol) fixes one seed-42 AdamW/warm-up/cosine configuration, 20 epochs, deterministic two-epoch audits, AP windows, fixed-0.5 F1 diagnostics, bootstrap uncertainty, low-cost controls, a sealed support-stratified pilot cohort, and the output boundary. No new selector outcome informed the decision. |
| P2 | Implement the selector integration, deterministic historical reproduction, and reusable analysis | **Done** | Added the exact EfficientNetV2-S/full-frame adapter, train/validation-only data boundary, weighted BCE and AdamW schedule, fixed-state AP/F1 audits, bootstrap and controls, hashed blind gate, manifests, thin CLIs, and read-only historical regression. All 64 repository tests pass; the real batch-8 FP32 gate passed at 3,632.20 MiB allocated and 5,362.00 MiB reserved. See the [implementation contract](../implementation_details/ingredient_selection.md). |
| P3 | Run and validate a bounded `v5` pilot | **Done** | The full 40-epoch v3 campaign completed at 14:36 UTC on 2026-09-28. Blind analysis validated all 21 audit points and exposed only 24 labels; all 24 final AP bootstraps are valid. [D4](../project_objective/model_comparison_methodology.md#phase-3-d4--pilot-frozen-numerical-profile-rule) froze simple absolute gates plus a bootstrap overlap band, bound to the campaign, cohort, pilot evidence and classifier source hashes. The [reviewed result](../experiment_results/phase3_d1_v3_pilot.md) reports 5 numeric candidates, 5 optimization-only, 1 contextual and 13 uncertain, without claiming a selected vocabulary. |
| P4 | Apply the numerical profile and verify its inclusion interpretation | **Done** | Original D4 application retained. The user-approved [D6 result](../experiment_results/phase3_d1_v3_d6_profile.md) now has statistic-consistent paired image-cluster uncertainty, independent reasons and fixed sensitivity: 59 eligible, 60 uncertain, 46 below the operational floor. Two full runs reproduce all artifacts; 100 repository tests pass with the documented generic metadata-test caveat. |
| P5 | Former mandatory semantic and visual-observability review | **Superseded** | [D5](../project_objective/model_comparison_methodology.md#phase-3-d5--numerical-selection-and-optional-interpretation-appendix) moves both manual reviews to an [optional appendix](../project_objective/ingredient_observability_protocol.md). The unannotated 64-pair packet and source are retained; no reviewer results are claimed. |
| P6 | Freeze the shared numerical-profile-based vocabulary and exploratory groups | **Done** | Published [`ingredients_selected_v5_d6_v1`](../../src/ingredient_selection/resources/ingredients_selected_v5_d6_v1.json): 59 selected, with 60 uncertain and 46 below-floor exclusions, original indices, independent reasons and source/evidence hashes. Repeated exports agree exactly; 13 focused and 113 repository tests pass. No metadata, model, score archive or test split is opened by the exporter. |
| P7 | Integrate the workflow and retire superseded scripts safely | **Pending** | Connect the explicit projection to canonical training/analysis configuration, preserve all records including empty projected targets, verify full/default and legacy parity, then clean legacy notebooks and launchers only after retention gates pass. No P7 implementation or deletion is included in P6. |

## Objective

Determine which ingredients in the standard `ingredients_target_v5_metadata.json` vocabulary provide a reproducible learning target for image-based models. Using the frozen 4B-D1 `M_ref`, identify optimization and validation signals and produce a shared numerical-profile-based projection without creating a second implicit default vocabulary. Manual relevance and observability judgments are optional interpretation outside selection under D5.

The result is not a claim that every retained ingredient is literally visible. A label may be directly visible, inferable from dish context, or learnable mainly through dataset priors. Relevant diagnostics remain visible, but numerical metrics alone do not identify those causal mechanisms or establish visual tiers.

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

**Current inclusion policy:** [D6](../project_objective/model_comparison_methodology.md#phase-3-d6--held-out-quality-inclusion-policy)
supersedes the original D4 membership vetoes after the reviewed objective-alignment
audit. Held-out quality, paired excess over prevalence and validation dispersion
govern eligibility; train acquisition/support/gap and cuisine remain independent
diagnostics. The original framework and dated D4 checkpoints below remain history.

### Evidence dimensions

The final inclusion decision must not collapse these dimensions into one unexplained score:

1. **Optimization:** whether class-wise train AP shows a sustained signal during the declared budget. A fixed-policy train F1 trajectory is supplementary continuity evidence, not the selection score.
2. **Generalization:** whether validation AP shows reproducible held-out ranking quality without test access. Fixed-policy validation F1, precision, and recall are supplementary only when the declared output policy needs them.
3. **Temporal stability and uncertainty:** whether the optimization and validation conclusions persist across nearby declared epoch windows and, where the bounded panel is run, remain qualitatively consistent across configurations. A single run cannot establish seed-level stability.
4. **Validity and mechanism:** whether support, prevalence, co-occurrence, a non-visual/context baseline, and an image-model advantage support the claimed signal rather than a shortcut or artefact.
5. **Optional visual interpretation:** `direct`, `contextual`, `not_inferable`, or `uncertain` judgments may be studied in the appendix but cannot affect selection. Model metrics alone cannot establish direct visibility.
6. **Evaluation support:** whether the label has enough numerical support for its intended projection. Manual research-relevance screening is outside selection; target validity remains owned by the existing Data contract.

### Operational profile outputs

The analysis must assign evidence and a reasoned provisional outcome rather
than a percentile rank or a forced binary class. The table below preserves
the original D4 profile. The adopted D6 review uses separate inclusion and
diagnostic axes; these names alone must not determine final membership.
Optional appendix judgments do not establish a causal signal mechanism or
literal visibility from predictive metrics.

| Provisional outcome | Minimum evidence pattern | Required action |
| --- | --- | --- |
| `no_sustained_optimization` | Train gain or absolute train AP below D4's gate. | The actual four cases gained train AP but failed its absolute floor; do not interpret the identifier as no learning or intrinsic impossibility. |
| `optimization_only` | Sustained train AP but weak or unstable validation AP. | Investigate overfit, split, support, regularisation, and label ambiguity. |
| `generalizable_candidate` | Sustained train and validation AP, sufficient support, image-model advantage, and no contradiction from the declared temporal or configuration checks. | Carry numerical evidence to P6's projection rule; report that it is neither seed-validated nor certified directly visible. |
| `context_predictable` | Held-out AP passes earlier D4 screens but the cuisine-prior margin is below its gate. | A comparison with privileged metadata does not identify the model's mechanism; D6 keeps this diagnostic out of primary inclusion. |
| `uncertain` | Low support, unstable late-window behaviour, configuration sensitivity, conflicting controls, or incomplete evidence. | Gather evidence, report uncertainty, or defer the decision. |

The profile labels are evidence summaries, not permanent metadata fields. A
label may move when the declared learner, data, support, or annotation evidence
changes.

### Candidate trajectory statistics

Phase 3-D1 fixes the compact statistics, with the active final aggregation
windows amended by Phase 3-D3; P3 will select
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

The original execution boundary above is retained as history. The active
[Phase 3-D2 amendment](../project_objective/model_comparison_methodology.md#phase-3-d2--effective-batch-and-main-workspace-execution-amendment)
requests effective batch 128, resolves a tested physical cap, and uses exact
Lightning accumulation. Physical 8/accumulation 16 is the measured candidate.

The exact checkpoint hash, resize arithmetic, padding rule, head initialization
boundary, limitations, and provenance requirements remain authoritative in
4B-D1. P2 now implements and tests the complete model, transform, loss,
optimizer/scheduler, audit, blind-analysis, and manifest boundary described in
the [implementation contract](../implementation_details/ingredient_selection.md).
No campaign outcome was inspected during implementation.

## Frozen P1 campaign contract

The binding [Phase 3-D1 decision](../project_objective/model_comparison_methodology.md#phase-3-d1--frozen-selector-campaign-and-measurement-protocol)
owns the original campaign values, amended by Phase 3-D2 for batch/provenance
and [Phase 3-D3](../project_objective/model_comparison_methodology.md#phase-3-d3--forty-epoch-campaign-amendment) for the 40-epoch horizon.
The completed implementation uses one seed-42
configuration with AdamW, a two-epoch linear warm-up followed by cosine decay,
40 complete epochs (2 warm-up + 38 cosine), effective batch 128 in true FP32, deterministic train and
validation audits before training and every two epochs, fixed-0.5 F1 diagnostic
trajectories, final-checkpoint validation AP bootstrap intervals, and no second
training-time robustness configuration.

The early AP window remains `{2,4,6}`; near and late windows are respectively
`{30,32,34,36,38}` and `{32,34,36,38,40}`. Final bootstrap scores come from epoch
40. The v2 disposable gate was interrupted before completion, and no v2
campaign started; its model cannot initialize v3.

The selection workflow must preserve the following execution boundaries:

1. Use only the frozen `v5` train and validation metadata and their saved class order. Selector commands must not open the test metadata.
2. Compute audit train AP in a separate deterministic evaluation pass from one fixed model state; never aggregate predictions from training batches whose weights changed during the epoch.
3. Generate and hash the 24-label support-stratified pilot cohort before model construction or outcome inspection. P3 may expose only that cohort until `profile_rule.json` is frozen.
4. Reuse the same sealed 40-epoch v3 run when P4 applies the rule to the remaining 141 labels. A second selector training is unnecessary unless an implementation or resource gate invalidated the first run before non-pilot inspection.
5. Derive every report field from validated configuration or run state; do not relabel transform, loss, weighting, or augmentation booleans in analysis code.
6. Keep train AP as optimization evidence and validation AP as held-out evidence. F1, bootstrap intervals, cuisine priors, and support relationships retain their declared diagnostic boundaries.
7. Treat the absence of repeated seeds and a configuration panel as missing stability evidence, not as agreement. Borderline labels remain `uncertain`.

P3 has frozen the numerical values that map these statistics to the five
provisional profile outcomes in [D4](../project_objective/model_comparison_methodology.md#phase-3-d4--pilot-frozen-numerical-profile-rule).
It used only the pilot cohort, kept the learner, metric, windows, cohort and
budget unchanged, and did not force a selected-label count.

## Planned architecture and artifacts

The exact module names may be finalized during P2, but the implementation boundary is fixed now:

- `src/ingredient_selection/` will contain reusable ingestion, trajectory metrics, selection rules, validation, and plotting data preparation;
- `scripts/ingredient_selection/` will contain thin command-line entry points for historical reproduction, campaign analysis, and report generation;
- canonical training remains under `src/training/`, with only the reusable logging/configuration extensions required by this study;
- notebooks may consume generated tables for exploration, but no selection decision may depend on notebook execution order or hidden state.

Each analysis execution will create one versioned
`analysis_outputs/ingredient_selection/<protocol_id>/` report directory
containing:

- `campaign_manifest.json` with run identity, Git base revision, exact source hashes and snapshot, actual worktree status, ordered class and data hashes, exact model/weight/head/loss/optimizer/scheduler/transform state, seed, deterministic-runtime state, environment, and device evidence;
- `pilot_cohort.json` before training and `profile_rule.json` after P3, both carrying the source hashes needed to enforce the analysis gate;
- `metrics_per_label_epoch.csv`, the tidy per-label and per-audit-epoch metrics table;
- compressed validation record identifiers, targets, and logits for every audit point, sufficient to regenerate precision-recall summaries and the final bootstrap intervals;
- per-label optimization, generalization, temporal/configuration sensitivity, support and numerical control evidence with uncertainty and profile reasons; optional manual evidence remains separate;
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
- Fix the complete 40-epoch learning-rate sequence in a unit test, including the warm-up/cosine transition at epoch 2 and resume at the next epoch boundary; reject the superseded 20-epoch budget and 18-epoch cosine in analysis.
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
- the original `v5` protocol and D4 freeze are preserved; any post-P4 inclusion amendment is explicitly versioned as outcome-informed, with its rationale and sensitivity recorded before downstream vocabulary experiments;
- the full campaign produces deterministic reports with complete provenance;
- every provisional profile outcome and final projection has explicit numerical evidence across optimization, generalization, support, temporal checks and declared controls, with unavailable seed/configuration sensitivity and the single-run limitation explicit; manual annotation is not required;
- headline and exploratory tiers are versioned named projections, not a replacement default vocabulary;
- the test split was not used for ingredient, model, threshold, or hyperparameter selection;
- required compatibility and replacement-parity checks pass before any legacy cleanup;
- the final decisions are promoted to the appropriate objective and implementation documentation.

## Risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| A training metric is interpreted as visual generalization | Frequent or memorized labels are called recognizable | Use train AP only for optimization, require separate validation AP, retain numerical controls and keep F1 diagnostic; do not claim literal visibility. |
| A relative rule retains labels even when all runs are poor | The selected vocabulary has an arbitrary fixed size | Use a profile with sustained optimization, validation, temporal/configuration checks, validity, and uncertainty evidence; allow an uncertain tier. |
| Per-label tuning overfits the selection process | Each label benefits from a different post-hoc configuration | Freeze one global configuration panel before inspecting selection outcomes. |
| Vocabulary reduction appears beneficial because metrics or cohorts changed | The improvement claim is invalid | Compare identical metrics, records, budgets, and single-run constraints; add matched-size controls. |
| One stochastic run is mistaken for stable evidence | A label is promoted or rejected due to seed-specific variation | Record the declared seed, use temporal windows and validation-score resampling for limited uncertainty checks, classify borderline labels as `uncertain`, and make no seed-stability claim. |
| Support dominates AP or F1 diagnostics | The study selects only frequent labels | Report support dependence and uncertainty, retain fixed gates and support-matched numerical controls rather than applying subjective relevance or visibility filters. |
| Notebook state or mislabeled configuration fields corrupt the report | The selection cannot be reproduced | Use deterministic source modules, manifests, schema validation, and generated plots. |
| Cleanup removes unique forensic evidence | Historical claims can no longer be verified | Apply the 2.1c retention manifest and hashes before deleting anything. |

## P3 resource-gate and launch checkpoint — 2026-09-27

The mandatory full training epoch completed on CUDA with 375 optimizer updates.
The launcher subsequently initialized a fresh model and started
`phase3-d1-v3` at 16:17 UTC (18:17 Europe/Rome) with the unchanged 40-epoch
protocol. The read-only launch audit verified the manifest's Git revision,
effective batch, 15,000 planned updates, gate hash and all 152 source ZIP entries.
Detailed memory measurements, artifact hashes and the post-fit CPU-validation
limitation are owned by the
[implementation verification](../implementation_details/ingredient_selection.md#verified-resource-and-test-evidence).

At this launch checkpoint, P3 remained **In progress**: no campaign outcome
had been inspected and no numerical rule existed. P4 and the other 141 labels
were **Deferred**. The next step was to finish this same campaign, then expose
only the sealed 24-label pilot.
Preserve the interrupted v1 and v2 history; do not pool their evidence or reuse
their model states. No additional commit is authorized by the monitoring run.

## P3 pilot and numerical-rule completion — 2026-09-28

The same fresh v3 campaign completed all 40 epochs at 14:36 UTC, retaining its
epoch-40 audit, final validation bootstrap and checkpoint. The maintained
analysis command verified the configuration, 21-point audit cadence, class
order, train/validation metadata hashes, final validation-record order and
pilot hash, then wrote only the 24-label `pilot_profile_evidence.csv`.
`validation_summary.json` reports `analysis_scope: pilot_only`, 24 visible
labels and no test access. The pilot evidence was copied immutably before the
rule was created, so later P4 output cannot erase its selection provenance.

The frozen `profile_rule.json` binds the v3 campaign identity, pilot hash,
pilot-evidence SHA-256, classifier version and exact classifier-source hash.
Its absolute train/validation AP, support, temporal/gap and image-advantage
gates are recorded once in [D4](../project_objective/model_comparison_methodology.md#phase-3-d4--pilot-frozen-numerical-profile-rule).
The dedicated [`report_pilot.py`](../../scripts/ingredient_selection/report_pilot.py)
command reclassifies only that archived pilot and produced the 24-row
`pilot_profile_decisions.csv` and its summary. The [reviewed pilot result](../experiment_results/phase3_d1_v3_pilot.md)
owns the quantitative findings and limits. A separate post-campaign
`pilot_analysis_source.zip` preserves the exact analysis/classifier sources;
it is not confused with the training source snapshot. The 72-test repository suite passes
after the bootstrap-band and pilot-report changes.

P3 is **Done**. The other 141 label outcomes remain unexposed; P4 is
**Deferred** until separately authorized. No P4/full report, reduced-vocabulary
training, semantic tier or final selected list was produced. The user-owned
dirty training/IDE files and the older v1/v2 artifacts remain untouched.

## P4 full-profile completion — 2026-09-28

The user separately authorized P4 after the P3 pilot and numerical rule were
committed as `fca417f`. The unchanged `profile_rule.json` was applied to all 165
labels from the same completed v3 campaign, with the original train/validation
metadata and audit artifacts. The full analysis validated the campaign, class
order, 21 audit points, validation-score record order, pilot evidence and
classifier source before writing the 165-row `profile_evidence.csv` (SHA-256
`5eb830a33192d0be77987e89dd4cb637f00ce714c31db0ab504c5f44a9239993`).
The P3 pilot summary was archived before the working validation summary was
replaced; all 24 archived decisions match the P4 rows exactly.

The new `report_campaign.py` command revalidates those inputs and writes
`p4_profile_report.json` with all named provisional groups, reason counts and
SHA-256 provenance, plus three diagnostic figures in SVG/PNG form. Its report
and all six figure files reproduced byte-for-byte on a second run. The
[reviewed full-profile result](../experiment_results/phase3_d1_v3_full_profile.md)
records 25 `generalizable_candidate`, 40 `optimization_only`, 13
`context_predictable`, 4 `no_sustained_optimization` and 83 `uncertain` labels.
These are numerical profile outcomes under one seed/configuration, not direct
visibility annotations or a final selected vocabulary. No test, P5/P6,
reduced-vocabulary training or legacy cleanup was performed. P4 is **Done**;
P5 is the next deferred gate.

## P5 preparation and pilot packet — 2026-09-29

**Historical checkpoint:** the mandatory dependency described below was superseded on 2026-10-04 by D5. The packet and software remain optional; no human review was completed.

The [P5 pilot protocol](../project_objective/ingredient_observability_protocol.md)
separates label-level semantic validity from instance-level direct, contextual,
not-inferable and uncertain image evidence. Its purposive eight-label pilot
tests the rubric rather than estimating population visibility. The maintained
pure-standard-library command generated 64 validation-only ingredient–image
pairs (48 target-present, 16 target-absent) with distinct image files, source
hashes and two different blinded reviewer orders. The packet hash is
`8c792c167bd7ec4583546836c2e923a74303293e2a8c38c049ef612ff7d5cf18`.
Re-running preparation accepted the byte-identical outputs. Two focused tests
cover deterministic sampling, agreement arithmetic and tamper/incomplete-
response rejection. The complete ML test suite could not be re-executed on
2026-09-29 because a bare `import torch` segfaulted in the WSL environment;
this is a runtime verification limitation, not a P5 result.

P5 is **In progress**. Neither reviewer has supplied an annotation; no human
agreement, semantic verdict, direct-visibility claim, final tier or P6 action
exists. The next step is two independent reviews of the sealed packet, then
pilot disagreement analysis and a frozen main-panel design. Reviewer answers
must remain outside Git and must not alter the recipe targets.

## P5 scope amendment and P6 handoff — 2026-10-04

**Dated handoff:** the later inclusion-policy review below reopens P4 before
this P6 export. The removal of manual review remains applicable.

The user moved both manual semantic-relevance and visual-observability reviews
outside the main study. [D5](../project_objective/model_comparison_methodology.md#phase-3-d5--numerical-selection-and-optional-interpretation-appendix)
owns the decision: selection assesses the model's learning on the existing
ingredient targets, and subjective judgments must not decide membership.
P5's mandatory role is **Superseded**, not successfully annotated. Its rubric,
64-pair packet and tools are retained as an optional future-results appendix.

P6 is **Pending** and no longer waits for reviewers. Its bounded tasks are:

1. Specify a deterministic mapping from D4 outcomes to the primary shared
   projection and explicitly document the treatment of contextual, train-only,
   uncertain and failed-optimization outcomes without changing D4 gates.
2. Export the selected names and original `v5` indices in saved class order,
   plus excluded/uncertain reasons and links to the P4 evidence.
3. Version the projection with rule, class-order and evidence hashes; verify
   deterministic regeneration, membership, and test isolation.
4. Record the result and downstream handoff for identical-vocabulary model
   comparisons and the later selected-versus-random controls.

This amendment does not publish `V_selected` or run P6. No appendix annotation
is required for plan completion, and no literal-visibility claim is introduced.

## P4 inclusion-policy review — 2026-10-05

The user requested policy correctness against the project objective, explicitly
without requiring more than 25 ingredients. The completed
[read-only audit](../experiment_results/phase3_d1_v3_inclusion_policy_audit.md)
reproduces all D4 outcomes, reports independent failures and shows that the
cuisine gate, train gap and second support cutoff answer additional questions
beyond held-out recipe prediction. The source-bound original classifier and
artifacts are unchanged. Counterfactual counts of 55/58/60 and the 0.15/0.20/0.25
floor sensitivity are diagnostics, not selected-vocabulary candidates chosen
by size.

P4 is reopened with the following bounded remaining work:

1. Record the adopted version of the [concrete proposal](../project_objective/model_comparison_methodology.md#post-p4-inclusion-policy-review--proposed-amendment), including the operational quality definition. Preserve D4 and explicitly identify the amendment as post-outcome.
2. Validate the validation-only resampling units, then compute uncertainty for the fixed five-checkpoint median and paired excess over the constant baseline from saved scores. Never treat an epoch-40-only interval as an interval for that median.
3. Report all independent evidence axes, changed membership and the fixed floor-sensitivity panel without choosing a desired count, ingredient list or favourable later result.
4. Freeze that reviewed rule/report and hand the deterministic projection to P6. Keep train acquisition and contextual diagnostics, the original full-vocabulary benchmark, and the single-run limitation visible.

This checkpoint completes the requested conceptual/quantitative review, not
the revised numerical analysis or P6. No training, test access, manual review,
artifact rewrite or commit was performed for it.

## D6 adoption and reopened P4 completion — 2026-10-05

The user approved the proposed policy without a retained-count target.
[D6](../project_objective/model_comparison_methodology.md#phase-3-d6--held-out-quality-inclusion-policy)
records the binding operational quality definition and its post-outcome status.
The maintained `review_inclusion.py` command validates the original artifacts,
recovers 5,847 validation image groups from 5,996 records, and computes matching
five-checkpoint-median and paired prevalence-excess intervals without training
or inference. D4's rule, classifier source and outputs remain unchanged.

The [reviewed result](../experiment_results/phase3_d1_v3_d6_profile.md) records
59 eligible, 60 uncertain and 46 below-floor labels; all original 25 candidates
remain eligible. The fixed sensitivity panel gives 82/59/41 eligible at
0.15/0.20/0.25. Two complete executions reproduce all six D6 artifacts exactly.
There are 26 new synthetic tests and 100 passing repository tests. The result
record explicitly distinguishes D6's test-isolated I/O from the existing
generic encoder test's metadata-only test-split compatibility check; no test
predictions or predictive metrics informed the policy or membership.

P4 and Work packages 3.2–3.3 are **Done**. P6 is **Pending**, with no metadata
projection exported yet. Its active mapping is D6 `included` to shared
eligibility; keep the 60 uncertain and 46 below-floor labels out of that
primary projection, preserve their reasons, and retain all 165 default outputs
for the full-task comparison. P7 integration and any cleanup remain deferred.
No extra selector run, manual review, test evaluation or commit was performed.

## P6 shared-projection completion — 2026-10-05

After commit `a1d17bc` retained the D6 review, the user-authorized P6 published
the shared [`ingredients_selected_v5_d6_v1`](../../src/ingredient_selection/resources/ingredients_selected_v5_d6_v1.json)
definition from the exact approved rule/report. It selects D6 `included`
only, preserves saved class order and original indices, and carries the
60 uncertain/46 below-floor exclusions with all independent reasons and axes.
No threshold, model, count target or human exception was added.

The [maintained exporter](../../scripts/ingredient_selection/export_projection.py)
uses standard-library-only artifact I/O and writes an immutable resource rather
than a second metadata generation. Two exports produce identical bytes.
Thirteen synthetic/unit checks and all 113 repository tests pass, including
class/column identity, reasons, empty-target handoff, approved-hash validation,
test/data-path isolation, write-once conflict handling and source/input changes
before publication. The full suite's existing test-metadata compatibility check
remains distinct from P6; no predictive test evaluation occurred.

The [reviewed P6 publication](../experiment_results/phase3_d1_v3_d6_profile.md#p6-publication--2026-10-05)
owns the artifact hashes; the [methodological handoff](../project_objective/model_comparison_methodology.md#p6-shared-projection-freeze--2026-10-05)
and [implementation contract](../implementation_details/ingredient_selection.md#p6-frozen-projection)
explain use and limitations. The full 165-label default, D4/D6 inputs and
user-owned training/IDE edits are unchanged. No selector training, metadata
rewrite, random-control experiment or cleanup was performed.

P6 is **Done** and P7 is **Pending**. Work package 3.5 remains **In progress**
because runtime integration and retention/parity gates are separate. P7 must
consume the saved selected order without independently fitting it per split,
retain every original record even when projection is empty, and keep opt-in
selection distinct from the default full task. Phase 6 owns later training and
matched-random controls. No P7 action is authorized by this completion record.

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
| 2026-09-24 | Completed P2 selector integration | Implemented the exact model/transform and train/validation-only campaign boundary, deterministic AP/F1 audits and analysis, hashed pilot/rule gate, historical regression, provenance schemas, and resource check. The 64-test suite and real RTX 4060 batch-8 gate pass; no selector outcome was inspected. P3 is next. |
| 2026-09-27 | Adopted Phase 3-D2 and restarted P3 preparation | Interrupted the incomplete batch-8/no-accumulation v1 campaign by user request without per-label inspection. Retained its artifacts in the main workspace. Added the rerunnable launcher, exact effective-128 resolution and tail weighting, source snapshots, and descending real CUDA capacity tests. Physical 128/64/32/16 OOM; physical 8 with accumulation 16 completes two optimizer steps. All 68 tests pass; the full-epoch gate must precede the fresh v2 campaign. |
| 2026-09-27 | Adopted Phase 3-D3 before v2 campaign launch | The user extended the sealed campaign to 40 epochs and authorized a commit. Interrupted the incomplete disposable v2 gate, retained the 2-epoch warm-up and two-epoch audits, extended cosine to 38 epochs, moved the final AP windows to epochs 30–40, and versioned the replacement as v3. All 72 repository tests pass, including manifest agreement and rejection of the old budget/scheduler. No ingredient outcome informed the change. Commit and relaunch the mandatory full-epoch gate before the fresh 40-epoch campaign. |
| 2026-09-27 | Passed the v3 full-epoch training gate and launched the sealed campaign | The resource gate completed 375 CUDA optimizer updates, then the launcher constructed a fresh model for 40 epochs from revision `192059e`. Verified the gate, manifest and all 152 source-snapshot entries without opening per-label outcomes. The post-fit validation check was CPU-based and is documented separately from the CUDA training-memory result. |
| 2026-09-28 | Completed P3 and froze the pilot-derived numerical rule | The fresh v3 campaign completed 40 epochs; blind analysis exposed only 24 labels. D4 freezes absolute AP/support/stability/advantage gates with a conservative bootstrap overlap band and source-bound rule hash. The pilot-only report is reproducible; P4 remains deferred. |
| 2026-09-28 | Completed P4 under separate authorization without changing D4 | Applied the frozen rule to the same 165-label run, confirmed exact pilot parity and deterministic full reports/figures, and retained 83 uncertain outcomes rather than forcing a target count. The numerical candidates require P5 relevance/observability evidence before P6 can freeze named tiers. |
| 2026-09-29 | Began P5 with a blind two-reviewer pilot | Adopted a separate semantic and visual rubric, generated a source-hashed 64-pair validation-only packet, and implemented deterministic preparation and agreement scoring. Human review and the main panel remain pending. |
| 2026-10-04 | Superseded mandatory P5 and retained an optional interpretation appendix | By user decision, manual relevance and observability judgments are outside the model-learnability selection objective and may introduce subjective membership bias. D5 removes reviewer dependencies, preserves D4/P4 evidence, and releases P6 numerical projection work without claiming a completed human review. |
| 2026-10-05 | Reopened P4 inclusion interpretation after the user's objective-alignment review | The audit reproduces D4 but identifies privileged cuisine, train-gap, support and interval-statistic concerns. A concrete held-out-quality proposal and bounded sensitivity are documented; amendment adoption, matching uncertainty and P6 freeze remain pending. No retained-count target is introduced. |
| 2026-10-05 | Adopted D6 and completed the reopened P4 review | User-approved held-out-quality policy, paired image-cluster intervals and independent diagnostics produce a separately frozen, twice-reproduced report. Original D4 evidence is retained, no cardinality target is used, and P6 export is next. |
| 2026-10-05 | Completed P6 shared-vocabulary freeze | Published the approved D6 `included` subsequence as a versioned, write-once definition with original indices and explicit excluded groups/reasons. Reproduction and 113 tests pass; the full default stays unchanged and P7 runtime integration is next. |

## Related documentation

- [`general_plan.md`](../general_plan.md)
- [`data_ingredient_refactor/yummly_data_phase.md`](data_ingredient_refactor/yummly_data_phase.md)
- [`../project_objective/problem_definition.md`](../project_objective/problem_definition.md)
- [`../project_objective/benchmark_decisions.md`](../project_objective/benchmark_decisions.md)
- [`../project_objective/ingredient_vocabulary_audit.md`](../project_objective/ingredient_vocabulary_audit.md)
- [`../project_objective/model_comparison_methodology.md`](../project_objective/model_comparison_methodology.md)
- [`../project_objective/ingredient_observability_protocol.md`](../project_objective/ingredient_observability_protocol.md)
- [`../experiment_results/phase3_d1_v3_pilot.md`](../experiment_results/phase3_d1_v3_pilot.md)
- [`../experiment_results/phase3_d1_v3_full_profile.md`](../experiment_results/phase3_d1_v3_full_profile.md)
- [`reference_selector_research.md`](reference_selector_research.md)
- [`../research/topics/label_learnability/learnability_assessment.md`](../research/topics/label_learnability/learnability_assessment.md)
- [`../../README_PROJECT_KNOWLEDGE.md`](../../README_PROJECT_KNOWLEDGE.md)
