# Experiment comparison and optional per-ingredient logging

**Created:** 2026-09-14
**Last updated:** 2026-09-14

## Objective and scope

Implement one command that accepts N experiment directories, analyzes trials within each experiment and compares compatible experiments, then writes a machine-readable JSON report and an interactive HTML report. Add optional per-ingredient metric logging to the Lightning model so new experiments can supply that additional analysis when needed.

This is the operational plan for [Work package 7.1, experiment observability and comparison tooling](../general_plan.md#71-experiment-observability-and-comparison-tooling), with a producer-side integration into [Macro-section 6](../general_plan.md#6-training-and-hyperparameter-tuning). Tooling can be developed against existing artifacts before the final benchmark-comparison gate is satisfied.

The completed [artifact audit](../implementation_details/experiment_artifacts.md) owns verified persistence behavior and its limitations. This plan owns requirements, implementation decisions and execution history. The maintained runtime contract is documented in [experiment comparison and per-ingredient logging](../implementation_details/experiment_comparison.md).

## Progress tracker

**Overall status:** Done
**Current task:** Initial implementation and target-v5 validation complete.
**Next action:** Use the maintained command for exploratory comparisons; add extensions only when a new analysis requirement is approved.

| ID | Task | Status | Evidence or completion result |
| --- | --- | --- | --- |
| P0 | Inspect artifacts and verify histogram extraction | **Done** | [Audit](../implementation_details/experiment_artifacts.md), [inventory](../../src_scratches/experiment_comparison_audit/inventory.json), [raw parameter example](../../src_scratches/experiment_comparison_audit/parameter_histogram_example.json) |
| P1 | Define logging, observation and report contracts | **Done** | Precision/recall/F1 at threshold 0.5; indexed keys; schema 1.0.0; explicit reconciliation and cohort rules |
| P2 | Implement optional Lightning-model logging | **Done** | Default/off, round-trip, full-epoch math, empty-split, HPO persistence and bounded Lightning-fit tests |
| P3 | Implement discovery, readers and session reconciliation | **Done** | Data-only config, sparse CSV, full TensorBoard, disposable-journal Optuna and read-only W&B readers; DINOv2 resume cases recovered |
| P4 | Implement scalar and hyperparameter analysis | **Done** | Curve descriptors, contributor curves, cohort selection and conditional HPO summaries |
| P5 | Implement parameter-distribution analysis | **Done** | Validated histogram moments, quantile intervals, near-zero bounds, CDF distance and trajectories |
| P6 | Implement experiment aggregation and comparison rules | **Done** | Loss-policy cohorts, pruning treatment, label-contract checks and optional ingredient trajectories |
| P7 | Generate consistent JSON and interactive HTML | **Done** | Strict JSON and self-contained HTML are generated from one report object |
| P8 | Validate end to end and document the maintained CLI | **Done** | All 42 repository tests, including 17 focused tests, passed; 200 target-v5 trials, two selected W&B sessions and two checkpoints processed |

Update this tracker at completed-step checkpoints, including evidence, decisions, newly discovered work and the next action. P2 and P3 can proceed independently after P1; the existing-experiment comparator must not depend on retraining or on enabling the new flag.

## Agreed requirements

| Topic | Requirement |
| --- | --- |
| Configuration owner | The optional ingredient-logging flag belongs to the Lightning model, `lgn_model`, not the internal torch model |
| Default | Disabled: `false` |
| Identifier | `log_per_ingredient_metrics`; launcher override `hp_log_per_ingredient_metrics` |
| Optional ingredient data | Absence must not prevent a general comparison report |
| Experiment hierarchy | One experiment contains X trials; compare both within and across experiments |
| Source of parameter dynamics | Reuse the numerical histograms already logged by W&B |
| Accepted tradeoff | Preserve original bin edges/counts and derive explicitly approximate tensor statistics |
| Parameter scaling | Model-parameter histograms need no gradient-scaler correction |
| Gradient interpretation | Existing backward-gradient histograms remain subject to the audited AMP/accumulation limitations |
| Code location | One entry point and helpers under `scripts/analise_exp/compare_experiments/` |
| Outputs | JSON with analysis results and provenance; HTML for visual exploration of the same results |
| Experiment access | Read existing artifacts without modifying them or launching training/inference implicitly |

Full tensor histories, exact coordinate-wise updates and successive-gradient alignment are not requirements of the first version. No change to mixed-precision training is required for the parameter-histogram path. The optional logging flag controls reporting; it does not change the model output vocabulary, training loss or availability of aggregate metrics.

## A. Changes to optional per-ingredient logging

### Configuration and persistence

Add `log_per_ingredient_metrics: bool = False` to `BaseLGNM`, with explicit forwarding through `BaseWithSchedulerLGNM` and the configuration-based loading path. Store it in `ExpConfig.lgn_model`, whose serialized section is `hyper_parameters`.

Supported configuration usage:

```python
ExpConfig(hp_log_per_ingredient_metrics=True)
```

Affected components and completion results:

| Component | Result |
| --- | --- |
| [src/commons/exp_config.py](../../src/commons/exp_config.py) | Added the false default and verified override/serialization round trips |
| [src/lightning/lgn_models.py](../../src/lightning/lgn_models.py) | Accepts, retains, forwards and reloads the flag; logs optional full-epoch per-ingredient statistics |
| [src/training/commons.py](../../src/training/commons.py) | No change required; the canonical configuration-loading path already forwards Lightning-model fields |
| [src/lightning/custom_callbacks.py](../../src/lightning/custom_callbacks.py) | No change required; reports use external trial configuration and lazy selected-checkpoint metadata |
| [scripts/launch_exps/](../../scripts/launch_exps/) | Existing defaults remain unchanged; opt-in usage is maintained in the implementation contract |

Missing flags in old configurations resolve to `false` for the new optional stream. Existing recorded per-label metrics remain readable. The implementation accounts for historical launchers that already configure vector-valued metrics, so the convenience flag does not silently reinterpret old artifacts or remove a supported legacy reporting path.

Persist the flag in experiment/trial configuration even when HPO uses `hparams_to_register` to restrict logger-visible hyperparameters. Checkpoint loading must use the existing full/light checkpoint contract rather than assuming every checkpoint embeds the full model configuration.

### Metric semantics and label identity

When enabled, emit the declared per-ingredient metric set for train and validation at epoch granularity. Test metrics are emitted only during an explicitly requested evaluation; enabling logging must never trigger a test run.

The initial metric set is precision, recall and F1 with threshold 0.5, per-label averaging disabled and undefined divisions resolved to zero under the validated TorchMetrics contract. AP is excluded because its exact state has a materially different memory contract. The switch does not introduce threshold optimization or calibration.

Accumulate the metric state over the split and compute the per-ingredient result at epoch end. Do not obtain epoch F1/AP by averaging batch-level F1/AP. Keep train/validation states separate, reset correctly between epochs, and account for sanity validation and resumed epochs.

Use stable label indices in metric keys and resolve names through the saved encoder class order. The implemented key shape is `val_per_ingredient/<metric>/<label_index>`. Reuse the existing serialized encoder when available, and attach its identity/order to the report. Do not infer label identity from alphabetical sorting or current dataset defaults.

With the flag disabled, avoid creating/updating the new per-ingredient metric state and emit no new per-ingredient series. Preserve general logging. In the report distinguish:

- intentionally disabled in a configuration that records the flag;
- absent in a legacy run whose logging policy is unknown;
- enabled but missing/incomplete;
- available with a validated label mapping.

The current scalar aggregation path has its own legacy semantics, documented in the audit. Do not silently relabel historical scalars as globally computed metrics. Any change to existing aggregate behavior needs an explicit versioned contract; it is not an implicit consequence of adding the optional stream.

## B. Comparison tool design

### Implemented layout

```text
scripts/analise_exp/compare_experiments/
    __init__.py
    __main__.py
    cli.py
    schema.py
    discovery.py
    readers/
        config.py
        csv_metrics.py
        tensorboard.py
        optuna.py
        wandb_local.py
        checkpoints.py
    normalization.py
    comparability.py
    analysis/
        curves.py
        hyperparameters.py
        distributions.py
        ingredients.py
        aggregation.py
    reporting/
        json_report.py
        html_report.py
```

Add helpers only when their responsibility requires them. Keep checkpoint access lazy and optional: inspect selected metadata only when needed, and require an explicit request for tensor-level analysis. If checkpoint metadata is unavailable, report that limitation rather than inferring the selected state from the curve minimum. The existing `scripts/analize_exps/` notebooks remain historical references; this work does not rename or execute them.

### Inputs and maintained CLI

The main input is a list of experiment directories. Discover `trial_<number>`, shared experiment configuration and the external W&B/Optuna stores. Accept explicit paths for moved/exported campaigns.

Maintained command:

```bash
python -m scripts.analise_exp.compare_experiments \
  --experiments experiments/basic_v5/resnets_htuning experiments/basic_v5/dinov2_htuning_v1 \
  --wandb-root experiments/wandb \
  --optuna-journal experiments/journal.log \
  --metric val_loss --direction min \
  --output analysis_outputs/basic_v5_comparison
```

Initial controls cover metric/direction, optional reached/censored target, W&B trial scope, TensorBoard, gradients, raw bins and checkpoint metadata. Experiment, tensor and ingredient selection is available in HTML. Resolved defaults are recorded in JSON. Using `val_loss` in this example does not authorize pooling weighted and unweighted objectives.

If shared logs are missing, produce the analyses supported by the supplied experiment files and mark unavailable capabilities. Keep unknown Optuna states unknown; directory presence or a short curve alone cannot establish completion or pruning.

### Discovery, reading and reconciliation

Normalize `experiment -> numbered trial -> session -> observation`. Retain aliases and source-file identities separately.

| Input | Required handling |
| --- | --- |
| Typed configuration JSON | Decode as data; preserve symbols as strings without importing the saved training implementation |
| `trial_best` | Link to its numbered source; exclude it from trial population counts |
| CSV | Handle sparse rows and distinguish batch/epoch metrics; do not blanket-fill missing observations |
| TensorBoard | Read all event files, including empty and resumed segments, without default sampling of required scalar history |
| Optuna journal | Read/replay without modifying the source; retain states, distributions, objectives and available intermediate values |
| W&B `.wandb` | Stream records, reconstruct nested histogram keys, and retain original bin edges/counts and enclosing history clocks |
| Multiple sessions | Deduplicate identical events, detect re-executed/conflicting steps, and preserve the selected reconstruction policy and alternatives |
| Model/layer keys | Retain original keys; normalize W&B graph counters without inventing cross-architecture tensor correspondence |

Isolate the internal W&B file format behind a versioned adapter. Record producer and reader versions; initially validate against the audited W&B 0.28.0 streams and report unsupported/corrupt records explicitly.

Use explicit experiment configuration and trial identity for study mapping; tolerate historical path spelling only when the match is unambiguous. Validate selected parameters against configuration where available.

CSV, TensorBoard, W&B and Optuna serve complementary roles. Do not choose a universal winner by file type. Resolve values by their metric semantics, session lineage and recording stage. Surface unresolved conflicts and exclude affected calculations when no defensible reconstruction is available.

Keep epoch, optimizer step, W&B history step and wall time as distinct axes. W&B histogram clocks identify the enclosing history record and may not pinpoint the exact forward/backward capture. Do not create a zero-origin or initialization measurement when the first recorded histogram occurs after training started.

### Scalar analysis

For each trial, report:

- configuration and available execution status;
- `best_observed`, `selected_checkpoint`, `study_objective` and `last_observed` separately;
- observed epoch/step coverage and elapsed-time coverage where recoverable;
- best epoch, early-to-late change, late-window mean/median, slope and variability;
- normalized learning-curve area over a declared common interval;
- target-crossing time when an explicit target is supplied; mark unreached targets as censored;
- descriptive train/validation gap only when metric definitions and observations can be aligned.

Specify every formula, direction, window and missing-data policy in the result contract. An area over the learning curve is not classification ROC AUC. Do not extrapolate pruned trials to the final epoch.

Within an experiment, produce rankings only inside valid comparison groups, plus conditional hyperparameter associations and optionally Optuna importance estimates when sample size/coverage supports them. These describe the sampled search, not causal hyperparameter effects.

### Parameter-distribution analysis

The initial path reads `parameters/...` histograms already present in W&B. Preserve their numerical bins exactly in an optional raw-data sidecar/cache; compute statistics from the full selected observations.

Initial derived measures:

- recorded population count and bin support;
- midpoint estimates of mean, standard deviation, RMS and L2 norm;
- quantile-bin intervals, with any optional within-bin interpolation labeled;
- near-zero and tail mass with declared thresholds and crossing-bin uncertainty;
- early/late changes and temporal dispersion of these descriptors;
- distribution distance over a common numerical support, with its approximation policy recorded.

Bins can change between samples. Comparing same-index bucket counts is not a valid distance when their edges differ. Parameter histograms require no AMP-scale correction, but they do not retain coordinate identity. Do not present histogram distance as the norm of actual weight updates.

Aggregate corresponding tensors only within compatible model structures. Across architectures compare explicitly defined roles or blocks, normalized where appropriate; separate frozen and trainable parameters. Report whether a summary weights tensors equally or by their element count.

Existing `gradients/...` can be included as an optional descriptive view, carrying their scaled-backward/microbatch semantics. Unknown AMP scale prevents claims about absolute effective-gradient magnitudes, fixed vanishing/explosion thresholds and gradient-to-weight ratios. Missing frozen-backbone gradients are expected, not a training defect. Exact unscaled gradient logging is a separate extension, not a first-version prerequisite.

### Intra- and inter-experiment comparability

Construct comparison groups from data/split provenance, vocabulary identity/order, metric semantics, loss weighting, model/adaptation protocol, budget and selection policy.

The audit already supplies mandatory real cases: heterogeneous ResNet architectures, weighted/unweighted BCE, high pruning rates, duplicated `trial_best`, multiple sessions, truncated CSV histories and checkpoint selection every two epochs. Details and exact measurements stay in the [artifact audit](../implementation_details/experiment_artifacts.md).

For each aggregate curve show the number of contributing trials and the completion/pruning composition. Separate common-budget comparisons from full-trajectory descriptions. Adaptive HPO trials with different settings are not repeated-seed replicates; their spread is not seed-level confidence.

Optional per-ingredient comparison requires compatible label mapping and metric definitions. Do not synthesize missing AP/F1 from aggregate precision/recall, or treat intentionally absent metrics as zero. The generic flag does not change the separately owned [ingredient-learnability protocol](recognizable_ingredient_selection.md).

### JSON and HTML

Implemented JSON sections are `schema_version`, `generated_at`, `inputs`, `settings`, `provenance`, `experiments`, `intra_experiment`, `inter_experiment` and `limitations`; coverage and comparability cohorts are nested under their owning experiment.

Each calculated result carries its method, units, source references, observation count, interval, approximation status and missing-data reason. Use strict JSON: undefined numeric results become `null` with a reason, not NaN or fabricated zeroes. Full computed results belong in JSON; raw source histograms can remain in optional sidecars.

Generate HTML from this result object, without independent recalculation. Include an overview, configuration/ranking tables, trial and experiment scalar curves, parameter distribution/statistic views, and conditional per-ingredient views. Provide experiment/trial/layer/metric filters and make exclusions/coverage visible.

The implemented rendering approach is self-contained HTML with embedded data and selective chart rendering. Display downsampling must not change computed statistics. Measure output size and browser responsiveness on the actual campaigns; avoid embedding raw W&B logs or creating every possible chart at page load.

The current campaigns contain roughly 122 GiB of experiment artifacts plus 8.79 GiB of W&B streams. The initial implementation uses sequential streamed ingestion and loads checkpoint metadata only on explicit demand. Source-identity caching remains a scaling extension for repeated `--wandb-scope all` analyses.

## Ordered execution and verification

| Step | Deliverable and meaningful verification |
| --- | --- |
| P1 | Define schemas, metric selection/thresholds, label-key policy, legacy behavior, session reconciliation and analysis windows. Use the real audit cases to specify expected outcomes before implementing the readers |
| P2 | Implement the flag and epoch-level ingredient metrics. Test false/default, true, legacy loading, configuration round trips, subclass forwarding, label order, split/epoch reset and aggregation on hand-checkable batches where average batch F1 differs from full-epoch F1 |
| P3 | Build readers and reconciliation. Test aliases, missing stores, sparse CSVs, multiple events, empty sessions and conflicting steps; reproduce the DINOv2 trials 42/49/93 recovery cases without modifying source files |
| P4 | Validate extrema/selected-checkpoint distinctions, observed-window area, slopes and reached/censored targets on small analytic curves and existing completed/pruned trials |
| P5 | Validate exact preservation of exported bins/counts, midpoint estimates and quantile bounds on known histograms, changed bin edges and constant tensors; reproduce the 168-observation ResNet parameter example |
| P6 | Verify weighted/unweighted objective separation, incompatible vocabulary/metric handling, frozen parameter treatment and contributor counts as trials are pruned; optional ingredient data must not affect general-report availability |
| P7 | Validate strict JSON serialization, shared JSON/HTML values, required controls and chart containers, and embedded JavaScript syntax |
| P8 | Run the maintained CLI on both current target-v5 experiments, exercise missing-W&B/Optuna cases, verify input integrity and measure runtime/output size. Run a bounded opt-in/off logging smoke check through the canonical Lightning path, then document the actual CLI and synchronize current implementation contracts |

Use tiny synthetic fixtures for mathematical and parser edge cases; use selected existing artifacts for integration checks. No full campaign retraining is required for this plan. Do not copy large checkpoints or raw W&B archives into the test suite.

## Completion criteria

- [x] The flag is owned by the Lightning model, defaults to false, persists correctly and does not require changes to the internal torch-model interface.
- [x] Enabled per-ingredient metrics have declared definitions, correct epoch aggregation and stable label identity.
- [x] One command reads N experiment paths and produces JSON plus usable HTML.
- [x] Both current target-v5 campaigns are supported with alias, pruning, restart and checkpoint-selection semantics represented.
- [x] Parameter histograms produce numerical trajectories with original-bin provenance and labeled approximation.
- [x] Compatible intra/inter-experiment comparisons and contributor counts are correct.
- [x] Reports remain usable without optional per-ingredient, W&B or Optuna evidence.
- [x] No unsupported claims about exact tensor updates, unscaled gradients, missing metrics or seed-level uncertainty appear.
- [x] Tests, bounded smoke checks, HTML structure/script checks and documentation updates pass.
- [x] Existing experiment artifacts and unrelated working-tree changes remain intact.

## Retained extensions

The initial implementation resolves the logging set, names, schema, session policy and analysis defaults in the maintained implementation contract. Future changes to those semantics require a schema or contract revision.

Remote W&B API ingestion, automatic checkpoint inference, exact optimizer-update logging, gradient-vector alignment, new training campaigns and final test-set analysis are outside the first-version scope. They may be added later as explicit extensions. The main project priorities and final benchmark gates remain owned by [general_plan.md](../general_plan.md) and the [comparison methodology](../project_objective/model_comparison_methodology.md).

## Decision and change log

| Date | Decision or change |
| --- | --- |
| 2026-09-14 | Completed the local artifact feasibility audit and demonstrated raw numerical histogram extraction |
| 2026-09-14 | Made per-ingredient logging optional with default false |
| 2026-09-14 | Superseded the initial torch-model flag placement: the owner is the Lightning model |
| 2026-09-14 | Accepted recorded parameter distributions as the initial storage/analysis tradeoff; exact coordinate histories are not required |
| 2026-09-14 | Consolidated the recap and implementation sequence into this operational plan; earlier scratch planning is superseded, while audit scripts and generated evidence remain retained |
| 2026-09-14 | Completed the optional Lightning logging, local readers, quantitative analyses, cohort rules, JSON/HTML reporting, focused tests and a full target-v5 validation run |
