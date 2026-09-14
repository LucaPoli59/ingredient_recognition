# Experiment artifacts and analysis limits

**Created:** 2026-09-14
**Last updated:** 2026-09-14

## Purpose and evidence boundary

This document owns the verified persistence and observability contract relevant to offline experiment comparison. It records the 2026-09-14 audit of `experiments/basic_v5`, together with the current producer code. It is not a final model ranking or a claim that the proposed JSON/HTML comparison application exists.

The reproducible [audit probe](../../src_scratches/experiment_comparison_audit/audit.py) and its [generated inventory](../../src_scratches/experiment_comparison_audit/inventory.json) cover all 200 numbered trial configurations, CSV files and TensorBoard inventories; the relevant Optuna studies replayed from a disposable journal copy; and one complete W&B session (`trial_72`) and one selected checkpoint per experiment. W&B histogram coverage was verified in those two sessions, not exhaustively across every session. No training, inference, remote synchronization or experiment rewrite was performed.

The reader environment and both sampled W&B producers use W&B 0.28.0; the reader also uses TensorBoard 2.20.0, Optuna 3.6.2, Lightning 2.6.1 and PyTorch 2.8.0+cu129. Source configuration and CSV hashes are retained in the inventory. The probe is a bounded exploratory tool, not a production reader for arbitrary historical formats.

## Observed campaigns

| Property | `resnets_htuning` | `dinov2_htuning_v1` |
| --- | --- | --- |
| Numbered trials | 100 | 100 |
| Optuna COMPLETE / PRUNED | 22 / 78 | 37 / 63 |
| Model configurations | 15 DummyBNModel, 28 ResnetLikeV1, 57 Resnet18 | 100 DinoV2B14 with frozen backbone |
| Weighted / unweighted loss | 22 / 78 | 23 / 77 |
| Configured epoch budget | 40 | 40 |
| `trial_best` alias | `trial_77` | `trial_61` |
| Local W&B session files | 103 | 105 |
| W&B bytes | 2,604,192,856 | 6,831,912,358 |

The saved configurations identify `ingredients_target_v5_metadata.json`, `ingredients_target`, cuisine category `all`, and 165 model outputs. Each experiment-level `hparam_config.json` contains an encoder with 165 classes and a 165-entry encode map; individual trial configurations have an empty encoder. The experiment-directory name alone is not evidence of dataset identity: future comparisons must inspect the actual configuration, vocabulary ordering and available split provenance. These are real target-v5 campaigns, despite older roadmap statements about historical runs generally being legacy evidence.

The ResNet campaign is heterogeneous, contrary to the assumption that every experiment contains only near-identical models. Use `hyper_parameters.torch_model.type` and its configuration as the model identity. Some ResNet configurations also retain a top-level `hyper_parameters.torch_model_type` pointing to DummyModel, which is not the nested model reconstructed by [BaseLGNM.load_from_config](../../src/lightning/lgn_models.py).

No `seed` field occurs in the audited trial configurations. This means a declared training seed cannot be recovered from those configurations; it does not prove that no seed was set elsewhere.

## Sources and recoverable information

| Source | Contents and use | Boundaries |
| --- | --- | --- |
| Experiment `hparam_config.json` | Shared configuration, encoder, sampler/pruner settings | Preserve serialized semantics and historical paths |
| Experiment `hparam_gen_config.json` | Search parameter names, categorical mappings, conditional configuration effects | Lambda placeholders do not encode numeric search bounds; read Optuna distributions |
| Trial `trial_config.json` | Effective selected model, loss, optimizer, learning rate, scheduler, augmentation and data settings | Decode as data without dynamically importing classes/functions |
| Trial `metrics.csv` | Sparse rows containing train/validation scalars, epoch, step, learning rate and momentum | Several rows per epoch; restarts can replace earlier CSV history |
| Trial `events.out.tfevents.*` | Scalar events with step and wall time; complementary restart segments | No histograms in any of the 200 audited numbered trial directories |
| `experiments/journal.log` | Study/trial identity, states, sampled parameters, distributions, objective, intermediate values and durations | Shared outside experiment folders; preserve study lineage and historical path spelling |
| `experiments/wandb/wandb/offline-run-*/run-*.wandb` | Local run records, scalar history, parameter/gradient histograms, timestamps and system records | Shared outside experiment folders; multiple sessions can represent one logical trial |
| `best_model.ckpt` and `checkpoints/*.ckpt` | Exact tensors at retained snapshots, epoch/step, optimizer/scheduler and AMP state | A few selected snapshots, not a full weight trajectory; sampled light checkpoints omit model/data/trainer hyperparameter dictionaries |

The two campaign directories occupy approximately 122 GiB, mostly checkpoints. Reading every checkpoint is unnecessary for scalar/histogram analysis. The 208 W&B files add about 8.79 GiB; a production reader should stream them and cache derived statistics, with lazy checkpoint access.

Producers are [training/htuning_exp.py](../../src/training/htuning_exp.py), [lightning/lgn_trainers.py](../../src/lightning/lgn_trainers.py), [lightning/custom_callbacks.py](../../src/lightning/custom_callbacks.py) and [lightning/lgn_models.py](../../src/lightning/lgn_models.py). The existing [config codec](../../src/commons/config_enc_dec.py) dynamically imports serialized symbols when decoding; a comparison-only reader does not need that behavior.

## Identity, resumption and selection

### Trial aliases and session identity

`save_best_trial` copies the chosen trial into `trial_best`. The audit verified identical CSV hashes between each alias and its numbered source. Counting the alias would duplicate an observation.

W&B session files cover every numbered trial, but ResNet trials 4 and 26 have respectively three and two sessions; DINOv2 trials 42, 49 and 93 have respectively two, two and four. Trial identity, training-session identity and remote-upload identity are distinct. [sync_wandb_runs.py](../../scripts/sync_wandb_runs.py) creates new remote IDs on synchronization, so remote run counts cannot define independent training trials.

### Missing and conflicting histories

DINOv2 trial 42's current CSV covers epochs 6–15; trial 49 covers 10–39; trial 93 covers 10–15. Earlier TensorBoard files retain the missing starts. The current journal also retains only 10, 30 and 6 intermediate validation values for those resumed trials, respectively, while the combined TensorBoard files recover 16, 40 and 16 unique validation steps. Journal intermediate values and recorded durations must therefore not be assumed to describe the entire pre-resume trajectory.

Trial 93 also contains two different validation losses at step 4124: approximately 0.1387317032 in an earlier session and 0.1387654990 after a subsequent restart. Another event file has no validation scalar at all. A reader must retain source/session provenance, distinguish duplicates from actual re-execution, and report conflicting steps. A policy that simply averages duplicate rows would invent a training history.

Pruned trials often have one more validation epoch than completed `train_loss_epoch` entries, because pruning interrupts before the final train-epoch logging. Missing values must remain missing.

### Observed minimum versus selected checkpoint

`OptunaTrainer` configures checkpoint consideration every two epochs and keeps two top checkpoints plus the last checkpoint during training. The objective returned by `_objective_wrapper` is the checkpoint callback's `best_model_score`, not necessarily the minimum over every validation measurement. A cross-check of all 59 COMPLETE trials reproduced their journal objectives from CSV validation values restricted to the every-two-epoch checkpoint cadence (absolute tolerance 1e-6).

For ResNet trial 77:

- the CSV minimum is 0.1293458193540573 at zero-based epoch 32;
- the Optuna objective is 0.12940621376037598;
- the copied best checkpoint is at zero-based epoch 33, global step 12750.

The sampled DINOv2 selected checkpoint is at epoch 35, step 13500. Its study objective is 0.1365121752023697. These values document selection semantics; they do not constitute a valid cross-family effectiveness conclusion.

A report must distinguish `best_observed`, `selected_checkpoint`, `study_objective` and `last_observed`. A pruned trial's terminal objective must not be mislabeled as a completed trial's best retained checkpoint.

## Scalar metric semantics

The observed CSVs contain `train_loss_step`, `train_loss_epoch`, `val_loss`, train/validation accuracy, precision, recall and Hamming distance, plus epoch, step and optimizer-specific learning-rate/momentum columns. They contain no saved AP/mAP, F1, per-label metric trajectories or prediction arrays. TensorBoard confirms scalar-only coverage.

The saved classification metrics request `average='weighted'`. Moreover, the current `_base_step` calls the MetricCollection on a batch and logs the returned tensors with epoch aggregation, rather than logging the Metric objects themselves. The recorded classification scalars therefore follow the batch-result aggregation path; they must not silently be relabeled as metrics computed once on the entire epoch's prediction set.

Consequences:

- The harmonic mean of recorded weighted precision and recall does not reconstruct micro F1 or weighted F1.
- AP/mAP, global F1, calibration, per-label breakdowns and paired sample bootstraps require saved predictions/targets or a separately authorized checkpoint-evaluation stage.
- Training metrics use the training preprocessing and evolving model, so a train/validation gap is a descriptive diagnostic with those qualifications.
- `weighted_loss=True` supplies `pos_weight` to BCEWithLogitsLoss for both train and validation. Its numeric loss scale differs from unweighted BCE. The shared study objective already mixes these objectives, affecting both selection and adaptive search/pruning. An analysis reader can expose this limitation but cannot retroactively undo it.
- Hyperparameter associations and Optuna importance estimates describe the sampled search under its conditional space and censoring. They are not causal effects or seed-level uncertainty.

## Parameter and gradient histograms

### Verified extraction

Both sampled local files were fully decoded as protobuf record streams. W&B stores histograms as counts and bin edges, sometimes as separate nested-key history entries rather than one JSON object. This is consistent with its documented [histogram representation and display](https://docs.wandb.ai/models/track/log/media#histograms).

| Sample | Parameter tensors | Gradient tensors | Samples per parameter / gradient series |
| --- | --- | --- | --- |
| ResNet `trial_72` | 62 | 62 | 168 / 150 |
| DINOv2 `trial_72` | 178 | 2 | 168 / 150 |

Both samples span zero-based epochs 0–39. ResNet histograms use 64 bins. DINOv2 also has 168 one-bin histograms for a constant tensor; a valid reader must accept degenerate ranges.

The 178 DINOv2 parameter series include the frozen backbone, while the two gradient series are the linear head's weight and bias. Absence of gradients in a frozen backbone is expected, not evidence of vanishing gradients.

The current trainer sets `watch(log='all', log_freq=2*log_every_n_steps)`, giving a nominal frequency of 100 hook calls here. W&B 0.28.0 records parameters in a forward hook, including validation forwards, and gradients in a backward hook. Their sampling clocks therefore differ. Logged epoch and `trainer/global_step` belong to the enclosing committed history record; they are not evidence that every tensor was captured immediately before the same optimizer update.

Names such as `gradients/graph_46model.layer3.0.bn1.bias` include a W&B graph counter. Strip that counter for a canonical layer identifier while retaining the original key. A ResNet stage and a transformer block have no automatic tensor-to-tensor correspondence.

### What the bins allow

For bin counts `n_i`, bin midpoints `m_i` and total count `N = sum(n_i)`, use these midpoint estimates:

- mean: `mu_hat = sum(n_i * m_i) / N`;
- RMS: `rms_hat = sqrt(sum(n_i * m_i**2) / N)`;
- L2 norm: `l2_hat = sqrt(N) * rms_hat`;
- variance: `var_hat = sum(n_i * (m_i - mu_hat)**2) / N`.

These are estimates of the underlying tensors, not exact moments. Quantiles can be bounded by their containing bins; any within-bin interpolation is an additional assumption. Near-zero/tail mass needs a declared threshold and bounds or an interpolation policy for bins crossing it. Bin widths change over time, so distribution distances require a common value support/CDF treatment rather than comparing same-index counts. Distance or entropy changes under different binning must not be interpreted without this qualification.

Useful descriptive outcomes include distribution widening/narrowing, weight-scale drift, changes in quantile ranges, and approximate distribution distances. For example, in the sampled ResNet trial 72, the midpoint RMS estimate for `parameters/graph_46model.layer2.0.conv1.weight` decreases from 0.0412464 at the first recorded sample to 0.0178255 at the last, approximately 56.8%. This is a measurable change in weight scale, not evidence that the layer became more useful or that generalization improved. Gradients need the additional qualifications below. An observed distribution shift is not an exact change in the same coordinates.

### Mixed precision and accumulation

The audited trainer selects `16-mixed`. In Lightning 2.6.1, `MixedPrecision.pre_backward` scales the loss, while `optimizer_step` unscales gradients after the closure/backward. W&B 0.28.0's `TorchHistory._hook_variable_gradient_stats` uses `parameter.register_hook` during backward. Thus these gradient histograms capture scaled backward gradients, before optimizer-time unscaling and any clipping.

Both inspected checkpoints contain `MixedPrecision.scale=4194304`. No scale history is present among the scalar keys of either sampled W&B session. A scale stored at one checkpoint cannot be applied to the entire earlier gradient trajectory. The observed expansion of histogram magnitudes must not by itself be diagnosed as exploding gradients.

With gradient accumulation, a backward hook can also describe a microbatch contribution rather than the final accumulated optimizer gradient. W&B's histogram code removes non-finite values and skips tensors without finite values, so a missing histogram or finite histogram does not establish absence of numerical failures.

A future exact diagnostic stream should record gradients after unscaling and accumulation, before clipping, with an explicit hook-stage field; optionally record post-clipping statistics separately. Log AMP scale, skipped updates, effective batch/accumulation, finite-value counts, and consistent optimizer-step/epoch/session identifiers. This is an instrumentation recommendation, not an implemented change. The upstream [Lightning AMP implementation](https://github.com/Lightning-AI/pytorch-lightning/blob/2.6.1/src/lightning/pytorch/plugins/precision/amp.py) documents the relevant execution order.

### What cannot be reconstructed

Histograms discard coordinate identity. They cannot recover exact per-weight trajectories, coordinate-wise update vectors, gradient cosines across steps, exact gradient/weight alignment, or update-to-weight ratios. A permutation can preserve the histogram while changing every coordinate.

Exact tensor differences can be computed only between retained, compatible checkpoints. The sparse, performance-selected snapshot set cannot reconstruct every update or the distance from initialization when the initial state is absent.

## Feasibility conclusion and implementation boundary

An offline JSON plus interactive HTML comparison is feasible from local artifacts. Scalar analysis can cover all trials now; W&B distribution analysis is demonstrated for both families and needs robust multi-session ingestion before general use. Exact optimizer dynamics and newly computed prediction metrics require separate instrumentation or evaluation.

The implementation should declare what each result measures, its unit/granularity, source coverage, comparison cohort, and whether it is recorded, estimated, unavailable or reconstructed. Aggregate curves must show the number of contributing trials at each point and separate completion/pruning cohorts. Common-budget comparisons and full-trajectory comparisons answer different questions.

The [scratch feasibility handoff](../../src_scratches/experiment_comparison_audit/README.md) contains the proposed module layout and a staged implementation boundary. The [general plan](../general_plan.md#7-results-comparison) retains the final benchmark gate; this preparatory audit does not release final comparison. Binding evaluation choices remain in [model_comparison_methodology.md](../project_objective/model_comparison_methodology.md) and [benchmark_decisions.md](../project_objective/benchmark_decisions.md).
