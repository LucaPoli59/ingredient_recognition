# Experiment comparison and per-ingredient logging

**Created:** 2026-09-14
**Last updated:** 2026-09-14

## Purpose and scope

This document defines the implemented contract for optional per-ingredient training metrics and the local N-experiment comparison command. It covers configuration, metric semantics, artifact reconciliation, quantitative analyses, output formats, and maintained limitations. The final benchmark methodology and test-set release gate remain owned by the project-objective documents and the general plan.

## Optional Lightning-model logging

BaseLGNM owns log_per_ingredient_metrics, with default False. BaseWithSchedulerLGNM forwards it explicitly, BaseLGNM.load_from_config treats a missing legacy field as false, and ExpConfig stores it under hyper_parameters. The canonical opt-in is:

~~~python
ExpConfig(hp_log_per_ingredient_metrics=True)
~~~

The flag is always retained in the Lightning module hyperparameters even when HPO restricts other logger-visible parameters through hparams_to_register. It does not alter the internal torch-model interface, output vocabulary, loss, aggregate metric configuration, test execution, or AMP behavior.

When enabled, the model allocates independent train, validation, and test MetricCollection states containing multilabel precision, recall, and F1 with:

- threshold 0.5 applied by TorchMetrics to logits;
- average="none", producing one value per output index;
- global split accumulation followed by one computation at epoch end;
- zero for undefined divisions under the validated TorchMetrics 1.3.2 behavior;
- no synthetic values when a split has no updates.

The stable key contract is:

~~~text
<split>_per_ingredient/<metric>/<label_index>
~~~

For example, val_per_ingredient/f1/42 is the validation F1 for output index 42. Names are resolved later from the serialized label-encoder class order. The model does not alphabetically reorder labels or place ingredient text in logger keys. AP is not part of this initial logging set because exact AP retains prediction history and has a materially different memory contract.

Old configurations without the flag are classified as unknown_legacy; an explicit false value is disabled; explicit true without observations is enabled_but_missing; and readable indexed series are available or available_legacy.

## Maintained comparison command

Run the comparison from the repository root:

~~~bash
python -m scripts.analise_exp.compare_experiments \
  --experiments experiments/basic_v5/resnets_htuning experiments/basic_v5/dinov2_htuning_v1 \
  --wandb-root experiments/wandb \
  --optuna-journal experiments/journal.log \
  --metric val_loss \
  --direction min \
  --wandb-scope best \
  --checkpoint-metadata \
  --output analysis_outputs/basic_v5_comparison
~~~

The command writes comparison.json and comparison.html. The analysis_outputs directory is ignored by Git because reports may embed full selected histogram trajectories and become large. Use --wandb-scope none for scalar-only analysis, best for the study-selected trial in each experiment, or all for every discovered trial. The all option can produce very large reports. The --include-gradients option enables the limited historical gradient view, and --preserve-raw-histograms embeds original edges and counts in addition to derived values.

The optional --target value reports the first reached coordinate or an explicitly censored result. TensorBoard ingestion is enabled by default and can be disabled with --no-tensorboard. Checkpoint metadata is lazy and enabled only with --checkpoint-metadata; state tensors are counted but not copied into the report.

## Discovery and read-only behavior

An experiment is a directory containing numbered trial_<n> directories. trial_best is matched to a numbered trial by CSV digest and excluded from population counts. Typed configuration JSON is decoded as data: recorded classes and functions remain strings and are never imported by the reader.

The readers apply these source-specific rules:

- sparse Lightning CSV cells remain absent;
- TensorBoard event files are read without scalar downsampling;
- CSV is selected at an optimizer step where both CSV and TensorBoard contain the requested metric;
- the latest TensorBoard event fills steps absent from a resumed CSV;
- conflicting TensorBoard restart values and CSV/TensorBoard disagreements are retained in conflicts;
- the Optuna journal is copied to a temporary directory before JournalStorage opens it;
- Optuna states, objectives, parameters, distributions, duration, and intermediate values are retained;
- W&B files are scanned with a read-only descriptor and their SHA-256 is checked before and after ingestion;
- multiple W&B sessions remain separate, with cross-session duplicates and conflicts reported instead of averaged.

Epoch, optimizer step, W&B history step, timestamp, and runtime remain distinct. Reconciled scalar curves use optimizer step when TensorBoard fills missing CSV history; CSV-only curves use epoch when present.

## Quantitative results

For every finite scalar curve the report includes the observed window, best observed value, last value, normalized trapezoidal learning-curve area, early and late means, late median, early-to-late change, late linear slope, and late population standard deviation. The Optuna objective remains distinct from the observed curve minimum.

Trial objectives are aggregated only for complete trials, or for trials with unknown status when no journal is supplied. Loss function, loss weighting, dataset field, metadata, and label-order identity define objective cohorts. Each cohort has its own best trial and objective distribution. When multiple cohorts exist, best_trial is null with a reason; study_selected_trial preserves the historical cross-cohort choice made by the original Optuna study without presenting it as a comparable best. Aggregate curves use exact recorded coordinates without interpolation and report contributor and state counts at every point.

Hyperparameter results use Spearman association for varying numeric parameters and count/mean/median objective summaries for categorical parameters. These are descriptive conditional HPO observations, not causal estimates or seed-level uncertainty.

Optional ingredient analysis reports label-index/name mapping, coverage, final-value summaries, and contributor-counted trajectories. General reports remain available when ingredient data is absent.

## W&B histogram analysis

The local reader is validated against the W&B 0.28.0 producer format. Graph-instance prefixes are removed while source-file and tensor-key lineage remain available. For each histogram it validates finite ordered edges, non-negative integral counts, and a positive population.

Derived measures include recorded count, support, midpoint estimates of mean, standard deviation, RMS and L2, containing-bin intervals for the 5th, 50th and 95th percentiles, bounded near-zero mass, first-to-last changes, and a normalized area between uniform-within-bin CDF estimates. Parameter trajectories use the enclosing W&B history clocks. The self-contained HTML exposes tensor selection and plots the RMS midpoint estimate over time.

Model parameters are not multiplied by the AMP gradient scale. Historical gradient histograms may describe scaled, accumulated, pre-clipping microbatch gradients. Histogram bins discard coordinate identity, so the report cannot recover exact per-weight histories, update vectors, update norms, successive-gradient cosine, or exact gradient-to-weight ratios.

## Report contract

Schema 1.0.0 is strict JSON and contains inputs, resolved settings, environment provenance, experiment records, intra-experiment summaries, inter-experiment cohort comparisons, and limitations. Non-finite numeric results become null; unavailable results carry a status or reason rather than a fabricated zero.

The HTML embeds the same result object and performs display rendering only. It provides experiment selection, trial tables, scalar curves, parameter trajectories and drift rankings, optional ingredient trajectories, inter-experiment data, coverage, and limitations. It does not recalculate analytical results independently.

## Verification

The maintained tests are [test_per_ingredient_logging.py](../../tests/test_per_ingredient_logging.py) and [test_experiment_comparison.py](../../tests/test_experiment_comparison.py). They cover configuration round trips, default/off behavior, full-epoch F1, empty splits, HPO persistence, a bounded CPU Lightning fit, data-only decoding, curve formulas, histogram formulas, resumed-history reconciliation, W&B session conflicts, objective-cohort separation, contributor counts, strict JSON, and HTML generation.

On 2026-09-14 the command completed against all 200 target-v5 trial directories, the matching Optuna studies, every TensorBoard event file, one study-selected W&B session per model family, and both selected checkpoint metadata files. It recovered 16, 40, and 16 unique validation points for resumed DINOv2 trials 42, 49, and 93, respectively, while retaining the trial-93 restart conflict. It selected historical study trials 77 and 61, decoded 62 and 178 parameter series, and left all source hashes unchanged.

## Maintained limitations

Remote W&B API ingestion, exact optimizer updates, unscaled gradient instrumentation, AP logging, automatic semantic correspondence between different architectures, interpolation across missing curve coordinates, causal HPO effects, repeated-seed uncertainty, and final test-set analysis are outside this version. The command supports exploratory and engineering comparison of current artifacts; it does not release the deferred final benchmark gate.

Related evidence and decisions are in [experiment_artifacts.md](experiment_artifacts.md), the completed [feature plan](../plans/experiment_comparison.md), [model comparison methodology](../project_objective/model_comparison_methodology.md), and [benchmark decisions](../project_objective/benchmark_decisions.md).
