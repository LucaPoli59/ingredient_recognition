# Phase 6 validation prior baselines

**Created:** 2026-10-18
**Last updated:** 2026-10-18
**Status:** Reviewed validation-only controls; operating points remain provisional

## Scope and provenance

These are the non-neural controls for the first [Phase 6-D1](../project_objective/model_comparison_methodology.md#phase-6-d1--finite-training-and-comparison-protocol) release, `experiments/benchmark_v1/phase62-20261018-v1/`. They use 47,965 training records to fit priors and score all 5,996 validation records. The [maintained implementation](../../src/training/benchmark_baselines.py) reads only train/validation metadata and does not open dataset images or test metadata. The five-model training campaign remains in progress; this record contains no neural-model result.

The exact full165 class order and selected59 base indices come from the frozen D6 projection. No rows are removed. `baselines/validation_controls.npz` preserves raw prior probabilities, ordered targets, recipe IDs, image names and base indices; `baseline_summary.json` records metrics, formulas, source and metadata hashes. Provenance is Git base `fae43019918c990206ef49bdf6db13956b0ae14d` plus the exact saved source hashes, since the 6.2 additions are uncommitted.

| Input/artifact | SHA-256 |
| --- | --- |
| Frozen train metadata | `30fa8d1aec04ca951c5a498f4c08c7f458b8bdfa754dd3a1fb839c9d898c221b` |
| Frozen validation metadata | `67ea919607e34ef249553b4fcf9c2851e4dc341a064a23bd405175c83f8ce3f6` |
| Phase 6 policy | `4a5fe311e083594de1d872cd3eea3701e8bfa641bfa7753c9b8f3d40d45d756c` |
| Ordered validation-control archive | `cc791c21c7953fe43874e8a9a559f70ff3b689b1014a9b552132b79a378dcb34` |

## Definitions

- **All-negative:** zero probability scores and empty binary predictions at 0.5.
- **Train prevalence:** each label's empirical training frequency, repeated for every validation image.
- **Train cuisine:** `(positive_train_records_in_cuisine + 1) / (train_records_in_cuisine + 2)`. Unknown cuisines fall back to the unsmoothed global train prevalence. This preserves the Laplace-smoothed formula in the [retained selector control implementation](../../src/ingredient_selection/analysis.py); cuisine normalization is shared with its data helper.

Metrics use the complete-validation [benchmark aggregation](../../src/training/benchmark_protocol.py). Finite logits are derived by clipping probabilities to [1e-12,1-1e-12] before conversion; the archive retains the original probabilities, including exact zeros. Every label has both validation outcomes in these two tasks. The current F1 column uses the fixed 0.5 probability threshold; Phase 6.5 owns later operating-point fitting. The fixed all-negative control retains its defining empty predictions.

## Results

| Task | Control | Label-macro AP | Micro F1 at 0.5 |
| --- | --- | ---: | ---: |
| Full165 | All-negative | 0.046751 | 0.000000 |
| Full165 | Train prevalence | 0.046751 | 0.143697 |
| Full165 | Train cuisine | 0.126409 | 0.239252 |
| Selected59 | All-negative | 0.091499 | 0.000000 |
| Selected59 | Train prevalence | 0.091499 | 0.195699 |
| Selected59 | Train cuisine | 0.230984 | 0.315012 |

Constant per-label scores tie every validation image, so both constant controls have AP equal to each label's validation prevalence and therefore identical macro AP. The all-negative control's AP is consequently nonzero even though its binary F1 is zero. Different label-wise constant probabilities can change micro F1 at 0.5 while leaving per-label ranking unchanged.

Cuisine-conditioned scores vary between recipe groups and supply a stronger contextual reference. They use ground-truth cuisine metadata and must be identified as a metadata-assisted diagnostic alongside image-only models. The selected task has a different label population; its larger AP is not an observed training gain or evidence that label reduction improved a model.

## Verification and limitations

The focused [execution tests](../../tests/test_benchmark_execution.py) verify train-only fitting, unseen-cuisine fallback, metadata hash rejection and train/validation-only loading. They also test sequential prediction export, class/target order, all-zero-row retention and restoration of the module's original mode/device. All 291 repository tests pass at this checkpoint; the pre-existing real test-split vocabulary check remains metadata-only. No predictive test evaluation, calibration or uncertainty analysis was performed here.

These baselines are validation evidence used during model development. Frozen operating points and final held-out comparisons belong to 6.5/Phase 7. The [training plan](../plans/training_hyperparameter_tuning.md) owns campaign status and the first-trial cost-review boundary; the [runtime contract](../implementation_details/benchmark_training.md) owns execution and artifacts.
