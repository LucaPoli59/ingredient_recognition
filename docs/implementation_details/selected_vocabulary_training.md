# Selected-vocabulary training launchers

**Created:** 2026-10-07
**Last updated:** 2026-10-07

## Purpose and scope

The user-requested parallel follow-up retrains the two existing historical
baseline protocols on the shared D6 vocabulary while Phase 5 proceeds.
[The launcher folder](../../scripts/launch_exps/selected_ingredients/README.md)
is also the intended home of future selected-task launchers for qualified new
models. No new-model launcher is supplied before its Phase 5 acceptance.

These are **historical-configuration transfers**, not another HPO study or
completed final benchmark. They follow the transfer principle in the
[methodology](../project_objective/model_comparison_methodology.md), but the
source studies predate the final metric/selection contract. Retraining does not
turn them into the future `H_base(m)` automatically.

## Configuration contract

[The canonical orchestration module](../../src/training/selected_vocab.py)
checks the exact source configuration SHA-256, loads it through `ExpConfig`,
and calls `load_datamodule` and `model_training`. It neither imports the
user-edited one-shot entry point nor restores the source model's trained weights.
Each run constructs the pretrained backbone and a fresh 59-output head.

| Launcher | Reviewed full-task source | Preserved protocol |
| --- | --- | --- |
| `train_resnet.py` | `basic_v5/resnets_htuning/trial_77` | Pretrained ResNet18, complete adaptation, hard augmentation, Adam, LR `3.133505394083749e-5`, decay `2.907966751604963e-4`, cosine warm restarts (`T_0=5`, `T_mult=2`, `eta_min=1e-6`). |
| `train_dinov2.py` | `basic_v5/dinov2_htuning_v1/trial_61` | DINOv2-B/14 with register tokens and frozen backbone, native DINO transforms, AdamW, LR `3.2107930647002535e-4`, decay `7.717276829911085e-5`, plateau schedule (`factor=0.2`, `patience=3`, `min_lr=1e-6`). |

Both use unweighted BCE, 40 epochs, logical batch 128 and validation every
two epochs, without early stopping or pruning. Mixed precision `16-mixed`
matches the source `OptunaTrainer` behavior; the replacement `BaseTrainer`
retains full checkpoints. Seed 42 is explicit for these fresh runs; historical
seed equivalence is not asserted. Per-ingredient precision/recall/F1 logging is
enabled as additional diagnostics, without changing model selection from
`val_loss` to the future benchmark metrics.

The source config hashes are pinned in `PRESETS`, rather than inferred from
the mutable `trial_best` alias. The
[reviewed historical result](../experiment_results/basic_v5_resnet_dinov2.md)
owns why trials 77 and 61 are the retained anchors.

## Vocabulary, batching and persistence

The explicit [P7 projection](ingredient_selection.md#p7-runtime-projection)
keeps the original v5 split membership, record order and all-zero projected
targets. It preserves the saved 59-label order, base-column indices and approved
artifact identity; the global 165-label default is unchanged. Canonical data
preparation reads/encodes test metadata eagerly. The launcher performs only
training and validation; test/predict loader calls are denied, and no predictive
test evaluation is included.

Default physical batches are ResNet 128 / accumulation 1 and DINO 32 /
accumulation 4. These are requested settings, **not new measured CUDA capacity**.
The current DINO implementation has no active batch cap, despite an older
launcher comment claiming 32. The new runner scopes the cap to the launch and
validates the actual physical batch/Lightning accumulation at fit start.
`--physical-batch` accepts only exact divisors of 128 and is persisted; changing
it requires a new run name. For ResNet, smaller micro-batches also change
BatchNorm behavior. Legacy accumulation weights the final incomplete group in
the usual Lightning way; this is not the Phase 5 exact-tail batching contract.

Outputs live under `experiments/selected_v5/<run_name>/trial_0/`:

- `launch_config.json`: complete selected-task `ExpConfig`;
- full checkpoints and canonical CSV/TensorBoard/offline W&B logs;
- after success, `best_model.ckpt`, selected using validation loss;
- parent `launch_manifest.json`: transferred source/hash, seed, precision,
  requested batching, exact saved config, base revision, selected runtime-source
  hashes, runtime versions, device, attempts and status;
- parent `trainer_scratch/`: isolated from the canonical shared scratch cleanup.

Every full checkpoint additionally stores `selected_vocab_launch`. Explicit
`--resume` requires the same launch contract, unchanged inventoried runtime
sources and a matching selected-task `last.ckpt`. It cannot resume the full
165-label source checkpoint, a completed run, a different encoder, or altered
settings. A failure before the first checkpoint requires a fresh run name.
Existing directories are never silently converted, resumed or overwritten.

The base revision and source inventory are bounded provenance, not a complete
immutable snapshot of third-party dependencies. Legacy ResNet's `DEFAULT`
weights and DINO's Torch Hub retrieval are preserved rather than promoted to
the stricter Phase 5 artifact contract; initial execution may retrieve cached
or remote pretrained resources. Run the scripts sequentially on the development
GPU. The remaining Phase 6/7 gates, paired full-model output projection,
random-vocabulary controls, AP/calibration implementation and final test
evaluation remain separate work.

## Verification

[Focused tests](../../tests/test_selected_vocab_launcher.py) cover configuration
transfer, exact selected order, config round trips, no-training dry runs,
invalid batch/name/source rejection, runtime batch guards, checkpoint/resume
rejection and a mocked canonical launch with scoped-cap restoration. Real
configuration dry runs for both retained sources pass. All 10 new tests and
39 focused existing regression tests pass. The existing projection/logging
regressions include tiny synthetic CPU fits and checkpoint restoration; these
are not ResNet/DINO experiment runs. No campaign, pretrained download, predictive
test evaluation or CUDA capacity probe was performed for this change.

See the [folder README](../../scripts/launch_exps/selected_ingredients/README.md)
for commands, preview, restart and output navigation.
