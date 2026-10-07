# Shared selected-ingredient experiment launchers

**Created:** 2026-10-07
**Last updated:** 2026-10-07

This folder contains rerunnable selected-task launchers, separate from the
ingredient-selection instrument. Add qualified new-model launchers here after
their Phase 5 gates, without conflating historical transfer with final benchmark
training. The maintained contract and limitations are in
[selected_vocabulary_training.md](../../../docs/implementation_details/selected_vocabulary_training.md).

## Launch from the main WSL repository

Use the `wsl_image_pytorch` interpreter, with this repository as working
directory. The scripts also support absolute-path invocation. Start one model
at a time on the development GPU:

```bash
python scripts/launch_exps/selected_ingredients/train_resnet.py
python scripts/launch_exps/selected_ingredients/train_dinov2.py
```

Each command retrains from fresh pretrained initialization on the same 59-label
`ingredients_selected_v5_d6_v1` projection, for 40 epochs and logical batch 128.
No source checkpoint is fine-tuned, no HPO is started, and no test evaluation is
performed. ResNet uses trial 77's hyperparameters with complete adaptation;
DINO uses trial 61's settings with a frozen backbone.

Preview without preparing data, constructing models or starting training:

```bash
python scripts/launch_exps/selected_ingredients/train_resnet.py --dry-run
python scripts/launch_exps/selected_ingredients/train_dinov2.py --dry-run
```

Outputs:

```text
experiments/selected_v5/
├── resnet18_from_full_trial77_d6_v1/
└── dinov2_b14_lp_from_full_trial61_d6_v1/
    ├── launch_manifest.json
    ├── trainer_scratch/
    └── trial_0/  # launch_config.json, checkpoints, metrics, best_model.ckpt
```

## Resource settings and explicit restart

Defaults: eight loader workers; mixed precision; ResNet physical batch 128,
DINO physical batch 32 and gradient accumulation 4. No new memory measurement
is claimed. To request a smaller micro-batch while preserving logical batch 128:

```bash
python scripts/launch_exps/selected_ingredients/train_resnet.py \
  --run-name resnet18_d6_physical64 --physical-batch 64 --workers 4
```

Only exact divisors of 128 are accepted. Smaller ResNet micro-batches affect
BatchNorm; these resource changes are recorded rather than claimed equivalent.
There is no automatic OOM retry or automatic resume. After interruption, use
the exact original arguments plus `--resume`, for example:

```bash
python scripts/launch_exps/selected_ingredients/train_resnet.py \
  --run-name resnet18_d6_physical64 --physical-batch 64 --workers 4 --resume
```

Resume requires that run's matching `trial_0/checkpoints/last.ckpt` and unchanged
inventoried runtime sources. A completed run cannot be resumed. For a fresh
repeat use a different `--run-name`; no existing experiment is overwritten.
Pretrained resources may be loaded from cache or downloaded at actual launch.

## Files

- [`train_resnet.py`](train_resnet.py): thin ResNet18 entry point.
- [`train_dinov2.py`](train_dinov2.py): thin frozen DINOv2-B/14 entry point.
- [`src/training/selected_vocab.py`](../../../src/training/selected_vocab.py):
  shared canonical configuration, provenance, guards and orchestration.

Future launchers must use one frozen shared vocabulary and canonical training
APIs; do not copy the selector's 384-pixel protocol or silently reuse a historical
configuration for a new architecture.
