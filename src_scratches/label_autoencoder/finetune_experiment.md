# Backbone fine-tuning with a frozen or trainable AE32 decoder

**Created:** 2026-10-22

**Last updated:** 2026-10-22

**Status:** Implementation prepared; real training has not been launched.
Execution is deferred until GPU capacity is available.

## Question and controlled comparison

The [frozen-feature pilot](image_experiment.md) favored a direct ingredient head.
This follow-up asks whether adapting the visual backbone improves AE32 prediction,
and whether adapting the decoder helps image prediction while changing its
original ingredient reconstruction behavior. This is a scratch experiment,
separate from the active benchmark and its adopted training protocol.

The [configuration](configs/backbone_finetune.json) defines three sequential arms
on all 47,965 train and 5,996 validation images with all 165 labels:

| Arm | Trainable components | Frozen components | Objective |
| --- | --- | --- | --- |
| `direct` | ResNet-18 and 512 -> 165 projection | Feature normalization | Ingredient BCE |
| `ae32_frozen_decoder` | ResNet-18 and 512 -> 32 projection | AE decoder, teacher encoder, normalizers | Decoded ingredient BCE + 0.1 times standardized latent MSE |
| `ae32_trainable_decoder` | ResNet-18, 512 -> 32 projection and AE decoder | Teacher encoder, normalizers | Same as frozen-decoder arm |

Both AE arms start from identical weights and buffers, receive identical epoch
shuffles and use the same objective/settings. Decoder trainability is the changed
factor. Gradients pass through the frozen decoder to update the visual model.
The teacher encoder never receives gradients or optimizer updates. The pure-MSE
image objective is not included in this follow-up.

## Initialization and training protocol

This continues the completed frozen-feature pilot, rather than initializing fresh
heads. All arms start from the same ImageNet-1K V1 ResNet-18 weights. The direct
arm starts from `direct.pt`; both AE arms start from the same `ae32_aligned.pt`
in `outputs/image_heads_20261022_seed42/`. Its decoder is checked bitwise equal
to the original ingredient AE32 decoder. No weights come from the active benchmark.

Feature and latent mean/std remain the train-fitted buffers from those heads.
The fixed teacher supplies code targets; the latent coordinate system stays fixed
even when the decoder adapts. Teacher normalization is rechecked before execution,
allowing small floating-point differences while retaining the saved normalizers.

The direct and AE heads have different pretraining histories and capacities:
their source checkpoints were selected at epochs 10 and 49, and the AE additionally
used ingredient-only pretraining. This is not an equal-total-compute comparison
or a fresh experiment independent of previous validation use. Epoch-zero image
metrics are recorded under the new runtime before any update.

| Setting | Prepared value |
| --- | --- |
| Optimizer | Adam; default betas/epsilon, no weight decay |
| Backbone / projection / decoder learning rate | 0.0001 / 0.001 / 0.0001 |
| Budget | Maximum 30 epochs, patience 7, minimum improvement 0.00001 |
| Batch / workers / CPU threads | 32 / 2 / 2 |
| Seed | 42 |
| Precision | Float32, no AMP or TF32; deterministic algorithms requested |
| Backbone adaptation | All parameters, including BatchNorm affine parameters; running statistics update during training |
| Preprocessing | Existing RGB fit/pad 224 and ImageNet normalization; no random augmentation |
| Checkpoint selection | Lowest validation ingredient BCE for every arm, including epoch zero |

Epoch zero remains eligible so an unsuccessful fine-tuning attempt can retain the
warm start; a selected epoch zero must be reported as no selected improvement.
Source checkpoints are never overwritten. No scheduler, search or automatic retry
is configured.

## Measurements and decoder retention

Image evaluation includes macro AP, micro AP, F1 at 0.5 and train-frequency
slices. Primary threshold-selected micro F1 searches every distinct validation
score, with the smallest threshold on ties. This policy is specified before the
new experiment, following the earlier boundary diagnostic. The old 0.05-0.95 grid
remains secondary; `*_per_label_grid.csv` explicitly uses its threshold.

Both AE arms report standardized code MSE and variance-weighted code R-squared.
The decoder is also evaluated on fixed codes from the true validation ingredients
before training and at the selected checkpoint. Record reconstruction BCE, macro
AP and F1 at 0.5, and relative L2 change in decoder parameters. The frozen arm must
retain its decoder exactly, including at the final state before checkpoint reload.

These reconstruction diagnostics are separate from image prediction and do not
select the checkpoint. Preserving reconstruction alone does not establish better
image generalization; losing reconstruction alone does not establish worse image
generalization. There is no extra reconstruction-loss regularizer in the trainable
arm, so this first comparison tests decoder adaptation at the stated learning rate.

Model/threshold selection uses validation. Test metadata/images are never opened.
One seed and adaptive validation give exploratory evidence, not an independent
test estimate or a stability/significance claim. A randomly initialized bottleneck
remains a separate control for attributing any benefit to AE pretraining.

## Preparation and explicit future execution

Use the existing WSL `wsl_image_pytorch` environment from the repository root.
The default command validates metadata, checkpoint hashes, label order, image
inventories and preprocessing without reading image bytes, evaluating the backbone,
creating a CUDA context or starting training:

```bash
python -m src_scratches.label_autoencoder.finetune
```

Optional `--plan <new-json-path>` saves a `prepared_not_started` record and refuses
overwriting. JSON paths are relative to the repository root except absolute and
`~` paths. The source checkpoints and cache image inventories are local ignored
artifacts required by this continuation.

The following is the future execution command; it has not been executed on the
real dataset during preparation:

```bash
python -u -m src_scratches.label_autoencoder.finetune \
  --config src_scratches/label_autoencoder/configs/backbone_finetune.json \
  --execute \
  --output src_scratches/label_autoencoder/outputs/backbone_finetune_seed42
```

Execution requires explicit `--execute` and a new output directory. It checks
physical GPU 0 with three driver samples: at least 6,144 MiB free and at most 10%
utilization each. `CUDA_VISIBLE_DEVICES` must be unset or `0`, matching the checked
GPU. Busy/missing telemetry fails immediately. There is no waiting daemon, CPU
fallback or scheduled launch. This point-in-time guard is not a reservation or
a measured memory fit; the actual CUDA path, throughput and memory use are untested.

Preparation validates the original image inventory; every future dataset read
checks actual bytes against its saved SHA-256. Changed/missing images abort
instead of silently changing the population. No feature cache is recomputed,
although backbone training processes the images every epoch.

Each improvement saves a self-contained best checkpoint with backbone,
head/decoder, normalizers, label order and transform. Histories persist each
epoch, and summaries distinguish running/complete/interrupted/failed.
`predict_image()` restores offline without original AE or ImageNet files.
Optimizer resume is not implemented: interrupted outputs are retained, and a
fresh run requires a new directory.

## Implementation and verification

- [finetune.py](finetune.py): explicit launcher, training loop, parameter groups,
  selection, diagnostics and offline image-only inference.
- [finetune_data.py](finetune_data.py): config/provenance checks and image dataset.
- [image_models.py](image_models.py): shared model definitions extracted without
  numerical changes. Importing leaves CUDA visibility unchanged; the older CPU
  launchers retain their own isolation. New manifests include the shared source hash.
- [test_finetune.py](test_finetune.py): CPU synthetic tests for launch isolation,
  integrity failures, matched initialization, gradient flow, parameter groups,
  decoder retention, epoch-zero selection, all-arm fitting, offline reload and
  the sequential launcher with a synthetic backend.

Earlier source hashes identify the original completed runs; this refactor does
not rewrite their artifacts or results.

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest src_scratches.label_autoencoder.test_finetune \
  src_scratches.label_autoencoder.test_image_features \
  src_scratches.label_autoencoder.test_image_heads \
  src_scratches.label_autoencoder.test_compare -v
```

Verification on 2026-10-22: all 26 tests passed (8 new and 18 previous).
The real-file preparation also passed for 47,965 train rows, 5,996 validation
rows and 165 labels, with CUDA uninitialized. Its local
[preparation record](outputs/backbone_finetune_preparation_20261022.json) is
marked `prepared_not_started`. Extracted shared model definitions were checked
AST-identical to their definitions in the preceding commit.

Only tiny synthetic training is used by the tests. No real-cohort training,
GPU allocation or change to the running main training is part of preparation.
The active benchmark, defaults and general project plan keep their existing scope.
