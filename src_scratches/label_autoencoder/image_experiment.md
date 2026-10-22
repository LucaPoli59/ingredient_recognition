# Predicting the AE32 code from frozen image features

**Created:** 2026-10-22

**Last updated:** 2026-10-22

**Status:** Full-cohort CPU pilot completed; direct head preferred within this exploratory comparison.

The separate [backbone fine-tuning follow-up](finetune_experiment.md) is prepared
with frozen/trainable decoder arms; it has not been launched. The results below
remain those of the completed frozen-feature pilot.

## Question and comparison boundary

Does the [ingredient-only AE32 representation](experiment.md) help predict all
165 ingredients from an image, compared with a direct head? This exploratory
experiment uses all 47,965 train and 5,996 validation rows from v5. All three
learned heads receive exactly the same frozen image features. No test split,
cuisine or flavor features are used, and no target enters image inference.

The image encoder is torchvision ResNet-18 with the existing local
`resnet18-f37072fd.pth` ImageNet-1K V1 weights. It is frozen in evaluation mode,
with its classifier replaced by identity to expose 512 pooled features. This
choice makes the first full-cohort pilot feasible on spare CPU capacity. It does
not reproduce or rank the active end-to-end benchmark models.

Preprocessing reuses `ExperimentalFitPad224` with flip probability zero: RGB,
full-frame bilinear fit/pad to 224 pixels and ImageNet normalization. There is no
random augmentation. CPU channels-last execution changes memory layout, not the
weights or intended transform. A given image feature is extracted once and used
by every head. All feature normalization statistics come from train only.

## Heads and objectives

Let `h(x)` be the frozen 512-dimensional image feature. Standardize each feature
coordinate with train mean and standard deviation. Let `E(y)` be the existing
AE32 ingredient encoder and `D(z)` its decoder. Standardize the teacher code
using the train codes' mean and standard deviation; floor both standard deviations
at 0.000001 to handle constant coordinates. Predictions are de-standardized before
passing to the decoder.

| Arm | Trainable image head | Training objective | Trainable / frozen head parameters |
| --- | --- | --- | --- |
| `direct` | Linear 512 -> 165 ingredient logits | Unweighted ingredient BCE | 84,645 / 0 |
| `ae32_aligned` | Linear 512 -> 32 standardized coordinates, then frozen `D` | Unweighted decoded ingredient BCE + 0.1 × standardized latent MSE | 16,416 / 25,509 |
| `ae32_mse` | Same code projection and frozen decoder | Standardized latent MSE only | 16,416 / 25,509 |

The two AE arms have identical initial projection parameters, seed and batch
ordering. Their projection bias starts at zero, representing the train-mean
code, with projection weights drawn from N(0, 0.01²). The direct head uses PyTorch
default projection weights and train log-odds bias. The latent coefficient 0.1 is
fixed before observing results; there is no coefficient or learning-rate search.

The MSE-only control answers the literal code-regression question. The aligned
arm checks whether explicitly optimizing the final decoded ingredient task
helps, since small latent errors need not imply good ingredient predictions.
The teacher encoder is used only to form train supervision and validation
diagnostic targets. The decoder weights are checked unchanged after fitting;
the ingredient encoder and image backbone are never trained in this experiment.

For inference, both AE heads are `image -> h -> projection -> D -> 165 scores`.
The saved head contains its decoder and normalizers; neither ingredient inputs
nor the original teacher encoder are needed. `predict_image()` also returns the
32-dimensional unstandardized code for these two arms.

All learned arms use Adam at 0.001, batch size 512, seed 42, maximum 100 epochs,
patience 10 and minimum improvement 0.00001. They select the checkpoint with the
lowest validation **ingredient BCE**, including the MSE-only arm, so model
selection targets the same downstream task. This is a fixed-setting pilot, not
a claim that any arm received its optimal hyperparameters. Parameter counts
differ, and the AE arms additionally benefit from the earlier label pretraining
(81 epochs, best epoch 81). The experiment does not isolate pretraining from
head capacity or match total compute.

## Evaluation and isolation

The code reuses the Stage A metric implementation and threshold grid in
[compare.py](compare.py). Report macro AP with micro F1 at fixed 0.5 and at a
validation-selected global threshold. Include the constant train-prevalence
baseline, per-label AP/recall and train-frequency terciles. The AE heads also
report standardized latent MSE, the train-mean-code baseline MSE and
variance-weighted latent R². The validation mean in the R² denominator is used
only to score predictions, not to fit normalizers or models.

Validation selects the original AE checkpoint, the image-head checkpoints and
the decision thresholds. Thus these values are adaptive validation evidence,
not independent test scores. This one-seed comparison cannot establish training
stability or statistical significance. The original reconstruction scores from
Stage A are not image-only baselines.

The feature cache records source metadata hashes, exact row order, each image's
SHA-256, pretrained-weight hash, transform identity/source hash, extraction source
hash and library versions. It checks for byte-identical images crossing train
and validation and refuses head training if such overlap exists. No rows are
dropped or reassigned. Image bytes are hashed when extracted; a validated cache
represents those bytes, without re-reading every image during head fitting.

## Resource policy and execution

The active benchmark was left running. Initial inspection found the RTX 4060
at 83% utilization with 7,303 MiB reported used; no reusable image feature cache was
available. A two-thread CPU probe on 128 real images took 4.57 seconds, suggesting
about 32 minutes for full extraction under that observed load.

Both scripts set `CUDA_VISIBLE_DEVICES=-1` before their Torch imports. Execution
uses Linux `SCHED_IDLE` (`chrt --idle 0`), idle I/O (`ionice -c 3`), initially two
and subsequently one Torch/BLAS thread, no DataLoader workers and no image writes. This gives the main training
priority on CPU and disk and avoids GPU allocation. Shared memory/cache/thermal
effects cannot be guaranteed absent; no claim of exactly zero throughput impact
is made. The extractor's recorded scheduler value is 5 (`SCHED_IDLE`).

A short resource probe compared 20 driver samples with extraction active to 20
while only the extractor was paused. Mean GPU utilization was 89.55% versus
95.55%. This is not an epoch-throughput measurement or a controlled estimate of
slowdown, but motivated a conservative reduction: at 07:56 Europe/Rome all five
extractor threads were restricted to logical CPU 7. The original training process
was never paused, reprioritized or rebound. Head fitting used CPU 7, one Torch
thread and the same idle CPU/I/O priorities. The original extraction estimate
no longer applies after this resource restriction. The local resource probe is
recorded in `outputs/resource_probe_20261022.json`.

Repeating the short check after the affinity restriction gave 91.25% utilization
with extraction active and 90.25% paused: no reduction was detected in that brief
window, without establishing zero throughput impact. Two Torch threads sharing
one logical CPU were inefficient, so extraction was gracefully interrupted at
20,480 committed train rows and resumed with one Torch thread on CPU 7. No
committed chunk was recomputed. Before resuming, 64 real images were recomputed
with one thread; their cached features matched exactly (maximum absolute error
0.0), recorded in `outputs/thread_parity_20261022.json`.

Cache schema v2 separates the versioned feature algorithm/data identity from
per-session resource settings and source hashes. The exact first implementation
is recognized by SHA-256 for this migration; its complete original identity and
session history are retained in the manifest. A copy of its source is preserved
at `outputs/feature_extractor_initial_20261022.py`. Subsequent feature-algorithm
changes must bump `FEATURE_ALGORITHM`; changing threads/affinity can reuse the
same chunks. This revision changes resource/provenance handling, not feature math.

Use the existing `wsl_image_pytorch` environment from the repository root:

```bash
taskset -c 7 chrt --idle 0 ionice -c 3 env CUDA_VISIBLE_DEVICES=-1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m src_scratches.label_autoencoder.image_features \
  --threads 1 \
  --output src_scratches/label_autoencoder/outputs/resnet18_features_20261022

taskset -c 7 chrt --idle 0 ionice -c 3 env CUDA_VISIBLE_DEVICES=-1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m src_scratches.label_autoencoder.image_heads \
  --threads 1 \
  --features src_scratches/label_autoencoder/outputs/resnet18_features_20261022 \
  --output src_scratches/label_autoencoder/outputs/image_heads_20261022_seed42
```

CPU 7 is specific to the current eight-logical-CPU WSL environment; choose an
available CPU on another machine. An interrupted extraction can resume with the
same command plus `--resume`.
Committed chunks are checked before reuse, including hashes, shapes and row order.
Feature-algorithm, data, weights or preprocessing changes require a new cache
directory; thread settings and source hashes are recorded per session. Head training requires
a complete cache and refuses an existing result directory; head checkpoints are
inference artifacts rather than optimizer-resume checkpoints.

## Files and verification

- [image_features.py](image_features.py): CPU-only extraction, chunk persistence,
  resume checks and cache loading.
- [image_heads.py](image_heads.py): train-only normalizers, three image heads,
  training/evaluation, checkpoint restoration and image-only `predict_image()`.
- [test_image_features.py](test_image_features.py): five synthetic tests for cache
  completeness, resume with a changed thread count, corruption/identity mismatch,
  duplicate-image isolation and explicit legacy migration.
- [test_image_heads.py](test_image_heads.py): four synthetic tests for normalizers,
  frozen-decoder gradient flow, matched AE initialization, all three trained arms,
  checkpoint restoration and prediction from an image without labels.
- [threshold_diagnostic.py](threshold_diagnostic.py): separate post-hoc sweep of
  every distinct validation score to check the original threshold-grid boundary.

The original nine tests and nine image-pilot tests passed, including the added
resource-change/migration checks. The integration tests reject Python CUDA initialization.
They use a tiny synthetic backbone rather than loading the real dataset or its
test split. Real ResNet inference was separately exercised by the CPU probe and
the completed-run image-only verification below.

```bash
python -m unittest src_scratches.label_autoencoder.test_compare \
  src_scratches.label_autoencoder.test_image_features \
  src_scratches.label_autoencoder.test_image_heads -v
```

Generated artifacts stay in ignored `outputs/`: feature chunks and their image
hashes, feature manifest, head manifest, per-head checkpoints, loss histories,
per-label CSVs, `summary.json` and `report.md`. Code/data/weight hashes identify the
actual run even while its implementation is uncommitted. No active training
configuration, benchmark plan or persistent project default is changed.

## Full pilot outcome

Extraction and head fitting completed on all 47,965 train and 5,996 validation
images. All 165 labels have validation positives. There were zero byte-identical
images across these splits; this check does not exclude perceptual near-duplicates.
The two extraction sessions totalled 2,966.12 seconds (49.44 minutes), excluding
the interruption between sessions. Head fitting and evaluation took 52.38 seconds.
The feature cache is reusable for further head comparisons without image extraction.

The prespecified global-threshold grid was 0.05 to 0.95 in steps of 0.01, selecting
the smallest threshold on ties. All values below are validation scores; AP is
threshold-independent. Rare AP averages the 55 least frequent train labels.

| Arm | Macro AP | Micro F1 @0.5 | Grid-tuned micro F1 | Threshold | Rare AP |
| --- | --- | --- | --- | --- | --- |
| Train prevalence | 0.0468 | 0.1437 | 0.3116 | 0.13 | 0.0122 |
| Direct 165 | **0.1281** | 0.2213 | **0.3736** | 0.19 | **0.0699** |
| AE32, BCE + 0.1 MSE | 0.0918 | 0.2284 | 0.3362 | 0.17 | 0.0320 |
| AE32, MSE only | 0.1038 | **0.2711** | 0.3201 | 0.05 | 0.0393 |

The direct, aligned and MSE-only arms ran 19, 59 and 22 epochs, selecting epochs
10, 49 and 12 respectively. Their selected validation ingredient BCE values were
0.14274, 0.16689 and 0.50314. The MSE-only head's higher F1 at the fixed 0.5 threshold
does not translate to better ranking or better F1 after threshold selection.

### Post-hoc threshold diagnostic

The MSE-only arm selected the lower grid boundary, prompting a separate diagnostic
over every distinct validation score for **all** arms. This uses the same saved
checkpoints, with no retraining or test evaluation. It is additional adaptive
validation analysis, not a replacement for the prespecified grid results.

| Arm | Best micro F1 over all score thresholds | Threshold |
| --- | --- | --- |
| Train prevalence | 0.3116 | 0.161972 |
| Direct 165 | **0.3737** | 0.189367 |
| AE32, BCE + 0.1 MSE | 0.3366 | 0.178448 |
| AE32, MSE only | 0.3425 | 0.002957 |

The extended sweep materially improves the MSE-only arm and reverses the two AE
arms' F1 ordering, while the direct head remains ahead. Different thresholds can
produce identical predictions; the prevalence baseline illustrates this. The
very low MSE-only threshold and its high ingredient BCE suggest poor probability
calibration in this setting, without constituting a dedicated calibration study.

### Interpretation and checks

The train-mean-code baseline has standardized validation MSE 0.99442. The MSE-only
image projection reaches 0.91924, with variance-weighted latent R² 0.07559: it
explains only about 7.6% of code variance under this frozen-feature, linear-head
setup. The aligned arm has latent MSE 1.41666 and R² -0.42462. Optimizing ingredient
BCE improves the decoded loss but sacrifices closeness to the teacher code.

Good ingredient reconstruction therefore did not establish a useful image-predictable
code in this pilot. Prefer the direct head for these frozen ResNet-18 features:
it leads macro AP, rare-label AP and threshold-selected micro F1. This does not
show that all label embeddings fail or that the images lack the relevant signal.
One seed, a linear visual projection, a frozen decoder, a fixed loss coefficient
and differing head capacities limit the conclusion. A trainable decoder and a
matching randomly initialized bottleneck would be separate controls before
attributing any later gain specifically to ingredient-AE pretraining. Metadata
and end-to-end fine-tuning remain untested.

Checkpoint/data/source hashes were verified against the completed run. The saved
decoder is bitwise unchanged from the ingredient AE. For the real validation image
`010005____french.jpg`, image-only inference matched predictions from its cached
feature exactly for all three heads (maximum probability difference 0.0), returning
165 scores and, for AE heads, 32 code coordinates. CUDA remained uninitialized.
The main benchmark process was still running with its original scheduling priority
at the final check; no experiment process or resource change was applied to it.

Local run artifacts, retained under ignored `outputs/`:

- [Feature manifest](outputs/resnet18_features_20261022/feature_manifest.json):
  complete rows, image hashes, resource sessions and extraction provenance.
- [Head manifest](outputs/image_heads_20261022_seed42/manifest.json) and
  [summary](outputs/image_heads_20261022_seed42/summary.json): exact settings,
  metrics, epochs, timings and checkpoint hashes.
- [Generated report](outputs/image_heads_20261022_seed42/report.md): original grid
  results; it intentionally remains the report of the prespecified evaluation.
- [Threshold diagnostic](outputs/image_heads_20261022_seed42/threshold_diagnostic.json):
  separately recorded post-hoc results and script/manifest hashes.
- [Completed-run verification](outputs/image_heads_20261022_seed42/verification.json):
  checkpoint integrity, frozen decoder and actual image-only inference checks.
