# Ingredient-only PCA and autoencoder comparison

**Created:** 2026-10-21

**Last updated:** 2026-10-21

**Status:** Implementation verified; first full CPU pilot completed.

## Purpose and scope

[compare.py](compare.py) implements Stage A of the [research proposal](README.md):
compress ingredient sets to 16, 32 or 64 coordinates and reconstruct the original
165 labels. It reads only the ingredient field from the current v5 train/validation
metadata. Cuisine, flavors, images and test evaluation are outside this pilot.
Every recipe is retained, including an empty target set if encountered. Labels
are binary without epsilon smoothing, with an alphabetical train-only vocabulary
matching the project's strict encoder convention. Unknown validation labels fail.

Validation inputs include their true ingredients because this is reconstruction.
These results do not measure image-to-ingredient prediction or an upper bound on
that future task. More compressible labels need not be more predictable from pixels.

## Compared methods

| Method | Fit and reconstruction |
| --- | --- |
| Global prevalence | Constant scores from train ingredient frequencies; no validation fitting. |
| PCA, k = 16/32/64 | Train-centered float32 PCA, full deterministic SVD, no scaling or whitening. One SVD supplies nested leading components for all three ranks. Reconstruction is inverse transform into 165 real-valued scores. |
| Autoencoder, k = 16/32/64 | `165 -> Linear(128) -> ReLU -> Linear(k) -> Linear(128) -> ReLU -> Linear(165)`. Linear latent coordinates; final outputs are logits. Sigmoid is used for scoring only. No input corruption, dropout, metadata or image branch. |

The AE uses unweighted `BCEWithLogitsLoss`, Adam with learning rate 0.001, batch
size 512, maximum 100 epochs, patience 10 and minimum improvement 0.00001. The
checkpoint with the lowest validation BCE is retained; patience uses the minimum
improvement separately. Each rank starts from seed 42 with the same epoch shuffle
seed. Decoder bias starts at train log-odds; other parameters use PyTorch defaults.
The same settings are used at every rank, without a hyperparameter search.

PCA and AE have different objectives (squared error versus binary cross entropy),
decoder families, parameter counts and fit costs. This is a practical comparison
of these two compression methods, not an isolated test of nonlinearity or an
equal-compute comparison. A linear BCE autoencoder would be a separate control.
The unweighted reconstruction loss is local to this pilot and does not change the
main vision benchmark's training loss policy.

## Metrics and selection

- Macro AP and micro F1 are reported together. The report includes micro AP,
  micro F1 at a fixed 0.5 threshold, and micro F1 at one global threshold selected
  for each arm from the same validation grid 0.05–0.95 in increments of 0.01.
  Equal F1 selects the smallest threshold. This bounded grid is not an exact
  optimization over every possible prediction score.
- AP uses raw PCA scores, preserving their ranking; applying a sigmoid or clipping
  would change interpretation or introduce ties. Raw and [0,1]-clipped MSE and the
  fraction of scores outside [0,1] are recorded separately. PCA scores are not
  calibrated probabilities. AE scores are sigmoid outputs.
- Per-label AP, support, recall and TP/FP/FN use the selected global threshold.
  Three equal-count frequency slices are defined from train support, with
  alphabetical order breaking support ties: 55 labels each for v5. Slice recall
  is a macro average. Labels without validation positives have null AP/recall and
  are excluded from macro means, with their count explicit.
- Validation selects checkpoints and thresholds and supplies these diagnostics.
  Therefore even the validation AP is subject to model selection; tuned F1 has
  further threshold-selection bias. These are exploratory values from one seed,
  not independent test estimates or seed-stability evidence. No confidence
  intervals or significance claim are produced.

## Running in the existing WSL environment

From the repository root, with the existing `wsl_image_pytorch` environment active:

```bash
nice -n 10 env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -u src_scratches/label_autoencoder/compare.py \
  --device cpu --threads 2 \
  --output src_scratches/label_autoencoder/outputs/my_new_run
```

Use a new output directory for each run: the program refuses to overwrite one
that exists. `--help` lists the configurable dimensions, seed, architecture and
optimization settings. Defaults select the existing v5 train/validation files;
there is no test-evaluation option and no dependency installation is needed.

The first pilot uses the command above with output basename
`20261021_ingredients_cpu_seed42` and the environment's absolute Python executable.
`nice` lowers scheduling priority; PyTorch and BLAS are limited to two threads,
with no DataLoader workers. This limits contention but does not promise zero CPU
interference with the ongoing image training.

Resource inspection on 2026-10-21 at 22:21–22:22 Europe/Rome found the RTX 4060
at 91–94% utilization and about 7,594–7,596 MiB used out of 8,188 MiB. A driver
query reported only 362 MiB free. The ongoing benchmark process was present even
though the WSL NVIDIA process table was empty, so that table is not treated as
proof that the GPU is idle. CPU was selected; this process has CUDA hidden.

Every run records three additional driver samples before fitting. An explicit
`--device cuda` is supported only when all three samples show at least 2,048 MiB
free and at most 30% GPU utilization, and PyTorch can use CUDA. There is no
automatic GPU fallback or override of this guard. This is a conservative
point-in-time check, not resource reservation or a guarantee against a later
workload change. The CUDA training path has not been exercised during this task.

## Saved artifacts and reproduction

Outputs are local, ignored by Git under `outputs/`:

- `manifest.json`: arguments, script SHA-256, Git HEAD, library versions,
  train/validation file hashes, full label order, threshold grid and GPU samples.
  Git HEAD alone is not the code identity when local changes exist; the script
  hash identifies the actual implementation that ran.
- `summary.json` and `report.md`: per-arm metrics, run status and fit cost.
  Partial results persist after completed arms; failure/interruption is explicit.
- `pca_k*.npz`: train mean, components, explained variance and label order,
  readable with `numpy.load(..., allow_pickle=False)`.
- `ae_k*.pt`: CPU state dict of the best epoch, dimensions, label order, seed,
  loss name and train hash; `load_autoencoder()` restores it with `weights_only=True`.
  These are inference/pretraining artifacts, not optimizer-resume checkpoints.
- `ae_k*_history.csv`: every completed epoch's train/validation BCE and elapsed
  time, saved when that arm finishes. An interrupted arm must be rerun.
- `*_per_label.csv`: label diagnostics for all seven arms.

Both PCA and AE validation predictions are obtained from saved/reloaded artifacts.
No dense validation predictions or recipe metadata are copied into the outputs.

## Verification

```bash
python -m unittest src_scratches.label_autoencoder.test_compare -v
```

The nine synthetic CPU tests cover train-only vocabulary fitting, unknown labels,
duplicates and empty sets, malformed inputs, test isolation, centered PCA and
component persistence, metrics against scikit-learn, missing positive labels,
the busy-GPU guard, all seven experiment arms, model reload and output protection.
The integration test rejects Python CUDA initialization while executing CPU training.

## First pilot results

Run `20261021_ingredients_cpu_seed42` completed all seven arms on 47,965 train
and 5,996 validation rows, with 165 labels positive somewhere in validation.
Total measured fitting/evaluation time was 125.28 seconds. Shared PCA fitting
took 0.54 seconds; AE fitting took 41.66 / 34.59 / 38.09 seconds for k=16/32/64.
The AE runs completed 100 / 81 / 80 epochs and retained epochs 100 / 81 / 70.
The 16-dimensional arm reached its cap while still improving; it has not
demonstrated convergence. The other two met the configured patience criterion,
which is a stopping rule rather than proof of optimality.

| Method | k | Macro AP | Micro F1 @0.5 | Micro F1 val-tuned | Rare-label macro AP |
| --- | --- | --- | --- | --- | --- |
| Prevalence | — | 0.0468 | 0.1437 | 0.3116 | 0.0122 |
| PCA | 16 | 0.2860 | 0.6450 | 0.6763 | 0.0884 |
| PCA | 32 | 0.3962 | 0.7809 | 0.7900 | 0.1212 |
| PCA | 64 | 0.5750 | 0.8901 | 0.8949 | 0.1889 |
| AE | 16 | 0.7711 | 0.8612 | 0.8620 | 0.6389 |
| AE | 32 | 0.9661 | 0.9630 | 0.9632 | 0.9365 |
| AE | 64 | 0.9945 | 0.9893 | 0.9896 | 0.9908 |

At the same rank, this AE reconstructs the validation ingredient sets better on
both primary metrics, including the train-defined rare-label slice. At k=32 it
already preserves most ingredient information according to these diagnostics;
k=64 offers a more faithful reconstruction at a larger latent size. These are
candidate representations for an image-branch experiment, not a selection of the
best image model. The experiment does not isolate the contribution of the loss,
architecture or optimization, and one seed does not establish training stability.

Full precision, thresholds, other metrics and provenance are in the local
[generated report](outputs/20261021_ingredients_cpu_seed42/report.md),
[summary](outputs/20261021_ingredients_cpu_seed42/summary.json) and
[manifest](outputs/20261021_ingredients_cpu_seed42/manifest.json). These ignored
artifacts remain on this machine; the table and protocol above are versionable.
The executed script SHA-256 is
`95167b1838ace6276aba8bae8c2e1f3c20853747e9250e2136352b4dc8119e35`.
The train SHA-256 is unchanged from [train_audit.json](train_audit.json); validation
SHA-256 is `67ea919607e34ef249553b4fcf9c2851e4dc341a064a23bd405175c83f8ce3f6`.

Verification after completion confirmed the input and code hashes, complete arm
set, all-label coverage, best-epoch selection against the saved curves, local
links and whitespace. The nine synthetic tests passed before the run. Pre-run
GPU samples were 94%, 99%, 99% utilized with 394–395 MiB free. The original image
benchmark process remained running during the CPU pilot; no quantitative claim
about its throughput impact is made.

This remains an exploratory scratch result; it does not modify the selected
vocabulary, active HPO studies, benchmark protocol or durable project status.
