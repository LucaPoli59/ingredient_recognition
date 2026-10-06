# Image data loading and runtime verification

**Created:** 2026-10-06
**Last updated:** 2026-10-06

## Purpose and scope

This document owns the current Yummly image-loading, target, worker and dashboard
consumer contract. It records the engineering verification that closed Data 2.4;
it does not report predictive performance or authorize a benchmark campaign.
Execution history remains in the [Data plan](../plans/data_ingredient_refactor/yummly_data_phase.md).

## Data and target contract

- [`ImagesRecipesBaseDataModule`](../../src/data_processing/images_recipes.py)
  reads a chosen `metadata_filename` from each split directory and resolves
  relative image names against the shared `images_subdir`, default `imgs/standard`.
  Split membership comes from metadata, not image placement.
- New [`ExpConfig`](../../src/commons/exp_config.py) instances use
  `ingredients_target_v5_metadata.json` and `feature_label="ingredients_target"`.
  The selected v5 population is 47,965 train, 5,996 validation and 5,996 test
  records. The full 165-label task remains the default.
- New target configurations use strict
  [`MultiLabelBinarizer`](../../src/data_processing/labels_encoders.py), with no
  `<UNK>` output. Training determines the vocabulary; validation/test reuse it.
  Unknown labels fail explicitly. Saved encoder classes and their order are
  retained in experiment/checkpoint configuration, not inferred per split.
- Alternative target fields remain configurable and default to the robust
  encoder. Serialized legacy encoders preserve their saved classes, output
  dimension and `<UNK>` behavior. Legacy metadata/checkpoints are not rewritten.
- The optional 59-label projection is a separate explicit configuration, governed
  by the [P7 contract](ingredient_selection.md#p7-runtime-projection). It does not
  replace v5 metadata, the full default, or the original record population.
- Canonical `prepare_data()` eagerly reads and encodes train, validation, test
  and predict metadata; predict currently aliases test. Preparing metadata is
  not predictive test evaluation. Restricted selector data access remains a
  separate specialized path.

## Pinned memory and workers

`pin_memory=None` is the portable automatic policy: enable only on native
Windows (`os.name == "nt"`), disable on WSL and other systems. Explicit `True`
and `False` override it; other types are rejected. Configuration preserves the
policy value rather than serializing one host's resolved Boolean.

Train, validation, test and predict loaders all use the resolved policy.
`persistent_workers` is enabled only when `num_workers > 0`. Worker count and
pinned memory are independent; this policy does not change batch size or add
custom prefetch behavior. It addresses the observed WSL pin-memory-thread OOM,
not a claim that unpinned memory is universally faster.

Verification lives in
[`test_images_recipes_dataloader.py`](../../tests/test_images_recipes_dataloader.py):
platform resolution, overrides/type validation, all four loader constructors,
zero/nonzero workers and configuration reload. These five maintained tests were
added on 2026-10-06; older plan references to that filename did not correspond
to a file present in the audited checkout.

## Model preprocessing and dashboard reconstruction

The canonical training path supplies the model's augmentation/plain transforms
to the DataModule. List-based generic transforms are wrapped with dataset
normalization; already composed model transforms retain their own preprocessing,
including pretrained-weight normalization.

The dashboard now follows the same contract through
[`load_visualization_datamodule`](../../src/dashboards/runtime.py). It validates
the saved projection configuration, reconstructs the DataModule with the loaded
model's transforms, prepares the saved encoder, verifies the model output count,
binds the projection and sets up the fit datasets. Previously the dashboard
omitted model transforms and could silently substitute generic preprocessing.

[`test_dashboard_data_contract.py`](../../tests/test_dashboard_data_contract.py)
uses synthetic split/image fixtures to cover shared image resolution, model
transform shape and normalization, exact saved label order, legacy `<UNK>`
compatibility and rejection of a model/output-count mismatch. Fixture-specific
image statistics are supplied explicitly; no arbitrary-dataset statistics
serialization extension is introduced here.

## Rerunnable bounded runtime smoke

From the repository root in the configured CUDA ML environment:

```bash
python scripts/validation/data_runtime_smoke.py --run-name <new-unique-name>
python scripts/validation/data_runtime_smoke.py --run-name <existing-name> --dashboard-only
python scripts/validation/data_runtime_smoke.py --run-name <existing-name> --serve-dashboard --port 8064
```

[`data_runtime_smoke.py`](../../scripts/validation/data_runtime_smoke.py) uses
canonical configuration, training, checkpoint and Dash APIs. A training run
refuses to overwrite an existing output directory. Dashboard-only verification
rejects metadata or checkpoint hashes that differ from the saved smoke record.
The optional server binds localhost and must be stopped after inspection.

The run uses pretrained ResNet18, seed 42, FP32 tensor precision, batch 16 and
two workers. It bounds training to four batches in one epoch and validation to
two batches, plus two sanity-validation batches. Pretrained weights were cached
for the recorded run; a fresh environment may need to obtain them. Canonical
`set_torch_constants()` settings are retained; this is not a cross-run numerical
reproducibility or TF32-policy experiment.

Artifacts are isolated under `experiments/runtime_smoke/<name>/`: the full
checkpoint, serialized configuration, metrics, `smoke_record.json`, dashboard
cache and trainer scratch directory. The existing trainer's scratch cleanup
therefore cannot touch unrelated scratch contents. W&B logging remains offline.
The record describes engineering checks, not a new dataset validation manifest.

### Verified checkpoint — 2026-10-06

Run `data24-20261006` used Git base `cc8e708` plus the dashboard fix and smoke
sources in this change. This base revision alone is not an immutable snapshot
of the uncommitted additions; the run is not part of the hash-bound selector
campaign and must not be substituted for its provenance.

| Check | Observed result |
| --- | --- |
| Real CUDA training | Four optimizer updates; finite loss; model parameter changed; 165 targets, no `<UNK>` and no selected projection |
| Memory | Peak allocated 1,559.42 MiB; batch 16, two workers, automatic pinned memory disabled on WSL |
| Canonical full-checkpoint restore | Saved class order unchanged; logits on the same validation image exactly equal before/after reload |
| Dashboard | HTTP 200 for routes/layout/dependencies and shared validation image; all 5,996 validation records available |
| Predictions and interpretation | Saved 165-label order and model preprocessing preserved; prediction table agrees with direct inference; Grad-CAM and feature-factorization figures generated |
| Browser inspection | Loaded `trial_0/best_model.ckpt`, image, labels, prediction table and interpretation graphs successfully rendered through the actual page controls |
| Data integrity | All three v5 metadata SHA-256 values unchanged; checkpoint unchanged during dashboard checks |
| Regression suite | `python -m unittest discover -s tests -q`: 149 passing tests, including nine new dashboard/loader tests |

The local evidence is `experiments/runtime_smoke/data24-20261006/smoke_record.json`
and `trial_0/best_model.ckpt` (SHA-256
`8366b554c479f53b2c349e3a6109b627aa7090f8d20e3247eab1e4acd8573690`).
Both train and dashboard checks prohibit test/predict loader access. Canonical
preparation still reads test metadata, as does an existing metadata-only
vocabulary compatibility test. No test image inference, test metric, HPO or
scientific model comparison was performed.

### Historical smoke disposition and limits

The old `experiments/dummy/dummy_experiment/trial_2` run is incomplete, retained
unchanged and not resumed. It has no `best_model.ckpt`; saved `last.ckpt` and
`last-v1.ckpt` report respectively epoch 1 / step 750 and epoch 2 / step 1,125
against an intended 50 epochs. Partial later CSV rows do not establish successful
completion. Its checkpoint hashes remain unchanged.

The new bounded run replaces that pending engineering gate only. Four updates
cannot establish convergence, generalization, full-run resource stability or
the feasibility of the Phase 5 model portfolio. Historical compatibility and
retention remain owned by the [2.1c retention gate](../plans/data_ingredient_refactor/yummly_data_phase.md#work-package-21c--historical-experiment-compatibility)
and its read-only verifier. No historical cleanup or selector retraining occurred.
