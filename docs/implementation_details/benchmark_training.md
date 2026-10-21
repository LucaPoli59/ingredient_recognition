# Phase 6 benchmark training runtime

**Created:** 2026-10-13
**Last updated:** 2026-10-21

## Scope and authority

This document owns the opt-in executable delta for [Phase 6-D1](../project_objective/model_comparison_methodology.md#phase-6-d1--finite-training-and-comparison-protocol). Scientific decisions belong there; execution belongs in the [training plan](../plans/training_hyperparameter_tuning.md). The 2026-10-13 checkpoint prepared the protocol. The 2026-10-18 first release now executes it under the separate Phase 6.2 authorization, reusing the accepted Phase 5 training foundation.

**Prospective policy, not current runtime:** the user approved [Phase 6-D2 additional plateau pruning](../project_objective/model_comparison_methodology.md#phase-6-d2--conservative-plateau-pruning-for-full-task-hpo) on2026-10-20, recorded2026-10-21. Its guard and separate scientific/source transition are not implemented or activated. The executable v1 policy/hash, Hyperband-only stopping and registered checkpoint-repair source bytes remain unchanged. The [plan's acceptance gate](../plans/training_hyperparameter_tuning.md#additional-plateau-stopping--approved-implementation-pending) owns that future work; approval does not release another trial.

The new path consists of the [primitive policy](../../src/training/resources/phase6_protocol_v1.json), [configuration/metric helpers](../../src/training/benchmark_protocol.py), [shared Lightning adapter](../../src/lightning/benchmark_lgn.py), [isolated study controller](../../src/training/benchmark_campaign.py), [matched-input anchors](../../src/models/benchmark_anchors.py) and [thin CLI](../../scripts/launch_exps/benchmark/train.py). It calls canonical `model_training` and reuses ExperimentalLGNM's exact batching, projection, complete-state restoration and identity checks. It does not edit legacy trainers, the old Optuna studies, selector sources/evidence, metadata or the published59-label artifact.

## Inspect first; execution is explicit

From the main repository and its ML environment:

```bash
python scripts/launch_exps/benchmark/train.py
python scripts/launch_exps/benchmark/train.py --model maxvit_t --export-config NEW_CONFIG.json
```

These commands print the policy or export a new configuration without constructing a model, evaluating images or opening an Optuna study. The default task is full 165; `make_config(..., selected=True)` explicitly builds the published59-label configuration for later transfers. Both orders are obtained from the frozen projection resource, not inferred anew from validation. All five model configurations serialize and round-trip through ExpConfig.

The following is the **future6.2 invocation form**, not an instruction that was executed during6.1:

```bash
python scripts/launch_exps/benchmark/train.py --execute --model maxvit_t --run-name CAMPAIGN_NAME
```

It creates a new model-specific directory below `experiments/benchmark_v1/`, refuses an existing directory and releases only trial 0 by default. After the user's measured-cost review, `--resume --cost-reviewed --additional-trials N` explicitly releases more attempts within the remaining100. All controllers share a nonblocking WSL file lock; it does not reserve the GPU against unrelated user processes, which must be checked before an actual launch. There is no generic CUDA smoke, automatic topology fallback or automatic trial replacement.

## Configurations, exact accumulation and validation ranking

`BenchmarkLGNM` inherits ExperimentalLGNM. It accepts only the declared optimizer/loss/schedule, seed/augmentation/effective-batch policy and matching policy-file SHA-256. That identity and expected full-validation row count are persisted in the complete training configuration, including light checkpoints. The three qualified models keep their existing contract payloads unchanged; the same contract class adds two opt-in `benchmark_anchor` identities without changing serialized legacy adapters.

Training uses the existing exact sample-mean accumulation, including the short final group. Full47,965-record epochs at effective 128 have375 updates. Validation gathers detached CPU logits/targets from every batch, computes per-label non-interpolated AP over the **complete record matrix**, then logs its unweighted valid-label mean once as `val_macro_ap`. A nonfinite score, nonbinary target, missing validation rows or changed valid-label mask is an error. Raw logits avoid sigmoid-saturation ranking ties. Micro F1 at 0.5 is an epoch-global diagnostic and never the HPO monitor. The temperature/threshold fitting frozen for6.5 is not run by this module.

`BenchmarkTrainer` is one shared thin Lightning adapter for all five protocols, constructed through canonical `model_training`. It fixes validation every epoch, FP32/single-device/40-epoch behavior and a maximizing AP checkpoint callback. An exact tie retains the first best epoch. Since the2026-10-20 repair, best and latest state use **two independent native `ModelCheckpoint` callbacks**: the AP-monitored callback has `save_last=False`; the unmonitored callback writes the fixed `last.ckpt` filename every validation epoch with full optimizer/scheduler state and version counting disabled. It runs independently of AP improvement and before epoch-end pruning. Only those two current files are retained; neither is renamed over the other.

Consumers obtain the selected path from `trainer.checkpoint_callback.best_model_path` and the latest path from **`trainer.last_checkpoint_path`**, not the monitored callback's `last_model_path`. After fit, the adapter verifies the latest payload's epoch/update horizon and optimizer/scheduler presence **before** restoring the best-AP weights. An epoch-cost JSONL records elapsed train/validation/checkpoint time and cumulative optimizer updates before pruning. This repairs future persistence without changing model updates, AP selection or stopping policy. The first release's earlier missing-final states remain an explicit retained exception below, not retroactively repaired files.

The Optuna callback reports the **current** full-validation AP at completed epoch1–40 after validation checkpoint callbacks. Trial0 is never pruned; other trials can be pruned before 40. Study selection considers only COMPLETE trials that actually reached 40 and 15,000 updates; its objective is the score of the retained best-AP checkpoint. Exact trial ties prefer the lower number. This route does not use the legacy loss monitor or last-on-tie selection behavior.

## Anchor initialization and source pinning

`BenchmarkAnchor` retains original ResNet18 or DINOv2 register-B/14 topology, with the protocol's common fit/pad224/flip transform and newly seeded head. ResNet replaces its classifier with biased512→L and adapts every parameter. DINO retains the four-last-CLS-plus-final-mean-patch3840 representation, replaces only the final biased linear head and keeps its encoder persistently frozen/eval while preserving requested input gradients. Historical `Resnet18`/`DinoV2B14` classes and their transforms remain unchanged.

Fresh anchor construction loads complete original weights from verified existing cache files. Offline reconstruction uses no original weight download. Missing/mismatched source or artifacts fail explicitly; no mutable remote Hub fallback is used.

The cached DINO source automatically detects installed xFormers, whose kernels do not support CPU inputs. The new anchor binds an instance-local attention dispatcher: CUDA retains the pinned official `MemEffAttention.forward`; CPU uses its official `Attention.forward`/PyTorch SDPA counterpart for dense batches. No weights, state keys, topology, global dependency flag or historical instance is changed. This execution choice is serialized in the anchor contract and the study records the xFormers package version. An initial CPU check exposed the unsupported xFormers path; the explicit adapter resolves it rather than treating that failed check as qualification.

| Artifact | Full SHA-256 |
| --- | --- |
| `resnet18-f37072fd.pth` | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| `dinov2_vitb14_reg4_pretrain.pth` | `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71` |
| DINO 139 Python files, sorted C-locale relative-path/hash inventory | `ebde9a04d8b7c1d09ad007f8bb6a189a639a223d769fb4365d97fb96b61b4e85` |
| DINO `hubconf.py` / `LICENSE` | `e8926c8d01f93e27a6a687a39a1d78716690037fa79153c6483141526ef2a7c2` / `600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7` |

This pins the available source **snapshot**, not an invented Git commit for its `_main` cache folder. A DINO study copies source/hub/license into its own ZIP and rejects a conflicting already-imported DINO package. Standard package notices and the earlier official-artifact evidence remain in the [model contract](experimental_model_contract.md). The controller verifies the full original artifact hash for every model, including the three existing adapters, before a launch.

Physical32 for the two anchors is a declared initial request, not a measured FP32 maximum. Phase 5's measured32/16/32 capacities remain conditional for the three portfolio protocols. Full/selected pairs preserve their declared physical policy and BN differences remain visible. A runtime failure stops the release for a documented decision, not an unrecorded batch/precision/model substitution.

## Study provenance, isolation and recovery

New studies use their own JournalStorage, stable protocol/model study name, maximize direction, seeded TPE and explicit Hyperband resources. Historical journals and `_restore_study` are never used. The fixed reference trial is enqueued with recorded parameters inside the100-attempt allowance. Failed/pruned attempts consume that allowance; exceptions stop execution and are not swallowed.

Artifacts include the policy/hash, model/study identity, package/original-weight identity, metadata hashes/base order, exact source-file inventory and ZIP tied to the Git base revision, per-trial complete configuration, actual epoch costs and separate best/last checkpoints. Selected output identity is already supported by the common config/checkpoint path; reference-control generation/training and operating-point artifacts are later6.3/6.5 work, not silently implemented by the full-study CLI.

Resuming is supported **between finished trials**. A checksummed local sampler snapshot preserves the actual TPE RNG state instead of resetting it from its initial seed. Only load this controller's trusted local snapshot; pickle is not a safe external import format. The controller verifies matching policy, model, Optuna/package/artifact identities and source hashes and uses `load_study`, without recreating trials. Existing trial numbers, intermediate values and costs remain intact. Partial/mid-trial interruption, RUNNING/WAITING states, missing sampler state or journal/snapshot disagreement cause an explicit recovery error. Existing full last checkpoints remain available, but automatic mid-trial recovery is deliberately not claimed; any recovery must account for prior work without resetting the100-attempt budget or silently replaying partial epochs.

### Explicit checkpoint-repair source transition

[`benchmark_provenance.py`](../../src/training/benchmark_provenance.py) supports the specifically reviewed **`final-checkpoint-v1`** transition, not a generic hash bypass. The [registration CLI](../../scripts/launch_exps/benchmark/register_runtime_patch.py) inspects by default; `--apply --accept-missing-final` explicitly registers the user-approved retention exception without starting a trial. It checks all five studies before applying and shares the nonblocking controller lock.

Registration requires exactly the single COMPLETE first-release trial with40 epochs/15,000 updates, unchanged scientific policy, verified original source ZIP/inventory and sampler checksum. The only allowed existing-source changes are `src/lightning/benchmark_lgn.py` and `src/training/benchmark_campaign.py`; the only added inventoried sources are the provenance helper and registration CLI. Any other modified, missing or added source is rejected. This is code-file review scope, not permission to change optimization behavior inside those files.

Each study gains only `runtime_patches/final-checkpoint-v1/`, containing a new exact source inventory/ZIP, `amendment.json` and a byte-preserved `sampler_before.bin`. The amendment records original inventory/ZIP hashes, all retained original artifact hashes, the original journal prefix size/hash, parent sampler hash, source differences, timestamp and explicit disposition: keep trial0 COMPLETE/eligible with its missing final state documented, without replay or retraining. Original manifests, journals, live sampler, configurations, logs, checkpoints and exports are not rewritten.

Before a later study is opened, validation checks the original snapshot, registered active snapshot, exact current inventoried bytes, unchanged retained artifacts, original journal prefix and saved pre-patch sampler. Journal append and normal live-sampler advancement are allowed; rewriting the original journal prefix is not. Existing package/artifact/study/sampler-state guards still apply. New attempts bind that source identity in `trial_N/source_identity.json` and an Optuna user attribute, and record their actual final checkpoint path. Trial0 is not retrofitted. Source registration does **not** approve costs, allocate attempts, enable early stopping or resume a partial epoch. Any future scientific/runtime change needs its own explicit transition; this repair record cannot be edited to absorb it.

DataModule preparation still reads all-split metadata. The execution boundary forbids test/predict dataloaders; no predictive test evaluation belongs to Phase 6. The source snapshot includes local source content independently of a clean Git tree. No commit/push or source-cleanup operation is part of the launcher.

## Verification boundary

### Phase 6.2 execution additions — 2026-10-18

[`first_trials.py`](../../scripts/launch_exps/benchmark/first_trials.py) and [`benchmark_queue.py`](../../src/training/benchmark_queue.py) provide a serial release of exactly one unpruned 40-epoch trial for each protocol in policy order. The CLI prints its plan unless `--execute` is present, requires a new campaign name and persists `campaign_queue.json` with PID, policy/Git identity, model statuses, timestamps and results. A failure retains all evidence and stops the queue. Successful completion produces `awaiting_cost_review`; it never releases the remaining99 trials itself. Interrupted or failed runs require the existing explicit recovery procedure rather than a silent new attempt.

The queue first executes [`benchmark_baselines.py`](../../src/training/benchmark_baselines.py): train-fitted all-negative, prevalence and Laplace-smoothed cuisine priors, unknown-cuisine global fallback, strict train/validation hashes and exact class/base-index identity. It creates ordered validation arrays and the reviewed [baseline result](../experiment_results/phase6_validation_baselines.md). It reads no test metadata or images. The neural DataModule still prepares all-split metadata, with test/predict dataloaders forbidden during execution.

Completed releases call [`benchmark_predictions.py`](../../src/training/benchmark_predictions.py) after Optuna has recorded their COMPLETE states. The exporter reconstructs the retained winner offline through the canonical helper, sets saved model preprocessing and physical batch on the validation loader, and explicitly moves the model to CUDA after Lightning teardown. It verifies sequential sampling, exact metadata image/target/class order, full row count and finite logits. Reproduced macro AP must agree with the retained objective within absolute tolerance1e-6. It stores raw logits, targets, ordered recipe IDs/image names, checkpoint/metadata/policy hashes and metrics under `validation/trial_N/`. Existing compatible exports are verified and reused; incompatible content is rejected. Export failure preserves the completed training trial and its checkpoints for artifact-only recovery.

The controller now invokes canonical `set_torch_constants()` to establish the spawn worker policy, then explicitly restores all frozen FP32/matmul/TF32/cuDNN settings before constructing the trial. This avoids inheriting the legacy numeric defaults. No scientific search or architecture setting changes.

All 15 focused benchmark tests and 291 repository tests passed at launch. Verification included an actual tiny synthetic CPU fit, configuration/persistence/selection tests, train-only prior formulas and ordered inference. Real training started in `phase62-20261018-v1` at16:39 UTC on2026-10-18 (18:39 Europe/Rome), from Git base `fae4301` plus the saved exact source inventory. The later audit below records completed training and a retention defect not detected by those tests; the launch counts do not establish correct final-state persistence.

### First-release artifact verification and retention limitation — 2026-10-20

The [reviewed first-release result](../experiment_results/phase6_first_release.md) and [read-only audit](../../src_scratches/phase6_first_release_audit/README.md) verify all five COMPLETE40-epoch/15,000-update runs, original/source/data identities, selected checkpoints and ordered CUDA validation exports. Review used saved arrays and CPU checkpoint inspection, not new inference. The queue stopped at `awaiting_cost_review`; further99/model are unstarted. Cost and provisional metrics belong to that result owner, not a runtime ranking here.

Installed Lightning2.6.1's validation-end monitored `ModelCheckpoint` calls its last-save path only when the top-k save ran at the current step (or for its link variant). Its train-end handler does not force a save if an earlier checkpoint exists. Therefore `save_top_k=1, save_last=True` in the current adapter **does not guarantee an end-of-training state**. All five `last.ckpt` payloads are byte-identical to the best checkpoint, at one-based epochs10/29/13/37/11, not40. Those full earlier checkpoints contain optimizer/scheduler state, but are not final40-epoch resume artifacts. No final state exists elsewhere in this release's checkpoint inventory. The journal/epoch ledger establishes completed training independently.

The previous claim that a separate last filename established a distinct final full checkpoint is superseded by this direct verification. Preserve these files and manifests as original evidence. Do not reconstruct a fictitious final state, rename best as final, recreate a trial or rerun training automatically. At that audit checkpoint no repair had yet been implemented. The subsequent checkpoint below implements future persistence and records the user-approved exception; the original missing states remain missing.

### Checkpoint repair and provenance verification — 2026-10-20

The user authorized the persistence fix and explicit provenance transition while leaving additional study release/stopping policy open. The production synthetic CPU regression now deliberately keeps the best checkpoint at epoch1 while training continues through40; `last.ckpt` contains epoch40/update40 of that tiny fixture, different trained tensors, complete optimizer/scheduler state, correct identity and successful offline restoration. Returned model weights still match the best checkpoint. A separate actual pruning-callback test verifies the latest full state exists before `TrialPruned` exits. These are bounded synthetic tests, not new campaign runs or GPU qualification.

All24 focused benchmark tests and300 repository tests pass. Temporary-study tests cover explicit exception acceptance, original artifact/journal/sampler preservation, source ZIP integrity, active-source drift, unexpected source changes/additions, partial-state/scientific-policy rejection, saved sampler corruption, idempotent registration and retained one-trial/99-remaining accounting. The real first-release registration and read-only resume checks preserve all original artifacts and do not call `ask`, `suggest`, `optimize` or fit a model. Existing real test-split vocabulary checking remains metadata-only. No early-stopping policy was activated, no additional campaign attempt was released and no commit is implied by this checkpoint.

### Retained 6.1 verification — 2026-10-13

Focused tests in [`test_benchmark_protocol.py`](../../tests/test_benchmark_protocol.py) cover epoch-vs-batch AP, ties/invalid columns, full/selected configurations for all five models, schedule and local-panel arithmetic, one-based pruning, actual CPU Lightning checkpoint selection and full/light restoration, missing-row rejection, isolated Optuna sampler continuation/budget accounting and anchor contract checks. These use synthetic data and temporary studies, not campaign HPO or food-image evaluation. The existing repository metadata-only real test-split check remains metadata-only.

The execution plan records that checkpoint's test counts and bounded cached-artifact/CPU anchor inspection. At the 6.1 checkpoint, new anchor GPU capacity, a real full-epoch benchmark rate and final validation quality had not been established, and no campaign had started. The later 6.2 launch is recorded above.
