# Phase 6 benchmark training runtime

**Created:** 2026-10-13
**Last updated:** 2026-10-13

## Scope and authority

This document owns the opt-in executable delta for [Phase 6-D1](../project_objective/model_comparison_methodology.md#phase-6-d1--finite-training-and-comparison-protocol). Scientific decisions belong there; execution belongs in the [training plan](../plans/training_hyperparameter_tuning.md). The accepted Phase 5 training foundation is reused, not requalified. No real benchmark trial has been launched by this implementation checkpoint.

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

`BenchmarkTrainer` is one shared thin Lightning adapter for all five protocols, constructed through canonical `model_training`. It fixes validation every epoch, FP32/single-device/40-epoch behavior and a maximizing AP checkpoint callback. An exact tie retains the first best epoch; `last.ckpt` remains distinct and is not renamed over the best checkpoint. The fit returns the model restored to the retained best-AP weights. An epoch-cost JSONL records elapsed train/validation/checkpoint time and cumulative optimizer updates before any pruning decision.

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

DataModule preparation still reads all-split metadata. The execution boundary forbids test/predict dataloaders; no predictive test evaluation belongs to Phase 6. The source snapshot includes local source content independently of a clean Git tree. No commit/push or source-cleanup operation is part of the launcher.

## Verification boundary

Focused tests in [`test_benchmark_protocol.py`](../../tests/test_benchmark_protocol.py) cover epoch-vs-batch AP, ties/invalid columns, full/selected configurations for all five models, schedule and local-panel arithmetic, one-based pruning, actual CPU Lightning checkpoint selection and full/light restoration, missing-row rejection, isolated Optuna sampler continuation/budget accounting and anchor contract checks. These use synthetic data and temporary studies, not campaign HPO or food-image evaluation. The existing repository metadata-only real test-split check remains metadata-only.

The execution plan records final test counts and the bounded cached-artifact/CPU anchor inspection. This checkpoint does not claim measured new anchor GPU capacity, a real full-epoch benchmark rate, final validation quality or a started campaign.
