# Phase 6 benchmark training runtime

**Created:** 2026-10-13
**Last updated:** 2026-10-21

## Scope and authority

This document owns the opt-in executable contract for [Phase6-D1/D2/D3](../project_objective/model_comparison_methodology.md#phase-6-d3--fresh-fifty-trial-studies). Scientific decisions belong there; execution belongs in the [training plan](../plans/training_hyperparameter_tuning.md). The original v1 first release and subsequent persistence repair remain dated evidence below. R1–R3 implement/accept/prepare v2; the R4 checkpoint records its authorized continuous launch.

**Current implementation:** the active default is `phase6-comparison-v2`, with50 started attempts per model and D2's independent plateau guard. New studies have fresh journals/samplers under `experiments/benchmark_v2/`; v1 studies are not imported or resumed. The [v1 policy](../../src/training/resources/phase6_protocol_v1.json) stays byte-identical. A registered hash resolver preserves saved v1 module/checkpoint semantics for offline reconstruction instead of relabelling them v2. R1–R3 are complete; R4 now has explicit continuous-release authorization under the historical-cost amendment. Neither a successful test nor CLI inspection authorizes training by itself.

The path consists of the [v2 primitive policy](../../src/training/resources/phase6_protocol_v2.json), [configuration/metric helpers](../../src/training/benchmark_protocol.py), [plateau/eligibility helper](../../src/training/benchmark_stopping.py), [shared Lightning adapter](../../src/lightning/benchmark_lgn.py), [isolated study controller](../../src/training/benchmark_campaign.py), [matched-input anchors](../../src/models/benchmark_anchors.py) and [thin CLI](../../scripts/launch_exps/benchmark/train.py). It calls canonical `model_training` and reuses exact batching, projection, complete-state restoration and identity checks. Legacy trainers/studies, selector evidence, metadata and the59-label artifact are unchanged.

## Inspect first; execution is explicit

From the main repository and its ML environment:

The commands below now inspect or build v2 configurations. The original100-attempt route is retained evidence, not a launch option. A new name alone cannot migrate an old study: the controller checks new location, protocol/model/budget identity and matching source/configuration before execution.

```bash
python scripts/launch_exps/benchmark/train.py
python scripts/launch_exps/benchmark/train.py --model maxvit_t --export-config NEW_CONFIG.json
```

These commands print the policy or export a new configuration without constructing a model, evaluating images or opening an Optuna study. The default task is full 165; `make_config(..., selected=True)` explicitly builds the published59-label configuration for later transfers. Both orders are obtained from the frozen projection resource, not inferred anew from validation. All five model configurations serialize and round-trip through ExpConfig.

The following is the **v2 invocation form**, not an executed campaign or permission to skip preparation/launch gates:

```bash
python scripts/launch_exps/benchmark/train.py --execute --model maxvit_t --run-name CAMPAIGN_NAME
```

It creates a new model-specific directory below `experiments/benchmark_v2/`, refuses an existing directory and releases only a newly trained trial0 by default. `--cost-reviewed --additional-trials N` explicitly acknowledges the measured-cost review for multiple/further attempts; `--resume` is only for an already-created v2 study between finished trials. Every release stays within50 total attempts. The existing host-wide nonblocking lock is shared with retained v1 controllers; it is a mutex, not an old-study import or a reservation against unrelated GPU processes. No automatic topology fallback, epoch extension or trial replacement is supported.

### Prepare without training

[`prepare.py`](../../scripts/launch_exps/benchmark/prepare.py) calls [`benchmark_preparation.py`](../../src/training/benchmark_preparation.py). The default prints only a deterministic plan; `--write` creates a new campaign, five empty studies and a `preparation_manifest.json`; `--verify` inspects the existing handoff read-only. Existing directories are never overwritten, and a partial failure retains a failure record instead of silently rebuilding.

Each study uses the controller's shared `prepare_study` path, also reused by ordinary fresh execution. It verifies approved cached weights/packages and frozen metadata hashes, saves the complete165-output configuration template, source inventory/ZIP, policy/environment/data identity and a fresh checksummed sampler. `prepared_study.json` hashes these exact files, including the **empty** journal, plus the pinned DINO source ZIP where applicable. Preparation never calls `ask`, `enqueue_trial`, `optimize`, model construction, training or prediction export. Metadata verification includes the retained test metadata hash, not test images or predictions.

An explicit later `first_trials.py --prepared --execute` verifies the whole campaign before entering its serial first release. Individual study startup similarly uses `train.py --prepared --execute`; it is distinct from `--resume`, which requires finished attempts. Prepared startup requires unchanged empty journal/sampler/configuration/source bytes, opens that same study and only then enqueues the fixed reference0 inside50. It cannot be repeated after an attempt starts. Forgetting `--prepared` still refuses the existing directory; no overwrite or migration bypass is introduced. Source changes after preparation fail verification; a Git commit with identical inventoried bytes alone does not invalidate the snapshot.

The preparation manifest also records the hashed reviewed first-release cost basis and conditional full-run-equivalent arithmetic for the50-attempt opportunity. Only old costs are consumed; old scores, weights and sampler observations do not enter the new cohort. Pruning savings remain unknown and subsequent Q2/Q3/Q4 work is excluded from that arithmetic.

## Continuous release after historical-cost approval

`first_trials.py --prepared --full-budget --cost-reviewed --execute` explicitly releases the entire50-attempt allowance. Both approval/prepared flags are required; default inspection still starts nothing. The shared queue first runs one fresh reference0 per model, then resumes each same study for49 further attempts, serially. It does not repeat references for cost estimation. The individual controller, sampler, Hyperband/plateau policy, failure accounting and strict source checks are unchanged. The legacy first-release mode still pauses when full-budget approval is absent.

Before baseline computation or training, `campaign_queue.json` records the historical-cost basis, frozen projection, approval and absence of a post-reference pause. Every finished model/phase records its result and remaining budget; a runtime exception stops the queue with retained failure state. Completion requires50 started/0 remaining per study and ends at `complete`, not an automatic6.3 handoff. Mid-trial recovery is still not automatic. Training failures consume their attempt but halt this release for explicit review; they are never replaced silently.

The original R3 preparation remains immutable. Because the queue/CLI source changed, `phase62-20261021-v2-release1` is freshly prepared with the updated exact bytes; `phase62-20261021-v2` remains a retained empty preparation, not another training cohort. The unchanged scientific policy is not re-versioned for this operational release amendment. The historical cost projection remains59.08 unpruned full-run-equivalent days; pruning savings and actual finish time are unknown.

## Configurations, exact accumulation and validation ranking

`BenchmarkLGNM` inherits ExperimentalLGNM. It accepts only the declared optimizer/loss/schedule, seed/augmentation/effective-batch policy and matching policy-file SHA-256. That identity and expected full-validation row count are persisted in the complete training configuration, including light checkpoints. The three qualified models keep their existing contract payloads unchanged; the same contract class adds two opt-in `benchmark_anchor` identities without changing serialized legacy adapters.

Training uses the existing exact sample-mean accumulation, including the short final group. Full47,965-record epochs at effective 128 have375 updates. Validation gathers detached CPU logits/targets from every batch, computes per-label non-interpolated AP over the **complete record matrix**, then logs its unweighted valid-label mean once as `val_macro_ap`. A nonfinite score, nonbinary target, missing validation rows or changed valid-label mask is an error. Raw logits avoid sigmoid-saturation ranking ties. Micro F1 at 0.5 is an epoch-global diagnostic and never the HPO monitor. The temperature/threshold fitting frozen for6.5 is not run by this module.

`BenchmarkTrainer` is one shared thin Lightning adapter for all five protocols, constructed through canonical `model_training`. It fixes validation every epoch, FP32/single-device/40-epoch behavior and a maximizing AP checkpoint callback. An exact tie retains the first best epoch. Since the2026-10-20 repair, best and latest state use **two independent native `ModelCheckpoint` callbacks**: the AP-monitored callback has `save_last=False`; the unmonitored callback writes the fixed `last.ckpt` filename every validation epoch with full optimizer/scheduler state and version counting disabled. It runs independently of AP improvement and before epoch-end pruning. Only those two current files are retained; neither is renamed over the other.

Consumers obtain the selected path from `trainer.checkpoint_callback.best_model_path` and the latest path from **`trainer.last_checkpoint_path`**, not the monitored callback's `last_model_path`. After fit, the adapter verifies the latest payload's epoch/update horizon and optimizer/scheduler presence **before** restoring the best-AP weights. An epoch-cost JSONL records elapsed train/validation/checkpoint time and cumulative optimizer updates before pruning. This repairs future persistence without changing model updates, AP selection or stopping policy. The first release's earlier missing-final states remain an explicit retained exception below, not retroactively repaired files.

The Optuna callback reports the **current** full-validation AP at completed epoch1–40 after validation checkpoint callbacks. Trial0 is never pruned; other trials can be pruned before 40. Study selection considers only COMPLETE trials that actually reached 40 and 15,000 updates; its objective is the score of the retained best-AP checkpoint. Exact trial ties prefer the lower number. This route does not use the legacy loss monitor or last-on-tie selection behavior.

### Independent plateau pruning in v2

[`PlateauState`](../../src/training/benchmark_stopping.py) tracks strict observed best AP separately from tolerance-based patience. A score must exceed the patience best by more than1e-4 to reset its counter; small cumulative improvements can eventually cross that threshold. Tracking starts with the first validation, but this guard cannot fire before20 completed epochs or before10 non-improving checks. Strict AP checkpoint selection still records every strict improvement.

`EpochPruning` first asks the unchanged Hyperband pruner. If Hyperband permits continuation, a plateau may additionally prune only when its strict best AP is not above the best eligible COMPLETE40/15,000-update trial in the **same study**. With no eligible incumbent it continues. A potential new winner is protected from this guard, not from Hyperband. Trial0 and completed epoch40 are exempt from both pruning paths. A callback is only attached for HPO trials, not standalone selected/Q3/Q4 runs; at fit startup v2 checks its study policy/model attributes against the module.

Both paths record a `pruning` Optuna user attribute and `pruning.json` containing reason (`hyperband` or `plateau`), policy identity, epoch/updates, current and best AP, patience state, applicable incumbent and summed consumed epoch seconds. Epoch/checkpoint saving and `epoch_costs.jsonl` precede the exception. Raising `TrialPruned` supplies the actual PRUNED state; the controller retains session wall time in its existing `finally` block. No incomplete trainer return is converted to COMPLETE or FAIL. Every attempted trial also saves `stopping_policy.json` and the matching Optuna attribute before fit, alongside its source/configuration identity.

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

New studies use their own JournalStorage, v2 protocol/model study name, maximize direction, seeded TPE and unchanged Hyperband resources. Their policy hash, model and50-attempt allowance are also saved as study user attributes and verified on resume. Historical journals and `_restore_study` are never used. The newly trained reference trial is enqueued inside50; failed/pruned attempts consume that allowance without replacement. Non-pruning exceptions stop execution and are not swallowed.

Artifacts include the policy/hash, model/study identity, package/original-weight identity, metadata hashes/base order, exact source-file inventory and ZIP tied to the Git base revision, per-trial complete configuration, actual epoch costs and separate best/last checkpoints. Selected output identity is already supported by the common config/checkpoint path; reference-control generation/training and operating-point artifacts are later6.3/6.5 work, not silently implemented by the full-study CLI.

Resuming is supported **between finished trials within the same v2 study**. A checksummed local sampler snapshot preserves actual TPE RNG state. Only load this controller's trusted local snapshot; pickle is not a safe external import format. Matching policy/model, package/artifact identity, source bytes and trial accounting are verified without recreating trials. Partial/mid-trial interruption, RUNNING/WAITING states, missing sampler state or journal/snapshot disagreement cause explicit recovery errors. Automatic mid-trial recovery is not claimed; prior work must remain accounted inside50 without silent epoch replay. Fresh creation refuses existing directories, and v1 identities/paths are rejected rather than adapted.

R2 also makes the failure boundary explicit: an oversized release is rejected before creating a study; resume requires existing identity, journal and sampler files, so a missing journal cannot be recreated implicitly. Saved `policy.json` and `data_identity.json` must equal the current approved policy and verified metadata hashes/base order, in addition to the existing environment, source and study-attribute checks. Incompatible input is rejected before starting an attempt, without rewriting its journal/sampler. These checks do not recover an interrupted attempt or change its allowance.

### Explicit checkpoint-repair source transition

[`benchmark_provenance.py`](../../src/training/benchmark_provenance.py) supports the specifically reviewed **`final-checkpoint-v1`** transition, not a generic hash bypass. The [registration CLI](../../scripts/launch_exps/benchmark/register_runtime_patch.py) inspects by default; `--apply --accept-missing-final` explicitly registers the user-approved retention exception without starting a trial. It checks all five studies before applying and shares the nonblocking controller lock.

This legacy registration helper is now explicitly pinned to the retained v1 policy. Current source drift from R1 is intentionally not absorbed into its old patch; replaying that registration against changed active sources must fail. New v2 studies use their own initial exact source snapshot, without a patch/import path from v1. The historical cutoff verifier remains retained, not relabelled as a current-source v2 validator.

Registration requires exactly the single COMPLETE first-release trial with40 epochs/15,000 updates, unchanged scientific policy, verified original source ZIP/inventory and sampler checksum. The only allowed existing-source changes are `src/lightning/benchmark_lgn.py` and `src/training/benchmark_campaign.py`; the only added inventoried sources are the provenance helper and registration CLI. Any other modified, missing or added source is rejected. This is code-file review scope, not permission to change optimization behavior inside those files.

Each study gains only `runtime_patches/final-checkpoint-v1/`, containing a new exact source inventory/ZIP, `amendment.json` and a byte-preserved `sampler_before.bin`. The amendment records original inventory/ZIP hashes, all retained original artifact hashes, the original journal prefix size/hash, parent sampler hash, source differences, timestamp and explicit disposition: keep trial0 COMPLETE/eligible with its missing final state documented, without replay or retraining. Original manifests, journals, live sampler, configurations, logs, checkpoints and exports are not rewritten.

Under D3 this registration/disposition remains historical v1 evidence, not an eligibility path into the fresh50-attempt studies. No extension of this patch or further v1 scientific/source migration is required for the new campaign; do not delete compatibility code or old artifacts as part of that simplification. Within-study continuation for future new studies still preserves their own sampler, policy and source identity.

Before a later study is opened, validation checks the original snapshot, registered active snapshot, exact current inventoried bytes, unchanged retained artifacts, original journal prefix and saved pre-patch sampler. Journal append and normal live-sampler advancement are allowed; rewriting the original journal prefix is not. Existing package/artifact/study/sampler-state guards still apply. New attempts bind that source identity in `trial_N/source_identity.json` and an Optuna user attribute, and record their actual final checkpoint path. Trial0 is not retrofitted. Source registration does **not** approve costs, allocate attempts, enable early stopping or resume a partial epoch. Any future scientific/runtime change needs its own explicit transition; this repair record cannot be edited to absorb it.

DataModule preparation still reads all-split metadata. The execution boundary forbids test/predict dataloaders; no predictive test evaluation belongs to Phase 6. The source snapshot includes local source content independently of a clean Git tree. No commit/push or source-cleanup operation is part of the launcher.

## Verification boundary

### R1 v2 policy and plateau implementation — 2026-10-21

The [implementation checks](../../tests/test_benchmark_plateau.py) and existing benchmark tests pass:31 focused tests and307 repository tests (83.662 seconds). The policy delta test verifies that model/training/Q3/Q4 fields are unchanged; only protocol/cohort,50-attempt allowance and D2 plateau fields differ from v1. V2 policy SHA-256 is `8896b66288f6a53002c30d1b8d050bb0f8bf7ec17370045d0630fd3d43f23181`; v1 retains `4a5fe311e083594de1d872cd3eea3701e8bfa641bfa7753c9b8f3d40d45d756c`.

Tests use synthetic AP histories, temporary scalar studies and actual tiny CPU Lightning fits. They cover tolerance/patience/floor, absent/ineligible/tied/beaten incumbent, independent Hyperband, trial0/epoch40 exemptions,50-attempt mixed-state accounting/resume, old-policy identity rejection and offline saved-v1 module reconstruction. A fixed synthetic AP trace with NopPruner isolates the actual plateau callback: it raises at epoch20, after a full epoch20 checkpoint and matching cost ledger, records PRUNED in the temporary study and persists the same reason/cost locally and in Optuna. This is not a changed campaign pruner or a measured GPU/convergence result.

Default CLI inspection prints v2/50 plus the plateau rule without constructing a model, creating a study or training. No real v2 campaign directory/journal was created, no old trial imported and no predictive test evaluation performed. The suite's existing real test-split vocabulary check is metadata-only. R2 remains the dedicated acceptance checkpoint (including controller-level release/negative-path checks), followed by R3 preparation; R1 alone does not release training.

### R2 controller and stopping acceptance — 2026-10-21

After R1 was committed as `0d28eea`, [14 dedicated acceptance tests](../../tests/test_benchmark_restart_acceptance.py) exercised the actual controller, journal/sampler persistence, source ZIP validation, Optuna optimization/state handling and CLI/queue boundaries in temporary directories. Together with the retained benchmark tests, **45 focused tests and321 repository tests pass** (full suite87.042 seconds).

| Gate | Verified evidence |
| --- | --- |
| Fresh studies and accounting | All five protocol studies start empty at trial0 with fresh reproducible seeded samplers; the controller enqueues the fixed reference inside50. Mixed COMPLETE/PRUNED/FAIL fixtures permit attempt49 and reject the51st attempt or an oversized release without replacement. Ineligible high-score COMPLETE fixtures do not displace an eligible winner. |
| Failure and continuation | A runtime exception stops the release as FAIL, retains its elapsed time/artifacts and consumes its attempt. An explicitly requested later release starts the next number, not a retry. A returned39-epoch or14,999-update double is FAIL, never an eligible COMPLETE. |
| Resume and isolation | Existing-directory overwrite, old identity/location, changed policy/data/environment/source manifests, corrupt sampler, incompatible study attributes, missing journal and an in-flight attempt are rejected before training. Rejected resumes preserve journal/sampler bytes. Source ZIP membership/content and per-trial stopping/source identities are checked. |
| Actual pruning and checkpoint order | The controller lets Optuna record actual `TrialPruned` from the production CPU trainer: forced Hyperband decision at completed5, independent plateau at20. Both save latest full optimizer/scheduler state, ordered cost rows, identical local/Optuna reasons and session time before returning to the controller. |
| Strict best versus patience | A synthetic AP change from0.2 to0.20005 at epoch5 saves the strict AP-best state without resetting the1e-4 plateau counter. Plateau stops at20 with best epoch5 and distinct latest epoch20 tensors; its earlier best is not a winner. |
| Unchanged execution boundaries | Existing tests retain floor/patience/tolerance, absent/tied/beaten incumbent, Hyperband independence, trial0/epoch40 exemptions, full-horizon selection and saved-v1 offline compatibility. Standalone runs have no HPO pruning callback; default CLIs only print and the synthetic first-trial queue stops at cost review. |

The initial negative tests exposed two hardening gaps: an oversized request created an empty study before rejection, and resume did not compare the saved policy/data manifests explicitly. R2 fixes these and guards against implicit missing-journal recreation. Policy files, search/training choices, original evidence and the registered v1 patch are unchanged; this is acceptance hardening, not a scientific amendment.

**Limits:** metadata verification, model construction and validation export are doubles at the controller boundary; the synthetic COMPLETE fixture's40/15,000 counters are injected, not measured training. The two pruning fits use17 synthetic rows/two outputs and one CPU update per epoch, not375. Hyperband's decision is forced to isolate its branch; its production class/resources stay unchanged. These checks do not measure GPU cost, convergence or savings and do not repeat model qualification. The full suite's existing real test-split vocabulary check is metadata-only, not predictive evaluation.

A separate read-only retention check confirms74 original v1 campaign input hashes, all five journal/sampler pairs and ten original/registered source ZIP inventories, with the old queue still `awaiting_cost_review`. There is no real `experiments/benchmark_v2/` directory, new campaign or predictive test access. R2 is accepted; R3 preparation and an explicitly authorized launch remain outstanding. R2 changes remain uncommitted.

### R3 preparation and inspected handoff — 2026-10-21

R2 was committed as `efcf607`. The actual preparation CLI and first-release dry run were inspected, then `prepare.py --run-name phase62-20261021-v2 --write` created `experiments/benchmark_v2/phase62-20261021-v2/` at2026-10-21T10:22:22.531649+00:00. Its state is **prepared_not_launched**: all five studies have0 trials and50 remaining, with no WAITING reference, trial directory, baseline output or campaign queue yet. All templates retain165 classes in exact base order, full-default projection, FP32/effective128,40 epochs and physical32/32/32/16/32 with accumulation4/4/4/8/4. Planned complete counters remain375 updates/epoch and15,000 total; no new updates were measured.

| Retained preparation identity | SHA-256 or value |
| --- | --- |
| `preparation_manifest.json` | `28d108baf21eb3aa807ee54724d7a7b88b0a7902b17bc6b9b7ca4f24afb9bb97` |
| All five84-file source inventories | `5a93b2e5fa1f0e36ad7a0e6866f8d5d6891eb9f53a05631ce06eb3d1646b4e1b` |
| All five source ZIPs | `97f8c6a54816175aa70e9216f0214b35e8ae5598377ba39ad9d06bccd3a0eab9` |
| Git base | `efcf60769d1b5508b15bef62c61213fafe78a976` plus exact saved working-tree bytes |

Actual `--verify` was exercised with `ask`, `optimize`, canonical training and downloads forbidden; subsequent read-only loading confirmed every study empty. Hashing every prepared file before/after confirmed no mutation. Approved original artifacts, package identities, DINO source and all-split metadata hashes pass. No model was constructed or image evaluated. The original v1 evidence/registered repair remain separate and unchanged.

The [six preparation tests](../../tests/test_benchmark_preparation.py) cover empty studies/templates, source/config/journal/sampler hash checks, explicit prepared startup/fixed trial0, rejection of overwrite/empty-resume/repeat launch, queue handoff, CLI inspection/name guards and retained preparation failure. Controller training/export and environment/data verification are synthetic doubles in those tests, not measured qualification. **51 focused and327 repository tests pass** (full suite89.172 seconds); the existing real test-split vocabulary check remains metadata-only.

The manifest's cost basis is the reviewed28.36 h first release, separately spent outside the new allowance. Multiplying that five-model release by50 yields59.08 continuous full-run-equivalent days for HPO alone, **not an ETA or bound**. Unknown pruning/speed variation and later transfers/controls/local work preclude a promised duration. R4 still requires explicit authorization; its first release runs only fresh reference0 per model and stops at the existing cost-review boundary. No R3 commit or training launch is implied.

### R4 continuous-release verification — 2026-10-21

R3 is committed as `9448fbc`. The additional release mode passes53 focused/329 repository tests (94.927 seconds); synthetic doubles verify approval-before-write, all five reference releases followed by49/model, exact attempt-account forwarding, complete termination and immediate stop on runtime failure. The existing controller/plateau/persistence regressions pass unchanged. These are not real250-trial completion evidence or a measured pruning speedup. The real test-split compatibility check remains metadata-only.

The actual full-budget CLI dry run printed50/model without work. `phase62-20261021-v2-release1` was then prepared and inspected with five empty studies before launch, preserving the original R3 preparation. Saved identities:

| Artifact | SHA-256 or value |
| --- | --- |
| Preparation manifest | `fc0bc23fbf172c990f39de14bbe3f9b61d4c6dd0c1b3f7abc88ece45ed28c2b7` |
| All five84-file inventories | `69716056ed8bf0a9978d472e496228db9eaa9113ca3a1b5f2b6965460da5ce05` |
| Source ZIP | `b512ba8b4c262c1bf3e01419866ae4bbb2d3ef0a37960656476bdcbda35bb6a8` |
| Git base | `9448fbcde3dcf6110a002fdb9dbd94efa589f74c` plus saved exact working-tree bytes |
| Scientific policy | Unchanged v2 SHA-256 `8896b66288f6a53002c30d1b8d050bb0f8bf7ec17370045d0630fd3d43f23181` |

The queue started at2026-10-21T10:41:22.466123+00:00, PID94359, with the persisted historical-cost approval. ResNet18 reference0 is running on CUDA; the other references are pending at launch. The [execution checkpoint](../plans/training_hyperparameter_tuning.md#r4-continuous-release--2026-10-21) owns operational state and monitoring. No new full epoch,40-epoch result, pruning saving or final comparison is claimed by startup. No predictive test/predict evaluation, metadata/projection/selector change or old-trial import occurred. Source changes remain uncommitted and must not drift during healthy training.

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
