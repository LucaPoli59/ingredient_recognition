# Phase 6.2 first full-task release and measured-cost review

**Created:** 2026-10-20
**Last updated:** 2026-10-21
**Status:** Reviewed provisional validation/cost evidence; original missing-final exception retained and explicitly accepted, future runtime repair verified separately

## Purpose and scope

Review the five predeclared, unpruned trial0 runs in `experiments/benchmark_v1/phase62-20261018-v1/` before releasing any other attempts. This is the measured-cost boundary in [Phase 6-D1](../project_objective/model_comparison_methodology.md#phase-6-d1--finite-training-and-comparison-protocol), not completed HPO, a final model ranking or permission to start the remaining99 trials per model. [Phase 6.2](../plans/training_hyperparameter_tuning.md#62--full-vocabulary-baselines-and-tuning) remains **In progress**.

The queue started at 2026-10-18T16:39:36.828289+00:00 and stopped normally at 2026-10-19T21:01:04.142807+00:00 (23:01 Europe/Rome), with `status=awaiting_cost_review`. No additional campaign training was launched during review. The monitoring heartbeat was retired at the handoff; another release requires an explicit decision.

## Evidence and verification

The [audit helper](../../src_scratches/phase6_first_release_audit/audit.py), [retained summary](../../src_scratches/phase6_first_release_audit/summary.json) and [usage note](../../src_scratches/phase6_first_release_audit/README.md) reproduce this review from local artifacts. The audit reads raw journal records without opening/replaying an Optuna study, deserializes trusted checkpoints on CPU without constructing models, and recomputes aggregate metrics from saved validation logits. It reads train/validation metadata only, not dataset images or test metadata. The original training DataModule's all-split metadata preparation remains distinct from predictive test access.

Verified for all five protocols:

- exactly one started trial, recorded COMPLETE; no failed, pruned or extra attempt;
- epochs1–40 in order,375 cumulative optimizer updates per epoch and15,000 at completion, corroborated by journal attributes and the complete epoch ledger;
- earliest maximum-AP checkpoint, saved policy/output identity, optimizer/scheduler state and finite model tensors;
- ordered validation exports with5,996 rows ×165 columns, exact targets/recipe IDs/image names/base-class order, finite logits, matching archive/checkpoint/metadata hashes and macro AP reproduced within absolute1e-6;
- unchanged original pretrained artifacts, data/projection identities, all78 source entries in each ZIP and their exact current source bytes; DINO's separate141-entry source/hub/license archive also agrees with the pinned cache;
- trusted sampler-envelope checksum, unchanged journal/input hashes and no predictive test evaluation.

**Important exception:** presence of `last.ckpt` did **not** establish a final resumable checkpoint. All five files are byte-identical to their selected best checkpoint and predate epoch40. This failed gate is detailed below; it is not hidden by the COMPLETE trial state.

All15 focused benchmark tests pass again (synthetic CPU fit and temporary scalar studies only), and401 relative documentation links/anchors, retained history, protected user-file hashes and diff formatting pass. These tests do not clear the failed final-state gate; the earlier-best/later-final persistence regression is still required. The previously reported291 repository tests are launch-checkpoint evidence, not a new full-suite run during this review.

Provenance:

| Identity | Value |
| --- | --- |
| Git base | `fae43019918c990206ef49bdf6db13956b0ae14d` plus the saved exact source inventory |
| Scientific policy SHA-256 | `4a5fe311e083594de1d872cd3eea3701e8bfa641bfa7753c9b8f3d40d45d756c` |
| Source ZIP SHA-256, identical for all five studies | `c0257171668285e4b7cfb9090b7f74454c3b9eafdda26120419815a1003a9c1a` |
| Audit helper SHA-256 | `d62b484383735fb324d739814123215364e4f790c7f44537f378e93517b4cf4f` |
| Retained audit summary SHA-256 | `2a8eaa78edbe26b48437bba35f1538685fa5d943242aaa8e2b6b44b9d955cd0e` |

The summary contains exact input, checkpoint and archive hashes. Scientific settings remain seed42,224 full-frame/flip0.5, trueFP32, effective128, unweighted BCE/AdamW, two warm-up/38 cosine epochs. Physical batches are32 except MaxViT16. DINO is a frozen-encoder protocol; the other four adapt fully. See the [runtime owner](../implementation_details/benchmark_training.md) for the executable contract.

## Measured cost

One full unpruned run per protocol, in policy order:

| Protocol | Median epoch, minutes | Sum of epoch ledger, hours | Model release wall time, hours | Conditional99 further full-run equivalents, days |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 full | 2.53 | 1.72 | 1.74 | 7.20 |
| DINOv2 register-B/14 frozen | 7.05 | 6.99 | 7.02 | 28.96 |
| EfficientNetV2-S full | 7.64 | 5.10 | 5.12 | 21.13 |
| MaxViT-T full | 13.57 | 9.13 | 9.16 | 37.80 |
| DICA-Net-S full | 7.78 | 5.28 | 5.30 | 21.88 |
| **Total** | — | **28.22** | **28.36** | **116.97** |

The epoch ledger measures elapsed train/validation/checkpoint-callback time. Model-release wall time additionally includes construction, startup, finalization and ordered validation export. Neither is measured GPU-active time. The five releases together consumed about28h21m of wall time; they do not establish throughput under every future workload.

DINO's epoch costs are notably nonstationary: the minimum/maximum are421.79/1,407.44 seconds; epochs7–16 are substantially slower than the later approximately422-second epochs. The retained logs do not identify a cause, so this review does not attribute the variation to a particular competing process, thermal event or model change. Multiplying its median epoch by40 would understate this run's actual cost. MaxViT is the largest observed single-run cost, about1.8× EfficientNet/DICA. Every protocol respected the40-epoch cap; none was extended from learning curves.

The last column is **99 × observed model-release wall time /86,400**, not an ETA, confidence bound or guaranteed upper limit. The remaining495 attempts would total about117 continuous days **if every attempt ran the full horizon at its observed trial0 rate**. All500 full-horizon equivalents, including the completed five, would be about118.15 days. Hyperband will change the consumed epochs, but its completion/pruning fraction is unknown and cannot be imported from historical studies. Future loading contention, failures and the required checkpoint repair can change cost. This arithmetic excludes selected transfers, Q3 controls, local panels and later evaluation; those remain separately budgeted. It is not grounds for silently changing trial caps, precision, adaptation, architecture or stopping rules.

## Provisional validation artifact checks

These aggregate values identify/reproduce retained trial0 artifacts. They are not a post-HPO ranking, unbiased estimates, calibrated F1 or evidence of training-seed stability. No per-label outcome was used to revise the protocol.

| Protocol | Selected epoch, one-based | Validation macro AP | Micro F1 at fixed0.5 |
| --- | ---: | ---: | ---: |
| ResNet18 | 10 | 0.170263 | 0.317901 |
| Frozen DINOv2 | 29 | 0.261662 | 0.391844 |
| EfficientNetV2-S | 13 | 0.222128 | 0.373678 |
| MaxViT-T | 37 | 0.245568 | 0.399972 |
| DICA-Net-S | 11 | 0.225900 | 0.380263 |

All165 AP columns are valid for the common validation targets. The [train-fitted controls](phase6_validation_baselines.md) remain a separate reviewed result; operating-point fitting belongs to6.5. These trials remain counted inside their studies even though a later retention defect was discovered.

## Failed final-checkpoint gate

In installed Lightning2.6.1, the validation-end `ModelCheckpoint(save_top_k=1, save_last=True)` path saves `last.ckpt` only when a top-k checkpoint was saved at that step. If AP does not improve again, `last.ckpt` stays at that earlier epoch; its train-end handler does not force an epoch40 save when an earlier checkpoint already exists. The [benchmark adapter](../../src/lightning/benchmark_lgn.py) relied on this callback to retain the final state.

Direct inspection of all ten checkpoint payloads confirms:

| Protocol | Actual `last.ckpt` epoch / updates | Required final state |
| --- | --- | --- |
| ResNet18 | 10 /3,750 | 40 /15,000 |
| Frozen DINOv2 | 29 /10,875 | 40 /15,000 |
| EfficientNetV2-S | 13 /4,875 | 40 /15,000 |
| MaxViT-T | 37 /13,875 | 40 /15,000 |
| DICA-Net-S | 11 /4,125 | 40 /15,000 |

The installed `lightning/pytorch/callbacks/model_checkpoint.py` source SHA-256 is `673b19f3b473c91649da9a2f8a75cfbfd444f0c01fdbc5baeb162e978ecbf1e7`; the relevant conditions are in `on_validation_end` and `on_train_end`. This is not an interrupted training run: the40 epochs/15,000 updates are independently recorded. Best-AP weights and their validation exports are intact. However, **no epoch40 weights/optimizer state were found in this release**, and scalar logs cannot reconstruct them. The two filenames do not represent distinct best/final states.

Do not relabel a best checkpoint as final, change COMPLETE to FAIL, recreate a trial, overwrite historical artifacts or retrain automatically. Before any further release, explicitly resolve acceptance of these five missing-final artifacts and repair future final-state retention with a regression whose best epoch is earlier than its final epoch. Save a genuine final full state before restoring best weights. Any runtime repair must retain the original source inventory, record the amended source provenance and handle the controller's strict source-resume guard explicitly; do not edit its old manifest merely to bypass that guard. This review records the unmet gate without making that disposition or altering the scientific policy.

## Handoff and limitations

The user must review the measured costs and the missing-final disposition before more attempts are released. All remaining99/model are unstarted. No6.3 training, selector rerun, vocabulary change, calibration, predictive test evaluation, source repair or new commit occurred in this review. The original campaign, journals, sampler snapshots, checkpoints, source inventories and user edits are preserved.

**Subsequent user-authorized disposition,2026-10-20:** after this original audit, the user requested the checkpoint fix and explicit source-transition step. Future independent best/latest full saves and the additive `final-checkpoint-v1` transition are now verified in the [runtime owner](../implementation_details/benchmark_training.md#checkpoint-repair-and-provenance-verification--2026-10-20). The five trial0 runs remain COMPLETE/eligible with their missing final states explicitly accepted; no epoch40 state is fabricated and no trial is rerun. The original78 reviewed input hashes, journals and samplers remain byte-identical, as checked by the [separate transition verifier](../../src_scratches/phase6_first_release_audit/verify_runtime_patch.py) and [retained report](../../src_scratches/phase6_first_release_audit/runtime_patch_verification.json). Original audit helper/summary and quantitative findings above are unchanged.300 repository tests pass; this is future persistence acceptance, not new campaign or comparative evidence.

**Subsequent stopping-policy approval,2026-10-20, recorded2026-10-21:** the user adopted [Phase 6-D2](../project_objective/model_comparison_methodology.md#phase-6-d2--conservative-plateau-pruning-for-full-task-hpo) for future full-task attempts. This changes no quantitative finding or original trial above. The [execution plan](../plans/training_hyperparameter_tuning.md#additional-plateau-stopping--approved-implementation-pending) retains the unmet guard implementation, explicit policy/source transition and next cost-release gates; no additional training is released by approval.

Reproduce the read-only review from the repository root in its ML environment:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python src_scratches/phase6_first_release_audit/audit.py
```

For another retained report, pass `--output NEW_PATH`; its parent must exist and an existing output is refused. The [feature-plan checkpoint](../plans/training_hyperparameter_tuning.md#first-release-review--2026-10-20) owns the operational next action; this record owns the measured findings and their bounds.
