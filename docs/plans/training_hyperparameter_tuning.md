# Training and hyperparameter tuning plan

**Created:** 2026-10-13

**Last updated:** 2026-10-20

**Status:** In progress — first-release cost review and final-checkpoint disposition required

## Purpose and scope

This is the execution owner for [Macro-section 6](../general_plan.md#6-training-and-hyperparameter-tuning). Turn the completed data, vocabulary and model work into a bounded training campaign that answers the [Q1–Q4 comparison questions](../project_objective/model_comparison_methodology.md#research-questions), accounts for the existing ResNet/DINOv2 investment, and hands frozen models and operating points to Phase 7.

The plan does not reopen the selected 59-label vocabulary, the three-model portfolio, or the single-seed constraint. Historical budget alternatives are retained below as dated planning evidence; the adopted [Phase 6-D1 protocol](../project_objective/model_comparison_methodology.md#phase-6-d1--finite-training-and-comparison-protocol) and its executable configuration supersede their open choices. The authorized first release finished training and validation export; its measured-cost review and missing-final-checkpoint disposition are now required. Predictive test evaluation remains unavailable.

The user explicitly removed the previously proposed standalone pipeline-qualification stage on 2026-10-13: the canonical training code and Phase 5 qualification are accepted foundations. The five stages below are renumbered; **6.2 now means full-vocabulary tuning**, not another validation of the training infrastructure. Any work required by a newly adopted metric, monitor or experiment configuration is scoped to that change within 6.1, rather than a repeat of Phase 5.

## Progress tracker

**Overall status:** In progress

**Current task:** 6.2 — first five trials/exports verified; cost review and final-checkpoint retention gate open

**Next action:** Review the [measured first-release costs and failed final-checkpoint gate](../experiment_results/phase6_first_release.md) with the user before any further attempt. Resolve the retained trials' missing-final disposition, then repair future final-state persistence with explicit source-provenance/resume handling. Do not retrain, recreate trials or release remaining allowances automatically.

| # | Task | Status | Evidence or result |
| --- | --- | --- | --- |
| 6.1 | Freeze matrix, historical reuse, budgets and training/selection protocol | **Done** | Phase 6-D1, primitive policy and opt-in canonical metric/configuration/HPO route agree. Five fresh 100-attempt/40-epoch studies; fixed-selector Q3 reuse; no overall GPU-hour cap. Verification is recorded below. |
| 6.2 | Run full-vocabulary baselines and tuning; select `H_base(m)` | **In progress** | Baselines and five COMPLETE40-epoch trials/ordered validation exports verified. Queue is `awaiting_cost_review`;28.36 h observed release wall time. Every `last.ckpt` duplicates its earlier best, so final-state retention is unmet. Remaining99/model, artifact disposition and full-study selection stay open. |
| 6.3 | Run selected-vocabulary transfers and reference-model random controls | **Deferred** | Requires the relevant full-task configuration, frozen control design and budget from 6.1. |
| 6.4 | Bounded selected-task local adaptation | **Deferred** | Included by Phase 6-D1: four additional LR/decay configurations per model after transferred runs. Remains separate from Q3. |
| 6.5 | Freeze calibration, thresholds and the Phase 7 handoff | **Deferred** | Requires required runs and the selected-checkpoint ledger; test evaluation remains Phase 7. |

Creating this plan completes the planning handoff, not 6.1. Later task checkpoints record actual configurations, results, costs, verification and the next action. Preserve completed and superseded evidence.

## Inputs and decision ownership

Read these inputs before making the corresponding decision:

| Input | Use in Phase 6 |
| --- | --- |
| [Problem definition](../project_objective/problem_definition.md) and [benchmark decisions](../project_objective/benchmark_decisions.md) | Image-only recipe-level prediction, frozen split/targets, baseline coverage, threshold-free selection and paired macro mAP/micro F1 reporting. Some dated statements describe pre-refactor history; current data and phase status are in the runtime owner and general plan. |
| [Comparative methodology](../project_objective/model_comparison_methodology.md) | One shared task, full-task HPO, transferred ablation, random controls, optional local adaptation and one seed per configuration. This remains the owner of adopted scientific policy. |
| [Experimental portfolio](../project_objective/experimental_model_portfolio.md) | EfficientNetV2-S, MaxViT-T and DICA-Net-S; ResNet/DINOv2 anchors; full adaptation and the common 224-pixel input boundary for the new portfolio. |
| [Historical basic_v5 result](../experiment_results/basic_v5_resnet_dinov2.md) and [artifact audit](../implementation_details/experiment_artifacts.md) | Actual old study cohorts, pruning, selection semantics, provenance gaps, selected-run durations and limits of reuse. |
| [Selected-task launchers](../implementation_details/selected_vocabulary_training.md) | Historical-configuration transfers on 59 labels; separate from a new benchmark `H_base`. Inspect their actual manifests before claiming a run has completed. |
| [Measured model readiness](../implementation_details/experimental_model_contract.md#measured-cuda-acceptance-and-readiness-matrix) and [comparison constraints](../implementation_details/experimental_model_contract.md#phase-6-comparison-constraints) | Conditional CUDA capacity, initialization, transforms, adaptation and physical-batch/BatchNorm implications. |
| [P7 projection contract](../implementation_details/ingredient_selection.md#p7-runtime-projection) | Fixed class order/base indices and all-record retention, including empty projected targets. |
| [Comparison tooling](../implementation_details/experiment_comparison.md) | Reuse existing artifact readers and reports; preserve objective/protocol cohorts rather than merging unlike studies. |

Operational task state belongs here; adopted numerical and comparison-policy decisions belong in `project_objective/model_comparison_methodology.md`. Reviewed predictive results belong in `experiment_results/`, current runtime changes in `implementation_details/`, and raw logs/predictions/checkpoints in experiment directories. This plan adds no parallel trainer, duplicate dataset or new permanent vocabulary.

## Historical budget and reuse baseline

**Retained planning record:** the recommendations, matrix alternatives and budget scenarios in this section and the next were written before the user's continuous-GPU and fixed-reference decisions. They are not current unresolved launch choices. [Phase 6-D1](../project_objective/model_comparison_methodology.md#phase-6-d1--finite-training-and-comparison-protocol) and the completed checkpoint below now own the adopted policy; preserve the earlier rationale and numerical scenarios as history.

### Verified starting point

The [reviewed result](../experiment_results/basic_v5_resnet_dinov2.md#comparable-cohort) and [artifact audit](../implementation_details/experiment_artifacts.md#observed-campaigns) supply the empirical record. A read-only 2026-10-13 check at `19e2653` re-read both experiment-level `hparam_config.json` files and all 200 numbered `trial_config.json` files. It confirmed the following saved configuration facts without constructing models or evaluating images:

| Field | Historical ResNet study | Historical DINOv2 study |
| --- | --- | --- |
| Experiment | `experiments/basic_v5/resnets_htuning/` | `experiments/basic_v5/dinov2_htuning_v1/` |
| Started trials / maximum epochs | 100 / 40 | 100 / 40 |
| Actual model allocation | 57 pretrained ResNet18, 28 ResnetLikeV1, 15 DummyBNModel | 100 frozen-backbone DinoV2B14 |
| COMPLETE / PRUNED, from reviewed journal audit | 22 / 78 | 37 / 63 |
| Comparable unweighted COMPLETE trials | 19 | 37 |
| Weighted / unweighted configurations | 22 / 78 | 23 / 77 |
| TPE startup trials | 7 | 8 |
| Hyperband configuration | `min_resource=2`, `max_resource=auto`, reduction factor 3 | `min_resource=5`, `max_resource=40`, reduction factor 3 |
| Historical selection | Minimum retained-checkpoint validation loss | Same objective, different pruning/search conditions |
| Retained anchor | Trial 77, pretrained ResNet18 full adaptation | Trial 61, DINOv2-B/14 frozen linear probe |
| Anchor duration, historical journal observation | 62.6 minutes | 112.0 minutes |

The shared configuration SHA-256 values are `2f078127cb1781959acc7870dabf1a84f81fe2a5e8dae06374440085f5b3be24` and `e9255ad7425247c6a9ec638a74302ff484958a9d4507416e9af12d21ecd4daa4`, respectively. Configurations identify `ingredients_target_v5_metadata.json`, 165 outputs and a 40-epoch cap. The reported completion counts and durations retain the earlier audit's provenance and restart limitations; this planning check did not replay the journal or measure total GPU use.

The [saved artifact semantics](../implementation_details/experiment_artifacts.md#scalar-metric-semantics) also matter: the studies mixed weighted and unweighted validation-loss objectives, lack stored AP/prediction arrays, and use batch-derived threshold metrics. No seed field was found in the audited trial configurations. Missing recorded identity is not proof that the underlying data/seed was wrong.

**100 started trials × 40 maximum epochs is a ceiling, not 4,000 observed training epochs or 100 completed ResNet18 runs.** The pruners, search dimensions, adaptation, transforms and precision also differ. The selected anchors' 62.6/112.0 minutes are not average trial costs or estimates for the newly qualified FP32 models. Mutable launchers are secondary evidence: for example, the current DINO launcher still describes 24 trials/25 epochs in prose although its executable settings and saved study specify 100/40.

### Reuse decision to close in 6.1

Retaining a similar nominal budget does not automatically make the old studies comparable; changing the budget does not automatically require discarding every old artifact either. Record a disposition for each anchor and follow-up:

| Intended use or change | Minimum defensible action |
| --- | --- |
| Historical/contextual baseline, outside an equal-budget ranking | Retain the existing studies and claims. If comparable new metrics are wanted, reconstruct saved class/preprocessing identity and evaluate the declared retained checkpoint on validation. Report its historical selection rule and search exposure. |
| Only additional metrics on a fixed old checkpoint | Recover predictions by bounded inference if authorized; no training is inherently required. New metrics do not retroactively turn loss-selected weights into an mAP-selected model. |
| A final equal-procedure comparison with the three new models | Match model protocol, objective/checkpoint selection, budget/search opportunity, data/class identity and training policy. Prefer fresh bounded canonical studies for both anchors when these differ; retain old studies separately. |
| Lower common trial budget | Do not use the winner of 100 historical trials as if it had only the new smaller opportunity. A retrospective chronological prefix can be a labelled historical sensitivity analysis, not a new prospective study. |
| Larger otherwise compatible budget | A traceable continuation is possible only with unchanged objective, sampler/pruner state, search space, data and configuration semantics. Otherwise create a separate study. No new write to the historical journal is implied by this plan. |
| Change horizon, augmentation, precision, adaptation, loss or optimization | Treat it as a changed training protocol. A saved checkpoint cannot supply unobserved training; budget the required new runs. DINO full fine-tuning is not implied by retaining the DINO anchor. |
| Already prepared 59-label ResNet/DINO transfers | Keep them tied to their historical full-task source. Reuse for the final paired ablation only if both sides meet the eventual frozen protocol; otherwise retain as exploratory transfers. |

**Working recommendation, not yet a freeze:** retain 40 epochs as the reference horizon when pricing alternatives, use a compact new common HPO opportunity for the three new models and any anchors included in the headline ranking, and retain all old campaigns as context. Avoid paying for another 100-trial mixed-architecture search solely to mimic a historical directory total. If the available time cannot fund five comparable studies, explicitly narrow the headline comparison to the three new protocols and label ResNet/DINO as historical baselines; adopting that scope requires a recorded methodology decision, not silent pooling.

## Experiment matrix and budget accounting

### Matrix to finalize

| Role | Full 165-label task | Shared selected 59-label task | Random-label controls | Claim boundary |
| --- | --- | --- | --- | --- |
| Pretrained ResNet18 anchor | Reuse or new bounded HPO decided in 6.1 | Fresh transfer of the corresponding full-task configuration, or explicitly historical follow-up | None required | Simple convolutional baseline; exact adaptation/protocol declared. |
| Frozen DINOv2-B/14 anchor | Same reuse decision; retain explicit frozen-feature role | Corresponding fresh transfer | None required | Pretrained representation/linear-probe baseline; not an equally adapted backbone comparison. |
| EfficientNetV2-S experimental adapter | One bounded HPO study | One fresh run with `H_base(m)` | None automatically | 224-pixel pooled-head portfolio comparator. |
| MaxViT-T experimental adapter | One bounded HPO study | One fresh run with `H_base(m)` | None required | Same common task/input and declared adaptation policy. |
| DICA-Net-S | One bounded HPO study | One fresh run with `H_base(m)` | None required | Whole custom readout; independent HPO does not isolate an attention-operator effect. |
| `M_ref`, if distinct from the experimental EfficientNet protocol | Explicit reference full-task budget and `H_base(M_ref)` route | One matching transfer | `R` deterministic, size/support-matched vocabularies | Q3 vocabulary effect, separated from model-family ranking. |
| All-negative, train-prevalence and train-cuisine priors | Compute from train and score on validation | Project and score on the identical task | Reuse scores for each control task | No neural HPO; cuisine remains metadata-assisted diagnostic evidence. |

The exact `M_ref` protocol is still 4B-D1: 384 pixels, its own head/dropout and weighted BCE. The experimental EfficientNet adapter uses 224 pixels and another readout contract. Therefore the Q3 row cannot silently reuse the experimental model just because both are EfficientNetV2-S.

In 6.1, price the literal current methodology, including a separate reference full-task study if needed. Also assess whether a less expensive Q3 design could reuse the completed fixed selector configuration and full-task evidence, or adopt the experimental EfficientNet as a separately named evaluation reference. Either alternative changes the current `H_base(M_ref)`/reference rule and needs an explicit prospective amendment before control outcomes; neither changes vocabulary membership or rewrites Phase 3. The fixed-selector route would test reduction under a fixed, untuned procedure. Do not count it as already adopted or as a free extra portfolio result.

All reduced tasks keep the same records and split membership, even for all-zero projected targets. Train selected/random models from the approved original initialization with fresh label-indexed heads; project the full model only for evaluation. Do not initialize the reduced training run from its trained full-task checkpoint.

### Budget scenarios, not an execution commitment

Use two separate budget axes: search opportunity (attempted configurations, permitted epochs/updates, pruning/stopping policy) and actual cost (GPU-active time, wall time, storage). Equal trial caps do not mean equal GPU-hours; equal GPU-hours allow different numbers of trials. Choose the primary fairness axis in 6.1 and report the other. The working preference is equal bounded search opportunity with compact, justified model-specific LR/decay ranges, plus measured cost reporting.

For illustration, with five fresh studies and a common 40-epoch cap:

| Trial cap per model `N` | Total started-trial allowance | Unpruned full-epoch ceiling across five models |
| --- | ---: | ---: |
| 20 | 100 | 4,000 |
| 40 | 200 | 8,000 |
| 100, historical nominal reference | 500 | 20,000 |

These are arithmetic scenarios, not recommended statistical sample sizes, expected completed trials, or calendar estimates. A different adopted horizon changes the entire row and must be applied consistently to any newly compared anchors. Pruning may reduce cost but is not guaranteed to deliver a known number of full runs. Historical outcomes already informed this planning, so the new study must not be described as uninformed by past validation evidence.

Let `M` be the set receiving new full-task studies, `N_m` their trial caps, `C` any necessary completion/confirmation runs, `R` the random draws, and `K` additional local-adaptation configurations per participating model. Count:

- full-task search: `sum(N_m)`, including any separate reference search;
- selected transfers: one per newly compared protocol, plus a separate reference transfer if required;
- random controls: `R` reference runs, not `R` runs for every architecture;
- local adaptation: `K` new runs per participating category, only if funded; the transferred setting is already available and is not charged twice;
- completion/confirmation: `C`, only where a qualifying retained HPO checkpoint cannot supply the final artifact; no automatic second seed or duplicate full-task training;
- validation prediction recovery, calibration, reporting and contingency for interruptions/storage.

For each protocol, estimate `T_m = sum_i(e_mi * t_epoch_m + overhead_mi) + T_inference_m`, using actual observed epochs/session coverage and an explicit timing basis. Audit old journal/CSV/TensorBoard coverage before summing historical durations: resumed trial duration can omit earlier work. Retain unknown time as unknown. The Phase 5 four-update timings establish feasibility, not reliable full-epoch throughput; a mechanical extrapolation must be labelled as such and cannot alone approve a calendar budget. If needed, record timings from a predeclared first campaign trial and pause at a stated cost boundary before the remaining allocation; this is budget accounting, not a separate qualification campaign or an opportunity to alter architecture from performance outcomes.

Before launch, the budget sheet must state usable GPU-hours/days, trial/horizon policy, estimates and uncertainty, control count/matching feasibility, whether local adaptation is included, inference/storage allowance and a predetermined stop/review condition. A cost-driven revision must be logged and handled consistently across comparisons, not silently grant extra trials to a promising model.

## 6.1 — Freeze matrix, budget and protocol

### Completed protocol/configuration checkpoint — 2026-10-13

The user permits continuous GPU use without a fixed total hour/day allowance and explicitly selected fixed-selector reuse for Q3. The adopted [Phase 6-D1](../project_objective/model_comparison_methodology.md#phase-6-d1--finite-training-and-comparison-protocol) closes the matrix, reuse, budget, search/recipe, selection, controls, local panel and operating-point decisions. The older open-question paragraphs below are retained checkpoint history, not outstanding user questions.

- Five fresh studies: pretrained full ResNet18, frozen DINOv2 register-B/14, experimental EfficientNetV2-S, MaxViT-T and DICA-Net-S. Keep old loss-selected studies/transfers separate. Each new study has100 started attempts and40 maximum epochs, TPE/Hyperband, with an unpruned fixed trial 0 counted inside 100.
- Common224 full-frame/flip input, FP32/effective 128, unweighted BCE/AdamW, two-axis LR/decay search, two warm-up epochs and38 cosine epochs. Native normalization/physical-batch and frozen-DINO differences remain explicit.
- Selection is full-validation macro AP, maximizing every epoch with earliest-epoch/lowest-trial tie rules. Only completed40-epoch trials supply `H_base`; final full checkpoints remain distinct from best-AP checkpoints.
- Q3 reuses `phase3-d1-v3`'s fixed384 recipe and final40-epoch full model, plus six fresh reduced runs: selected 59 and five deterministic support-matched random59 vocabularies. All use final40-epoch weights. No reference HPO, changed selector evidence, manual vocabulary revision or mixing with experimental224.
- Include the four-run LR/decay local panel for each of the five categories, separately from transferred Q2 and fixed Q3. Temperature/global-threshold fitting and projected-task handling are fixed prospectively for6.5.
- Release only the first trial per model before a measured-cost review. MaxViT has the clearest large wall-time risk, and no cap is extended automatically if a model is still improving at 40. The conditional smoke arithmetic remains explicitly not an ETA.

The [primitive policy](../../src/training/resources/phase6_protocol_v1.json), [configuration/metric helpers](../../src/training/benchmark_protocol.py), [shared Lightning delta](../../src/lightning/benchmark_lgn.py), [isolated HPO controller](../../src/training/benchmark_campaign.py), [matched-input anchors](../../src/models/benchmark_anchors.py) and [CLI](../../scripts/launch_exps/benchmark/train.py) implement the6.1 preparation. They reuse canonical `model_training`/ExperimentalLGNM, rather than a per-family trainer. The CLI only prints the policy by default; actual execution requires `--execute`. Historical training, metadata, projection membership and selector sources are unchanged. Study resume uses the retained journal and sampler state, not legacy delete/recreate recovery; mid-trial interruptions explicitly require recovery rather than an uncounted retry.

Verification at base `e0653dd` plus this working-tree change: **11 focused tests and287 repository tests pass**. Tests include an actual40-epoch **tiny synthetic CPU** fit, AP checkpoint/tie and full/light restore checks, five full/selected config round trips, and temporary scalar Optuna studies for continuation/accounting; these are not campaign HPO or real-food training. Additional cached-artifact CPU inspection verifies fresh ResNet and DINO weight hashes, DINO persistent-eval/frozen parameters and finite input gradients on a synthetic224 tensor. The initial CPU xFormers failure was corrected with an explicit instance-local official CPU attention path, without modifying cached or historical sources. New anchor physical 32 is still a configured request, not a measured GPU cap. Existing real test-split verification is metadata-only, not predictive evaluation. See the [runtime owner](../implementation_details/benchmark_training.md) for exact provenance and recovery boundaries.

**Outcome:** 6.1 **Done**, 6.2 **Pending**. No real campaign, selector rerun, predictive test evaluation, new commit or push occurred. The scheduled continuation was removed at the user's request; work resumed in this chat.

### Historical cost/disposition checkpoint — 2026-10-13

The [reviewed cost baseline](../experiment_results/phase6_budget_baseline.md) and [reproducible summary](../../src_scratches/phase6_budget_audit/summary.json) complete the retained-evidence cost audit, not the 6.1 freeze. Recorded journal sums are 41.98 h for ResNet and 106.91 h for DINO, with restart-coverage limitations. Phase 5 timings do not establish full-epoch throughput; they also show no basis for treating 59-label transfers as cheaply proportional to output count. No new training or inference was performed.

**Recommended disposition, pending the affordable matrix:** preserve the old 100-trial studies and selected artifacts as historical evidence; use fresh compact studies for ResNet18 and frozen DINO if they join a matched-budget headline with the three new models. Do not resume the old loss-selected studies under the new mAP objective. Historical selected-task launchers stay attached to trials 77/61; no completed selected campaign was found in the main workspace's expected `experiments/selected_v5` location, which does not establish whether the user ran them elsewhere.

The default proposal remains five explicitly named protocols with equal bounded search opportunity, not equal clock time, keeping 40 epochs as the horizon to price. However, **20 trials is not yet a recommended minimum or approved cap**: allocate selected transfers, reference controls and inference first, then size the HPO pool within the remaining envelope. If unaffordable, present either a smaller common search or an explicitly narrowed headline comparison; never silently mix historical 100-trial winners into a small-search ranking. Optional local adaptation can be omitted explicitly before compromising the required paired controls.

The user has been asked for total GPU-days, continuous versus intermittent availability and any deadline. Until that input arrives, the trial/control counts and precise stopping allocation stay open. The separate 384-pixel reference versus 224-pixel experimental EfficientNet decision also remains open: the audit supplies cost evidence, not authority to merge the protocols. A source/configuration implementation of the new monitor follows the adopted recipe; there is no repeat pipeline-qualification gate.

### Ordered decisions and deliverables

1. **Historical reuse and scope.** Complete the disposition table for ResNet, DINO and any selected-task follow-up; identify missing evidence without retraining by default. Distinguish a common-budget headline from an explicitly historical/fixed-baseline table. Confirm the five-protocol matrix or record any scope amendment.
2. **Compute allocation.** Price the scenarios, including the separate-reference issue, choose the fairness axis, trial cap, epoch/update horizon, pruning policy, failure accounting and total envelope. Use one declared training seed per configuration, with sampler/control/bootstrap randomness recorded separately; no replicated-seed campaign.
3. **Common training recipe and compact search.** Freeze the fields below before opening a study. Record which fields are fixed/common, fixed/model-specific or tuned. Use the old results to justify reasonable ranges, not import old winner weights or grant unrecorded extra search.
4. **Selection and evaluation semantics.** Use the problem definition's threshold-free validation macro mAP intent and paired final micro F1; freeze epoch-level aggregation, metric edge cases, checkpoint cadence, tie-breaks, pruning/early-stop monitors and the distinction between observed best and actually retained checkpoint. Do not silently keep loss optimization because it is the old runtime default.
5. **Record the freeze and minimal configuration work.** Save the adopted policy in the methodology owner, link a versioned machine-readable manifest/search specification, and configure thin launchers through the canonical APIs. Address only concrete gaps introduced by this policy, with focused checks of changed behavior. Update this tracker and the general plan when 6.1 is complete.

| Field to freeze | Existing constraint or concrete choice needed |
| --- | --- |
| Model/weights/adaptation | Exact artifact and head for each protocol; full adaptation for the qualified portfolio. Frozen DINO remains explicitly a different baseline protocol. No unbudgeted DINO fine-tuning or custom-size search. |
| Data and preprocessing | Frozen v5 split, 165/59 class identities, full-record retention. Common 224 fit/pad for the new portfolio; explicitly resolve anchor-transform differences if anchors enter the matched comparison. |
| Augmentation | One declared common policy/opportunity; decide fixed versus tuned. Do not inherit historical hard crops or selector flip settings silently. |
| Loss and class weights | BCE reduction, train-only weighting formula and whether weighting is fixed or searched. Use an objective with common semantics across trial loss choices. Preserve retained-label weights exactly under projection where the formula permits; document any label-dependent renormalization. |
| Optimizer and search | Fix optimizer/schedule families where possible to keep the small search meaningful; specify LR/decay ranges and any justified parameter groups. Avoid reintroducing architecture identity as a trial parameter. |
| Horizon and stopping | Evaluate retaining 40 epochs versus a changed common cap; declare scheduler horizon, checkpoint cadence, pruning resources and any early stopping before outcomes. Matched transferred procedures use the same declared budget/stopping rule, with realized epochs/updates reported. |
| Batch, precision and BatchNorm | Choose effective batch and precision explicitly. Measured FP32 portfolio capacities are 32/16/32 at effective 128; they are conditional. Decide common feasible physical batch versus disclosed model-specific batches, and keep the same physical policy within full/selected/random pairs. Accumulation does not equalize BatchNorm; native BN settings also differ. |
| Selection and operating point | Full-validation macro mAP aggregation; declared tie-break/checkpoint rule. Final micro F1 uses a validation-selected global threshold after model selection. Calibration method, fitting boundary and identity/reuse under projection must be explicit. |
| Reproducibility | One seed/configuration; pinned source/config/artifact/class identity, full/light/resume meaning, fresh initialization, study names and exact recorded trial budget. |
| Q3 controls and Q4 option | Reference protocol, number of draws, deterministic support matching and allowable overlap; optional local panel shape and equal per-category opportunity. No manual ingredient-review dependency. |

### Accepted training foundation and narrow remaining delta

Phase 5 already verified model construction, CUDA updates, exact experimental batching, full/selected persistence and real consumers. Do not repeat a general model/trainer/dashboard qualification, add a dummy qualification study, or reopen the architecture.

The source inspection on 2026-10-13 confirms `val_loss` remains wired into [trainer callbacks](../../src/lightning/lgn_trainers.py), [Optuna study naming](../../src/training/htuning_exp.py) and the [experimental scheduler interface](../../src/lightning/experimental_lgn.py). Epoch-level benchmark AP, calibration and threshold semantics are distinct from proving that training works. Any required monitor/metric additions belong to the 6.1 protocol delta or the advanced Phase 7 evaluation work. Reuse existing tested components and check only changed aggregation, selection and identity behavior; no broad prerequisite stage is recreated under another name.

**Completion:** one explicit matrix and reuse disposition; an affordable cost sheet; no unresolved run-defining fields; adopted methodology and executable configurations agree; the selected metric reaches checkpoint/HPO consumers with correct semantics. Then 6.2 can start. Merely creating this document does not satisfy that gate.

## 6.2 — Full-vocabulary baselines and tuning

### First-release review — 2026-10-20

All five predeclared trial0 runs finished normally; the queue stopped at `awaiting_cost_review` on2026-10-19 at21:01 UTC. The [reviewed result](../experiment_results/phase6_first_release.md) and [reproducible audit](../../src_scratches/phase6_first_release_audit/README.md) verify each journal's single COMPLETE trial, all40 epochs/15,000 updates, exact source/weight/data identities, AP-selected checkpoint and ordered5,996×165 validation export. No additional campaign attempt, image inference or predictive test evaluation was performed during review. The completed training remains counted inside each100-attempt study. All15 focused tests pass again on synthetic CPU/temporary studies;401 documentation links/anchors, retained history, protected hashes and formatting pass. The291-test full-suite result remains the earlier launch checkpoint, not a new full-suite run.

Observed model-release wall times are1.74/7.02/5.12/9.16/5.30 h for ResNet/DINO/EfficientNet/MaxViT/DICA, totaling28.36 h. The result separates epoch-ledger/session/release timing and records substantial DINO rate variation. Multiplying the five observed release times by99 gives116.97 continuous days **only as unpruned full-run-equivalent arithmetic**, not an ETA or bound. Hyperband's actual pruning fraction and future cost are unknown; selected/reference/local work is additional. No budget/horizon or scientific setting was changed.

**Unmet artifact gate:** CPU inspection of all ten checkpoint payloads shows that `last.ckpt` is byte-identical to the selected checkpoint, at epochs10/29/13/37/11 rather than40. Installed Lightning2.6.1 updates this monitored callback's last file only when a top-k save occurs. The best artifacts and exports are valid; the final epoch40 weights/optimizer state were not retained and cannot be recovered from scalar histories. The previous launch checkpoint only verified filename existence and did not establish final-state semantics.

Before releasing more trials, obtain the user's cost review and explicit disposition of this missing-final evidence. Future retention needs a focused earlier-best/later-final regression and a genuine final full save before best-weight restoration. Preserve original journals/checkpoints/source inventories; any repair must document the new source identity and strict-resume transition rather than overwrite old manifests. Do not relabel best as final, invalidate/recreate COMPLETE trials or retrain them automatically. No source repair or new commit occurred in this review. The monitoring heartbeat is retired because training is finished and user direction is required;6.2 remains **In progress**.

### First-release checkpoint — 2026-10-18

Committed the complete 6.1 preparation as `fae4301` after all287 existing tests passed, excluding the three user-owned dirty files. The user then authorized6.2. The [first-release launcher](../../scripts/launch_exps/benchmark/first_trials.py) now computes the required priors and executes the five predeclared trial0 configurations in policy order, one at a time. It records persistent queue/model state and stops at failure or `awaiting_cost_review`. It does not release the other99 trials automatically.

Campaign: `experiments/benchmark_v1/phase62-20261018-v1/`; parent log: `experiments/benchmark_v1/phase62-20261018-v1.log`. Launch was 2026-10-18T16:39:36.828289+00:00 (18:39 Europe/Rome), queue PID386, Git base `fae43019918c990206ef49bdf6db13956b0ae14d`. ResNet18 trial0 started at16:39:39 UTC. The first study verifies the original artifact, records its source inventory/ZIP, prepares the complete fitted configuration and enters CUDA training. The remaining DINO/EfficientNet/MaxViT/DICA protocols are queued. Scientific policy hash remains `4a5fe311e083594de1d872cd3eea3701e8bfa641bfa7753c9b8f3d40d45d756c`.

The first ResNet epoch completed375 optimizer updates in193.63 seconds, including validation/checkpoint callbacks. This is one initial epoch, not a stable campaign ETA. All78 saved source entries match the ZIP and current source bytes; ZIP SHA-256 is `c0257171668285e4b7cfb9090b7f74454c3b9eafdda26120419815a1003a9c1a`. A same-chat monitor checks every four hours while the queue runs, preserving the earlier control interval preference. No additional trial allocation is released automatically.

The [reviewed baselines](../experiment_results/phase6_validation_baselines.md) are complete on both full165 and selected59 validation columns. They use only train-fitted priors, retain every row and store ordered scores/targets/IDs. Full-task macro AP is0.046751 for either constant control and0.126409 for the cuisine diagnostic; current F1 uses fixed0.5 and remains provisional for the operating-point phase. Baseline computation reads no test metadata or images; neural DataModule preparation retains its existing all-split metadata access while test/predict image loaders are forbidden.

Execution additions preserve ordered validation logits from the retained checkpoint after a completed release, using explicit CUDA inference after teardown and canonical offline reconstruction. They verify the actual record/class/target identity and AP reproduction, without fitting calibration or selecting a different checkpoint. These additions and the original metric/configuration path pass15 focused and291 total repository tests, including synthetic CPU fits and temporary scalar studies. No selector source/evidence, target metadata, projection membership or user-owned dirty file changed. The 6.2 changes remain uncommitted.

**Still required at this launch checkpoint:** complete the first five runs and their validation exports, review actual costs, execute the remaining approved study budgets, select eligible `H_base(m)` artifacts and publish the complete study/cost/selection ledger. The package remains **In progress**; a successful launch is not its completion. The later review above records actual completion and the newly discovered retention exception.

- Compute all-negative, prevalence and cuisine-prior predictions from train; evaluate on validation with the same metric definitions. Preserve the cuisine diagnostic boundary.
- Run only the frozen matrix on all 165 labels through canonical training/Optuna. Each model protocol gets its own identified study; retain complete, pruned, interrupted and failed trials without treating them as equivalent completed observations.
- Record configured and consumed budget, seed, transforms, physical/effective batch, adaptation, finite loss/update state, and exact selected checkpoint/configuration. Use explicit resumption with the same study protocol; do not mix a new objective or budget into an old study silently.
- Select `H_base(m)` by the frozen rule. Reuse the qualifying retained checkpoint when sufficient; a further completion run must have its role and cost predeclared, with the same single-seed limitation.
- Obtain ordered validation scores needed for paired projection and final calibration. A historical checkpoint evaluated under new metrics remains marked historical unless its full reuse gate passed.

**Output:** full-task study/selection ledger, checkpoint and validation-score identities, consumed-cost summary, baseline scores, and the exact per-model hyperparameters transferred in 6.3. No test evaluation.

## 6.3 — Shared selected task and random controls

For each participating protocol, train a fresh 59-output model under unchanged `H_base(m)`. Compare its predictions with the corresponding 165-output model projected onto exactly the same 59 columns, records and evaluation policy. Preserve the full-model checkpoint identity and the order/hash of both output spaces.

Generate and freeze `R` deterministic random vocabularies of cardinality 59 before their training outcomes. Use the methodology's train/validation support matching, record strata, overlap and residual mismatch, and never restrict draws only to the 106 excluded labels. Do not select random draws from model scores or discard an inconvenient control result. Random-vocabulary seeds index different tasks, not repeated training seeds.

For the declared reference protocol, report the paired gain on each vocabulary,
`delta_q(V) = score_q(reduced-trained model on V) - score_q(full-trained model projected to V)`, for macro mAP and micro F1. Compare these gains across selected and random vocabularies, rather than absolute scores on different label sets. Apply the predeclared calibration/threshold rule consistently and identify whether a reported F1 gain includes operating-point fitting; AP remains threshold-free. Use the transferred configuration for every control, without local adaptation.

**Output:** one auditable full-versus-selected pair per category, the complete reference control ledger, per-record validation predictions and descriptive paired gains. A small `R` does not establish seed stability or formal significance, and these selection-influenced validation results are not final test estimates.

## 6.4 — Optional local adaptation

If funded in 6.1, run a small declared LR/decay neighbourhood around `H_base(m)` on the shared selected task. Specify the complete candidates or deterministic generation rule, common per-category opportunity and selection rule in advance. The already-run transferred setting is the baseline candidate; do not pay for it again.

Record `H_local(m)` separately from `H_base(m)` and never substitute locally optimized results into Q3's selected-versus-random calculation. No second unrestricted HPO or multiple-seed experiment is required. If omitted, preserve this row as **Deferred** with the explicit budget decision and report only transferred selected-task performance; optional omission does not prevent Phase 6 completion.

## 6.5 — Freeze operating points and Phase 7 handoff

- Fit the declared calibration and global threshold procedure on validation after model/configuration selection. Freeze any validation partition/cross-fitting choice and finite-sample limitations beforehand; never tune on test. Store raw logits as well as fitted parameters so ranking and calibration remain distinguishable.
- Publish a compact ledger separating common full-task comparisons, transferred selected-task comparisons, reference random controls, optional optimized selected results, and historical anchors. Include every required result or explain a failed/missing gate; do not silently replace a failed model.
- Hand off exact model/checkpoint and original-weight hashes, metadata/class/projection identities, source/config/environment, seeds, search spaces/budgets, consumed epochs/updates/time, predictions/record IDs, calibration/thresholds and the planned comparison-table schema.
- Keep bootstrap/uncertainty and per-label reporting compatible with the frozen exact-image groups. Preserve the distinction between finite evaluation-sample uncertainty and unavailable training-seed variability.

**Completion:** every mandatory experiment/control in the frozen matrix is accounted for, required baseline and model predictions exist, all operating-point choices are sealed using validation only, and Phase 7 can evaluate the selected artifacts once on the untouched test task. No final ranking or test metric is required to close this execution plan.

## Components and expected artifacts

Reuse [canonical training](../../src/training/commons.py), [HPO orchestration](../../src/training/htuning_exp.py), [experiment configuration](../../src/commons/exp_config.py), [experimental runtime](../../src/training/experimental_runtime.py), [batching](../../src/training/batching.py), [trainer callbacks](../../src/lightning/lgn_trainers.py) and [projection APIs](../../src/ingredient_selection/runtime.py). Thin launchers belong under `scripts/launch_exps/`; scientific metric work reuses existing evaluation/analysis components where suitable. Concrete new filenames follow from the frozen protocol, not a new architecture of training services.

Required experiment artifacts are a run/study manifest with source and configuration provenance, full/selected/random class identity, search distributions and trial ledger, selected checkpoint, train/validation histories and ordered validation logits/targets. A compact budget and selection ledger must be reproducible from those artifacts. Keep generated files outside `docs/`; preserve historical journals, campaign inventories and hash-bound Phase 3 sources. No physical cleanup or commit is implied by this planning task.

## Planning checkpoint and change log

| Date | Event | Result |
| --- | --- | --- |
| 2026-10-13 | Created the Phase 6 plan after Phase 5 completion; incorporated the user's historical-budget question and removal of standalone pipeline qualification | 6.1 **In progress**; five stages, explicit reuse routes, priced-scenario framework and reference-protocol decision. Canonical runtime accepted; narrow new-metric/configuration work identified. No training, inference, model construction or study mutation performed. |
| 2026-10-13 | Completed the read-only historical-cost subtask of 6.1 and checked all 200 saved configurations against retained inventory | Reviewed cost baseline and reproducible metadata-only audit retained; recommended historical/new-study separation. 6.1 remains **In progress**, awaiting the user compute envelope and protocol/configuration freeze; no campaign started. |
| 2026-10-13 | Adopted Phase6-D1 after continuous-GPU availability and explicit fixed-selector reuse; implemented and checked the narrow executable protocol delta | 6.1 **Done**; 6.2 **Pending**. Five bounded fresh studies, separate fixed-reference controls and local panel, global AP/maximizing selection, strict configuration/provenance and non-destructive study continuation. No campaign launched. |
| 2026-10-18 | Committed 6.1 as `fae4301`; completed prior controls and launched the authorized first full-task release | 6.2 **In progress**. Baseline artifacts/result retained; ResNet18 trial0 running, four later protocols queued.291 tests pass; measured-cost review and full-study selections remain open. |
| 2026-10-20 | Reviewed all five completed reference trials and measured costs; found the monitored callback did not retain final epoch40 state | Training, selected checkpoints, source identities and validation exports verified;6.2 remains **In progress** at cost review and explicit artifact-disposition/retention-repair gate. No further training or commit; monitor retired. |

The planning and cost checkpoints remain retained history. The completed6.1 checkpoint supersedes their open choices, with scientific authority in Phase6-D1 and runtime behavior in its implementation owner. The first release is finished; the reviewed cost/artifact exception now gates another release and later study selection.
