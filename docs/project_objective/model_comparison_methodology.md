# Comparative model and vocabulary-reduction methodology

**Created:** 2026-08-12
**Last updated:** 2026-10-06
**Status:** Active and binding design through D6 and its P6 projection freeze. D4 remains the immutable original pilot-frozen profile; D6 adopts the reviewed, outcome-informed inclusion policy without changing the training campaign or default vocabulary. P6 publishes one shared selected vocabulary. The later benchmark also requires the independent Subphase 4A model portfolio.

## Purpose and scope

This document defines the research methodology that couples ingredient selection
to model training and comparison. It is the binding owner of the distinction
between:

1. comparing model categories fairly on a common prediction task;
2. testing whether a learnability-selected vocabulary helps beyond an arbitrary
   reduction in output labels; and
3. obtaining a well-adapted model for the reduced task.

It applies to Macro-section 3, Subphases 4A and 4B, and Macro-sections 6 and 7 of
[general_plan.md](../general_plan.md). The selection workflow remains owned by
Macro-section 3 and its operational plan. Training implementation, hyperparameter
optimization (HPO), and final result production remain owned by Macro-sections 6
and 7. It prescribes the model-side `M_ref` protocol under 4B-D1 and its
campaign-side protocol under Phase 3-D1, and the original numerical profile
gates under D4 and the revised inclusion policy under D6. P6 freezes the
shared selected vocabulary below; the Phase 6 HPO budget remains a later decision.

## Research questions

The methodology keeps these questions separate because they require different
comparisons.

| ID | Question | Required common task |
| --- | --- | --- |
| Q1 | Which model category performs best on the complete normalized ingredient task? | The frozen full v5 vocabulary. |
| Q2 | Which model category performs best on the learnability-selected task under the matched transferred procedure? | One shared selected vocabulary. |
| Q3 | Does the selected vocabulary help more than reducing the number of labels at random? | Like-for-like selected and support-matched random vocabularies, evaluated by the reference model. |
| Q4 | What is the best performance a model can attain after adapting its hyperparameters to the selected task? | The shared selected vocabulary, reported separately from Q3. |

Q1 and Q2 compare model categories. Q3 is a vocabulary-reduction ablation. Q4
is an optimized reduced-task result. Conflating them would either compare
different prediction problems or attribute a hyperparameter change to the
vocabulary selection.

## Terms and fixed boundaries

| Term | Meaning |
| --- | --- |
| V_base | The frozen 165-label FoodOn-first v5 vocabulary derived from ingredients_target. It is the common full task. |
| M_ref | The 4B-D1 reference selector: Torchvision EfficientNetV2-S with the exact supervised ImageNet initialization, full-backbone adaptation, full-frame 384-pixel transform, independent 165-logit pooled head, and weighted BCE boundary frozen below. It is a selection instrument, not the automatically preferred final model. |
| V_selected | A versioned, shared projection of V_base produced by Macro-section 3 from M_ref's frozen numerical learnability profile on the existing recipe-ingredient targets; manual relevance or observability judgments do not determine membership. |
| V_random^(r) | One deterministic random projection of V_base with the same cardinality as V_selected and support strata matched to it; r identifies the draw. |
| H_base(m) | Hyperparameters selected for model category m on V_base using validation only and the predeclared Phase 6 budget. |
| H_local(m) | A small, predeclared local adaptation panel around H_base(m) for V_selected. It is not a second unrestricted HPO campaign. |
| Q_q | One of the frozen headline metrics from D9: label-macro mAP or micro F1. Both are reported; no single scalar silently replaces the pair. |

Every selected or random vocabulary is an explicit experimental projection. It
never replaces the default ingredients_target vocabulary, and validation or test
labels never expand or reorder its saved class order.

## Model-research evidence and decision boundary

Subphases 4A and 4B share reusable evidence: broad discoveries, primary-source catalogs, architecture and pretraining descriptions, current integration audits, licence/checkpoint facts, and measured resource constraints. The same evidence may be cited by both without being duplicated.

They own different decisions. Subphase 4A chooses model categories for the experiment; Subphase 4B chooses the single learnability measurement instrument `M_ref`. Eligibility, exclusion, or ranking in one subphase does not transfer automatically to the other, even when the same architecture appears in both.

### Subphase 4A experiment-portfolio design

Subphase 4A follows the dedicated
[experimental-model research plan](../plans/experimental_model_research.md).
It first discovers three to five family-level candidates, treating close depth,
width, checkpoint, and size variants as one candidate. It then completes the
same deep-research evidence schema for every retained family and selects exactly
two established families through hard eligibility gates and a qualitative,
source-linked comparison.

The adopted families and their protocol/interpretation boundaries are owned by
the [experimental portfolio](experimental_model_portfolio.md). That decision
also records the handoff to custom research; it does not change Q1–Q4 or select
`M_ref`.

The third intended experiment category is one project-specific attention model.
Its larger research stream receives a separate feature plan, re-synthesizes the
problem and the established-family evidence, studies reusable attention
components, proposes three coherent network topologies with small/medium/large
scales, and selects one topology for implementation. Scale variants do not count
as separate candidates or architectures.

This design yields two literature-derived families plus one custom family for
Macro-sections 5–7. Required non-visual and simple visual baselines remain
benchmark controls and are not automatically counted among those three
categories. The portfolio is selected without candidate training, HPO, or test
access; concrete implementations and resource smoke measurements belong to
Macro-section 5, while tuning and comparative performance belong to
Macro-sections 6–7.

## Binding design

### 1. Freeze the reference selector in Subphase 4B before vocabulary selection

Subphase 4B chose and froze M_ref before new v5 ingredient-selection training
begins. The decision used focused model research and a declared selection
protocol: scientific fit to multi-label visual learnability, availability of
per-label score trajectories, representativeness, compute cost, and integration
feasibility. It was not chosen from a candidate training tournament, test
result, or selected-vocabulary outcome.

Macro-section 3 then owns the execution: it implements the binding model- and
campaign-side protocols, runs the learnability profile without test access, and
produces exactly one shared V_selected plus the associated evidence and
provenance. A different
selected vocabulary for each model category is rejected for the primary study,
because it would make a model comparison a comparison of different tasks.

Subphase 4A independently selects the model categories for the benchmark.
M_ref is not thereby declared the winning category; it only fixes the operational
meaning of “learnable” for the vocabulary-selection study.

#### 4B-D1 — Frozen reference-selector protocol

**Adopted:** 2026-09-15. The model-side selector identity is frozen below.
The complementary campaign-side configuration is now frozen under Phase 3-D1.
Neither decision may be revised silently after label outcomes are inspected.
The execution amendment Phase 3-D2 below supersedes the original batch and
clean-worktree requirements. Phase 3-D3 sets the active 40-epoch budget for
the replacement `phase3-d1-v3` campaign.

| Field | Binding choice | Consequence or implementation requirement |
| --- | --- | --- |
| Task and outputs | Frozen FoodOn-first `v5` train/validation split, ordered 165-label encoder, raw independent logits. Test data is unavailable. | Validation and test cannot add or reorder labels. The exact encoder classes and their hash travel with every run and checkpoint. |
| Constructor and weights | Torchvision [`efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)`](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.efficientnet_v2_s.html) under Torchvision `0.23.0+cu129`. The official artifact URL ends in `efficientnet_v2_s-dd5fe13b.pth`; the verified file is 86,721,253 bytes with SHA-256 `dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74`. | Use the exact enum, not `DEFAULT`. Preserve the URL, size, hash, Torch/Torchvision versions, [BSD notice](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE), and successful offline-load check in provenance. |
| Pretraining boundary | Supervised ImageNet-1K initialization only; no food-domain, text, label-graph, or downstream ingredient pretraining. | “Learnable” remains conditional on this visual prior and does not become an intrinsic label property or proof of direct visibility. Possible pretraining-data overlap is not established as absent. |
| Trainability | Train every backbone, BatchNorm affine parameter, BatchNorm running statistic, and new head from the first optimizer step. Do not freeze stages, use a linear-probe warm-up, or apply layer-wise progressive unfreezing. | The selector measures acquisition under local end-to-end supervised adaptation (`P0 + A3`), not fixed-representation accessibility. The implementation must assert the trainable parameter set. |
| Full-frame input transform | Decode as three-channel RGB, convert to a float tensor in `[0,1]`, then set the long side to 384 and round the short side to the nearest integer with half rounded up. Resize once with bilinear interpolation and antialiasing. Center-pad to 384×384 with the ImageNet mean `(0.485, 0.456, 0.406)`, assigning an odd residual pixel to the right or bottom, then normalize by mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. | The deterministic fit/pad geometry is identical for train and validation before the train-only flip, and preserves the complete image without stretching or center cropping. P2 must test landscape, portrait, square, odd-padding, and RGB-conversion cases and serialize the transform identity and parameters. This deliberately differs from the checkpoint's official center-crop evaluation transform. |
| Primary training augmentation | Apply only `RandomHorizontalFlip(p=0.5)` after full-frame fit/pad and before normalization. Validation is deterministic and has no stochastic augmentation. No random crop, rotation, vertical flip, colour transform, `TrivialAugmentWide`, MixUp, or CutMix belongs to the primary selector configuration. | The primary learning trajectory cannot lose border evidence through augmentation. Phase 3-D1 adopts no additional training-time robustness configuration, so this policy is the only campaign augmentation. |
| Independent head | Retain the stock adaptive global average pool and flattening. Retain `Dropout(p=0.2, inplace=True)`, replace only the 1000-way linear layer with a newly instantiated biased `Linear(1280, 165)`, and return raw logits. Instantiate the layer after the declared global seed using the pinned PyTorch `2.8.0` default initialization; record an initial head-state hash. | No label text, dependencies, query decoder, per-label architecture, sigmoid layer, or pretrained classifier weights enter the head (`H0`). Class count comes from the saved encoder rather than a hard-coded label list. |
| Loss family and weighting | `BCEWithLogitsLoss(reduction="mean")` with train-only `pos_weight[c] = (N_train - P_c) / P_c`, where `P_c` is the number of positive train records for label `c`. No label smoothing, focal term, or validation/test-derived weight is allowed. | Persist the ordered positive counts, `pos_weight` vector, formula, and hash. P2 must implement this selector-specific vector rather than silently reuse the current DataModule's differently normalized [`classes_weights`](../../src/data_processing/images_recipes.py). Weighting improves sensitivity to minority positives but changes calibration; AP remains ranking evidence and later probability calibration stays validation-only. |
| Execution boundary | Primary implementation target: true FP32, physical batch 8, no gradient accumulation, on the 8 GB development GPU. R2 measured 3,744.1 MiB peak allocated and 4,250.0 MiB peak reserved for a synthetic full forward/BCE backward/Adam step at 384 pixels. | This is engineering feasibility, not throughput or accuracy evidence. The complete instrumented pipeline must repeat the measurement. A physical-batch or precision change alters optimization/BatchNorm semantics and requires a recorded protocol revision before campaign execution. |
| Campaign and logging handoff | Use one declared seed per configuration, a fixed epoch budget without primary-run early stopping, and per-label train/validation AP at every declared evaluation point. | Phase 3-D1 below fixes the remaining values and instrumentation. No seed-level stability claim is permitted. |

The exact resize rule can be expressed without floating ambiguity: if `W >= H`,
set `(W', H') = (384, max(1, floor(384 * H / W + 0.5)))`; otherwise set
`(W', H') = (max(1, floor(384 * W / H + 0.5)), 384)`. Set
`left = floor((384 - W') / 2)`, `right = 384 - W' - left`,
`top = floor((384 - H') / 2)`, and `bottom = 384 - H' - top`.
Padding is applied in float space before normalization so the padded region is
exactly zero after normalization.

This selector protocol is independent from the 4A comparison protocol even
though EfficientNetV2-S appears in both roles. The 4A portfolio currently uses
the common 224-pixel comparison boundary and removes classifier dropout;
4B-D1 instead uses the 384-pixel full-frame boundary and retains stock dropout.
Configurations cannot be shared silently between those roles.

The earlier R2 smoke established only that a direct 165-logit adapter could
load, differentiate, and fit the development GPU. Phase 3 P2 has now promoted
that prototype into maintained runtime support: the exact wrapper, full-frame
transform, weighted loss, deterministic AP/F1 audit path, blind cohort gate,
and provenance artifacts are implemented and covered by the repository suite.
The real-data batch-8 FP32 resource gate passed on the development RTX 4060.
See the [implementation contract](../implementation_details/ingredient_selection.md).
This implementation result does not contain a label outcome or revise 4B-D1.

### 2. Freeze the Phase 3 campaign and measurement protocol

#### Phase 3-D1 — Frozen selector campaign and measurement protocol

**Adopted:** 2026-09-24. The original values remain as historical evidence:
Phase 3-D2 supersedes batch/provenance, and Phase 3-D3 supersedes the budget,
cosine duration, audit horizon and final analysis windows. Other fields remain
binding. This decision completes P1 without inspecting a new
`v5` selector outcome. The numeric profile-promotion gates remain a P3 pilot
output, but the data that may inform them, the optimization path, measurements,
controls, and isolation procedure are fixed here.

| Field | Binding choice | Consequence or implementation requirement |
| --- | --- | --- |
| Primary configuration | Exactly one primary configuration and no additional training-time robustness configuration. Run it once with global seed `42`. | Configuration sensitivity is unavailable rather than assumed. Borderline evidence must remain `uncertain`; no seed- or configuration-stability claim is permitted. |
| Optimizer | One parameter group containing every trainable backbone and head parameter; `torch.optim.AdamW(lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=False, foreach=False, fused=False)`. Apply decay uniformly; use no gradient clipping. | AdamW keeps weight decay separate from the adaptive moments, while the explicit single-tensor path avoids the additional peak memory documented for `foreach`. The learning rate and decay are predeclared conservative values, not an optimality claim. See the [AdamW paper](https://openreview.net/pdf?id=Bkg6RiCqY7) and [PyTorch 2.8 API](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html). |
| Learning-rate schedule | Step once per epoch. Use `LinearLR(start_factor=0.1, end_factor=1.0, total_iters=2)` followed through `SequentialLR(milestones=[2])` by `CosineAnnealingLR(T_max=18, eta_min=1e-6)`. Do not restart or react to validation metrics. | The first two epochs protect the pretrained representation and new weighted head from an abrupt full update; the remaining fixed cosine path reaches the declared floor at the end of the budget without selecting a schedule from outcomes. The implementation follows the pinned [LinearLR](https://docs.pytorch.org/docs/2.8/generated/torch.optim.lr_scheduler.LinearLR.html), [SequentialLR](https://docs.pytorch.org/docs/2.8/generated/torch.optim.lr_scheduler.SequentialLR.html), and [CosineAnnealingLR](https://docs.pytorch.org/docs/2.8/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) semantics. |
| Training budget | `20` complete epochs, no early stopping, no pruning, no SWA, physical batch `8`, no accumulation, and no automatic batch-size change. With 47,965 training records and `drop_last=False`, this is 5,996 optimizer steps per epoch and 119,920 planned optimizer steps. | The profile measures acquisition under one fixed resource budget. Only a non-finite value, failed invariant, interrupted run, or resource-gate failure may abort the campaign; changing the protocol requires a versioned pre-outcome revision. |
| Precision and randomness | Lightning `precision="32-true"`; no autocast or gradient scaler. Call `seed_everything(42, workers=True)` before model/head and DataLoader construction; seed the train generator with `42`; set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, disable cuDNN benchmarking, request deterministic algorithms, set `torch.set_float32_matmul_precision("highest")`, and disable TF32 through both `torch.backends.cuda.matmul.allow_tf32=False` and `torch.backends.cudnn.allow_tf32=False`. Preserve the resolved PyTorch matmul/TF32 flags in the manifest. | P2 must fail clearly if the pinned stack cannot execute deterministically instead of falling back silently. Reproduction remains conditional on the same release, platform, and device, as stated by the [PyTorch reproducibility note](https://docs.pytorch.org/docs/2.8/notes/randomness.html). |
| Batch sampling | Shuffle all training records without replacement each epoch through the seeded generator; use no class-aware sampler and `drop_last=False`. Audit train and validation passes are ordered, deterministic, use the validation transform, and run with dropout and BatchNorm in evaluation mode. | The class imbalance intervention remains solely the frozen `pos_weight`; batch composition cannot become an unrecorded second intervention. Audit train AP is not accumulated from batches produced while the weights are changing. |
| Evaluation cadence | Run full deterministic train and validation audits before optimization (`epoch=0`) and after epochs `2, 4, 6, 8, 10, 12, 14, 16, 18, 20`. Log ordinary training loss and learning rate every training epoch. | Train and validation AP at a named checkpoint refer to one fixed model state and identical evaluation geometry. The extra train pass is intentionally separated from stochastic training. |
| AP trajectory statistics | For each label and split retain AP at every audit point. Define `W_early={2,4,6}`, `W_near={10,12,14,16,18}`, and `W_late={12,14,16,18,20}`. Precompute initialization-to-late gain, early-to-late median gain, late median, late IQR, near-versus-late median shift, and the late train-minus-validation gap. | These are candidate profile inputs, not numerical inclusion gates. P3 may choose absolute gates only from the isolated pilot cohort; it may not replace the robust windows with a maximum or tune a different window per label. |
| F1 policy | At every audit point apply `sigmoid(logit) >= 0.5` globally and report per-label precision, recall, and F1 with zero-division value `0`, plus aggregate micro F1. Do not calibrate, search, or vary this threshold by epoch or label. | F1 is a historical-continuity and operating-point diagnostic only. It cannot promote or reject a label in the Phase 3 learnability profile. Later benchmark threshold selection remains validation-only under D10. |
| Finite-sample uncertainty | At the final checkpoint, compute a deterministic 95% percentile interval for each validation AP from 1,000 record-level bootstrap resamples. Use `42_000 + class_index` as the bootstrap seed for that label; reject resamples lacking either class and record any label for which 1,000 valid samples cannot be obtained within 10,000 draws. | This interval describes validation-sample uncertainty only. It is not a seed, training-run, or temporal-stability interval and cannot compensate for the single-run design. |
| Low-cost controls | Always report train/validation support and prevalence, the `epoch=0` reference, Spearman associations between support/prevalence and profile statistics, a constant train-prevalence baseline, and a train-only cuisine-prior baseline with `(P_c,l + 1)/(N_c + 2)` and global train prevalence for an unknown cuisine. Compare late validation AP with both baselines. | Cuisine is a diagnostic unavailable to the image-only model, not an inference input or fair competitor. A positive image advantage does not establish direct visibility. Co-occurrence and shuffled-label controls are not primary requirements; a shuffled control requires a recorded pilot addendum if cheaper controls cannot distinguish signal from an artefact. |
| Vocabulary-reduction controls | Do not train selected or random reduced vocabularies during P1–P4. Preserve the support and prevalence fields needed by Macro-section 6 to generate deterministic support-matched random vocabularies after `V_selected` is known. | This prevents Phase 3 from paying for or interpreting the later Q3 ablation prematurely. The selected-versus-random training comparison remains owned by Macro-section 6. |

#### Pilot isolation without a second selector training

P2 must generate the exact P3 pilot cohort before model construction and before
any new selector metric is read. Sort the 165 labels by `(train positive count,
class name)`, split that ordered list into three contiguous 55-label support
strata, rank labels inside each stratum by the SHA-256 digest of
`phase3-pilot-v1\0<label>`, and take the first eight from every stratum. Persist
the resulting 24 names, indices, supports, generation string, and manifest hash.

The single sealed run (originally 20 epochs, now 40 under Phase 3-D3) emits
all 165 logits because the frozen head is shared,
but P3 analysis is allowed to expose only these 24 labels. P3 freezes simple
absolute gates and an `uncertain` band from that cohort; it may not optimize a
fixed retained count. A machine-readable rule file and its input hashes must
exist before P4 can expose or classify the remaining 141 labels. P4 then applies
the frozen rule to the same sealed run, so the pilot does not require another
full training. Any optimizer, schedule, transform, loss, seed, budget, or audit
change invalidates that reuse and creates a new protocol version.

#### Frozen output and provenance boundary

One versioned report under
`analysis_outputs/ingredient_selection/<protocol_id>/` must contain at least:

- `campaign_manifest.json`: protocol ID, clean Git revision, command, UTC times,
  environment and device, determinism flags, exact 4B-D1 fields and hashes,
  optimizer/scheduler state, seed, metadata hashes, ordered classes and hash,
  positive counts/weights and hash, transform identity, and initial head hash;
- `pilot_cohort.json` and, after P3, `profile_rule.json` with their source hashes;
- `metrics_per_label_epoch.csv`: one row per run, audit epoch, split, and class,
  including support, prevalence, AP, fixed-policy precision/recall/F1, and the
  actual learning rate/configuration identity;
- compressed validation record IDs, targets, and logits for every audit point,
  sufficient to rebuild PR curves and final bootstrap intervals;
- `profile_evidence.csv`, named projections, plots, and
  `validation_summary.json`, each generated from validated inputs rather than
  notebook state.

The original v1 campaign required a clean tracked worktree; Phase 3-D2 permits
the main workspace with an immutable content-addressed source snapshot. Large checkpoints, raw
scores, and generated reports remain outside `docs/`; durable documentation
records only reviewed decisions and results. P2 owns schema tests, class-order
and hash validation, test-split access denial, deterministic rerun checks, and
the complete instrumented 8 GB resource gate.

#### Phase 3-D2 — Effective-batch and main-workspace execution amendment

**Adopted:** 2026-09-27, following the user's explicit request before any
per-label outcome was inspected. `phase3-d1-v1` was interrupted during epoch
1 (zero-based), retained as a superseded incomplete campaign, and cannot be
combined with the replacement evidence. The replacement starts from the exact
pretrained weights and a newly seeded head, never from the interrupted run or
a disposable capacity-test model. Its planned protocol ID was `phase3-d1-v2`.
Before that campaign started, the user adopted Phase 3-D3 below; D2's batch
and provenance rules remain binding for v3.

The requested effective batch is **128**. Test physical divisors in descending
order (`128,64,32,16,8,4,2,1`) and set the model's `MAX_ALLOWED_BATCH_SIZE` to
the largest passing physical divisor. Lightning uses
`accumulate_grad_batches = 128 / physical_batch_size`. Capacity is resolved
before campaign training and cannot change adaptively during the run.

On the development RTX 4060, quick trials at 128, 64, 32 and 16 raised CUDA
OOM, while physical 8 completed two accumulated AdamW steps. The current cap
is therefore **8** with accumulation **16**, conditional on a mandatory full
disposable epoch plus deterministic validation-inference gate. All tests use
the same real train data, weighted BCE, FP32, full adaptation, and optimizer
settings as the campaign. They expose no ingredient AP/F1 outcomes and their
models are discarded. Raw capacity records are under
`analysis_outputs/ingredient_selection/capacity_v2/`.

The allocator is capped using
`min(total_VRAM - 512 MiB, free_VRAM_at_start - 256 MiB)` through
[PyTorch's per-process memory limit](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.memory.set_per_process_memory_fraction.html).
The cap prevents a trial from being accepted through host/shared-memory
oversubscription, observed locally in the initial unbounded batch-128 OOM.
Both the gate and campaign record the resolved limit.

With 47,965 records, each epoch has 374 groups of 128 plus a final group of
93: **375 optimizer updates per epoch and 7,500 over 20 epochs**. Keep all
records (`drop_last=False`). Since Lightning divides each microbatch loss by
the accumulation count, scale the returned loss by
`accumulation * actual_microbatch_records / actual_group_records`; this gives
the final incomplete group its correct sample-mean gradient. A CPU Lightning
test compares the actual updates with direct full-group BCE updates.

The LR, decay, warm-up, cosine schedule, epoch budget, seed, transforms,
weights, audits, pilot rule, and analysis statistics remain fixed. The number
of optimizer updates changes substantially; the old and new runs are not
equivalent optimization protocols. No linear learning-rate scaling is assumed
or tuned from outcomes. Accumulation aggregates gradients, whereas
[BatchNorm](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BatchNorm2d.html)
continues to use the physical microbatch's statistics; effective 128 does not
give BatchNorm 128 simultaneous examples.

The canonical rerunnable launcher is
[`train_selector.py`](../../scripts/launch_exps/ingredient_selection/train_selector.py).
It runs the full-epoch gate in a fresh subprocess, then starts a fresh campaign
only if that gate passes. Code provenance is the Git base revision **plus** a
SHA-256 inventory and saved ZIP of the exact Python sources, including new
maintained source/launcher files. Record the actual tracked-worktree status;
the gate must match the same revision, source-content hash, batch plan, data,
weights, and worker settings. This permits execution in the main workspace
without requiring unrelated local changes to be committed. Data, secrets,
raw outputs, and checkpoints are excluded from the source snapshot.

#### Phase 3-D3 — Forty-epoch campaign amendment

**Adopted:** 2026-09-27 by explicit user request, before any per-label outcome
inspection and before the v2 campaign started. The disposable v2 full-epoch
gate was interrupted; no completed gate or training result is reused. The
active protocol ID is **`phase3-d1-v3`**. After verification and commit, rerun
the one-epoch capacity gate against the exact committed source, then construct
a freshly initialized campaign model. Never resume v1 or the discarded gate
model.

| Field | Active binding value | Rationale |
| --- | --- | --- |
| Training budget | 40 complete epochs, no early stopping | Explicit budget extension, not an outcome-selected stopping point. |
| Scheduler | The same 2-epoch linear warm-up, then `CosineAnnealingLR(T_max=38, eta_min=1e-6)` | Extend the decay through the new horizon without a restart or unrequested LR scaling. |
| Audit cadence | Initialization plus every 2 epochs through 40: 21 fixed-state train/validation audits | Preserve the measurement cadence and include the actual final checkpoint. |
| AP windows | `W_early={2,4,6}`, `W_near={30,32,34,36,38}`, `W_late={32,34,36,38,40}` | Preserve the early acquisition reference and the five-point end-of-budget windows; do not analyze epoch 20 as the final state. |
| Optimizer-update budget | 375 per epoch, 15,000 over 40 epochs, with the same correctly weighted final group of 93 records | The effective-128 sampling arithmetic is unchanged. |
| Final uncertainty | Bootstrap from epoch-40 validation scores | Keep the same statistic and resampling policy at the actual final model state. |

All other D1/D2 choices remain unchanged, including effective 128, physical 8
and accumulation 16 on the development GPU, true FP32, weights, seed, loss,
initial LR and decay, transforms, pilot-generation rule, and blind P3/P4 reuse.
Capacity testing remains one disposable full epoch, not a 40-epoch experiment.
The manifest and analysis validate the 40-epoch budget and 38-epoch cosine
duration explicitly. Earlier campaign and capacity artifacts remain separate
historical evidence and are never pooled with v3 selection evidence.

#### Phase 3-D4 — Pilot-frozen numerical profile rule

**Adopted:** 2026-09-28, after the complete v3 campaign and inspection of only
the precommitted 24-label pilot. This is the P3 numerical-gate freeze permitted
by D1; it does not revise the learner, seed, data, metrics, windows, budget,
audits or the P3/P4 isolation rule. The other 141 label outcomes were not
inspected and must remain sealed until a separate P4 execution.

The machine-readable authority is
[`profile_rule.json`](../../analysis_outputs/ingredient_selection/phase3-d1-v3/profile_rule.json)
(artifact hash `7cf03371245860bf1a5be0c61a9fe54282e358fc910f21d1da6204f91354cda1`).
It binds the campaign identity, hashed 24-label cohort, immutable pilot-evidence
copy, and classifier source hash. The [reviewed pilot result](../experiment_results/phase3_d1_v3_pilot.md)
owns the observed 24-label outcomes, not this methodology record.

| Gate | Absolute value | Role |
| --- | ---: | --- |
| Minimum final-train positive support | 500 | Lower support remains `uncertain`; this is measured after the final recipe filters, not assumed from the target-builder's nominal support filter. |
| Minimum initialization-to-late train-AP gain | 0.10 | Require acquisition beyond the pretraining/random-head reference. |
| Minimum late-median train AP | 0.35 | Require a sustained absolute train ranking level, not one maximum. |
| Minimum late-median validation AP | 0.20 | Require an absolute held-out ranking level, with bootstrap overlap assigned `uncertain`. |
| Maximum late-window train/validation AP IQR | 0.03 each | Flag temporal dispersion rather than selecting a favourable epoch. |
| Maximum late train-minus-validation AP gap | 0.50 | Mark a large optimization/generalization divergence `uncertain`. |
| Minimum late image-minus-cuisine-prior AP advantage | 0.10 | Require a material image-model margin over the declared non-visual diagnostic. |

The classification order is fixed in
[`classify_profile`](../../src/ingredient_selection/metrics.py): missing,
invalid-bootstrap or low-support evidence is `uncertain`; failed train gain or
level is `no_sustained_optimization`; clearly sub-gate validation AP is
`optimization_only`; excessive IQR/gap is `uncertain`; a clearly sub-gate
image advantage is `context_predictable`; and only a fully passing profile is
`generalizable_candidate`. A case near either validation or image-advantage
gate is `uncertain` when the epoch-40 validation AP bootstrap interval straddles
the corresponding absolute boundary. The image comparison subtracts the
point-estimated cuisine-prior AP from both bootstrap AP bounds. That interval
is a conservative screen, **not** a formal interval for the late-window median
or the model-minus-prior difference. F1 never changes these outcomes.

This rule has no retained-label quota. Its pilot calibration does not certify
seed/configuration stability, direct visual observability, semantic relevance,
or a final selected vocabulary. P4 may apply it unchanged to the same v3 run
only after the separate execution gate. Phase 3-D5 below supersedes the manual
P5 gate; P6 owns the numerical-profile-to-projection decision.

#### Phase 3-D5 — Numerical selection and optional interpretation appendix

**Adopted:** 2026-10-04, by user decision after P4 and before vocabulary freeze.

The primary study assesses model learning on the frozen recipe-ingredient
targets. Manual judgments of semantic relevance or literal visual observability
answer a different question and could introduce subjective membership decisions.
They are therefore outside the primary selection and comparison protocol.

- Retain the existing `v5` targets, D4 thresholds, P4 outcomes, campaign and
  single-run interpretation limits. This amendment neither reclassifies a label
  nor certifies direct visibility.
- Supersede mandatory P5 semantic review, two-human annotation, agreement
  measurement and main-panel execution. P6 can proceed without any manual review.
- In P6, specify and version a deterministic mapping from the frozen numerical
  outcomes to one shared `V_selected`, with explicit handling of excluded and
  uncertain outcomes. Membership must not depend on whether a person finds an
  ingredient relevant or visible. The final projection is not frozen by D5.
- Preserve the prepared rubric, code and unannotated packet as an
  [optional interpretation appendix](ingredient_observability_protocol.md).
  It may support descriptive discussion of future results, but cannot change
  selection membership, targets, thresholds, tuning, or primary model rankings.
  If undertaken, its reviewer agreement and sampling limitations remain explicit.
- Mapping defects remain owned by the existing Data methodology. A discovered
  defect requires a separate versioned data decision, not a manual exception to
  vocabulary selection. Test outcomes cannot inform the projection.

The former P5 dependency is retained in dated planning history as superseded;
it is not a current completion gate for Macro-sections 3, 6 or 7.

<a id="post-p4-inclusion-policy-review--proposed-amendment"></a>

#### Phase 3-D6 — Held-out-quality inclusion policy

**Reviewed:** 2026-10-04–2026-10-05. **Adopted:** 2026-10-05 by explicit user
approval of the proposed policy, before its corrected resampling is executed.
This outcome-informed amendment supersedes D4's inclusion vetoes, not its
historical profile or original evidence. It does not itself export a vocabulary.
The user requested a check of policy correctness against the project objective,
explicitly allowing a small vocabulary if justified. The
[reviewed audit](../experiment_results/phase3_d1_v3_inclusion_policy_audit.md)
owns the measurements and sensitivity results. D4 and its original outputs
remain reproducible without alteration.

The selected task is **recipe ingredients with sufficiently strong,
sustained held-out ranking under the fixed reference-selector protocol**.
Image-derived dish/context information is valid for this task. Acquisition on
train, held-out ranking, and possible signal mechanisms remain separate axes.
The selection is not a census of every label showing any learning, a direct
visibility test, or evidence that excluded labels cannot be learned by another
model. There is no target cardinality.

| Evidence | Adopted inclusion role | Reason |
| --- | --- | --- |
| Complete, finite, class-aligned evidence with both classes represented and valid resampling | Required | An uninterpretable estimate cannot support inclusion. |
| Late validation AP | Retain `0.20` as the inherited operational quality floor; require the lower bound for that **same statistic** to reach it. | This defines a sufficiently strong ranking subset. It is not a literature-standard learnability boundary or 20% ingredient-prediction accuracy. Values below it may still represent real learning. |
| Advantage over a constant image-independent score | Require a positive lower bound for paired `late_validation_AP - validation_prevalence`. | An absolute AP level alone can reward common labels. Recompute AP and prevalence on the same resample; a point prevalence subtracted from an unrelated AP interval is not that interval. |
| Late validation AP IQR | Retain the existing `0.03` dispersion screen; flag failures as uncertain. | It limits within-run oscillation at the fixed budget, not variation between seeds. Keep nearby-window sensitivity visible without choosing a new window per label. |
| Train AP gain, absolute train AP and train–validation gap | Report as optimization/overfit diagnostics, without independent membership vetoes. | Stronger fitting of train examples must not disqualify an otherwise equally useful held-out predictor. Report weak train evidence explicitly rather than inventing an absence-of-learning claim. |
| Cuisine-prior AP and its `0.10` advantage screen | Retain as diagnostics of the comparison with privileged metadata, without a membership veto. | Ground-truth cuisine is unavailable to the image-only model. Failing the margin neither disproves image-based prediction nor identifies the model's causal mechanism. |
| Additional final-train support cutoff of 500 | Report support continuously and retain the low-support flag; do not repeat it as a second hard vocabulary filter. | The base vocabulary already passed the Data support policy. Sampling precision and validity should control evidence quality; the final-filter support cliff is not a biological or statistical boundary. No replacement cutoff is chosen from observed counts. |

For the primary statistic, retain D3's five late checkpoints
`{32,34,36,38,40}` and define `Q_l` as the median of their separate validation
AP values. Do not average their logits into an undeclared ensemble. Estimate
intervals by resampling the **same validation units across all five
checkpoints**, recomputing each AP, their median, and the constant-score AP
(the resampled positive prevalence). Use the original 1,000 valid draws,
95% percentile convention, deterministic per-label seed, and invalid-draw
reporting. Where exact-image groups contain multiple validation records, use
those groups as the resampling unit and carry all their records together.
No new model inference or training is needed to use the saved scores; validation
image bytes may need to be hashed to recover exact-image groups.

The rule promotes a label only when all required evidence and the
validation-dispersion screen pass, `Q_l >= 0.20`, `lower_95(Q_l) >= 0.20`, and
`lower_95(Q_l - prevalence_l) > 0`. Intervals crossing either boundary remain
uncertain; an upper bound below the quality floor means *below the declared
quality floor*, not *unlearnable*. Retain all applicable reasons per axis,
even when another gate has already failed. These are nominal per-label
screening intervals, not simultaneous guarantees, seed uncertainty or unbiased
post-selection performance estimates.

The `0.20` floor is adopted because it preserves the existing absolute
quality requirement while removing conceptually unrelated vetoes. Its practical
adequacy is a project convention, not established by the pilot or cited papers.
Report the fixed exploratory sensitivity panel at `0.15`, `0.20`, and `0.25`
without picking the value that yields a preferred count, ingredient list, or
later model result. A prevalence-adjusted AP can accompany the report, but
replacing the floor by a new normalized cutoff would introduce another
convention and does not automatically establish usefulness.

All 165 label outcomes have already been exposed. Consequently this is an
**outcome-informed exploratory amendment**, not a new blind pilot. Before P6,
implement statistic-consistent resampling and multidimensional reasons,
report membership/sensitivity changes against D4, and freeze the resulting
rule/report separately under `inclusion_d6_v1/`. The earlier legacy-interval
counterfactual counts are not that artifact. The original training campaign,
full-vocabulary benchmark, test isolation, common-task comparison and optional
appendix boundaries stay applicable. No additional training or manual review
is required by this decision.

The executable convention uses 1,000 valid draws with at most 10,000 attempts,
NumPy PCG64 seed `42000 + original_class_index`, equal-probability draws of
as many exact-image groups as observed groups, and all records of every drawn
group with their multiplicity. Group order is first occurrence in the frozen
validation record order. Percentile bounds use linear interpolation at 2.5%
and 97.5%. Scores preserve the original float64 sigmoid/tie semantics. A draw
without positives or negatives is invalid and reported; insufficient valid
draws or invalid evidence cannot promote a label. Inclusion is conjunctive,
with independent reasons retained for every axis. Valid, stable evidence with
an upper quality bound strictly below the floor is `below_quality_floor`;
otherwise a non-passing or conflicting profile is `uncertain`. Equality at
the quality lower bound passes; equality at the excess lower bound does not.
These conventions do not introduce new tuned parameters or a count target.

The adopted executable rule is
[`inclusion_d6_v1/inclusion_rule.json`](../../analysis_outputs/ingredient_selection/phase3-d1-v3/inclusion_d6_v1/inclusion_rule.json),
policy ID `phase3-d6-held-out-quality-v1`, canonical artifact hash
`851c52cc485279895cf369368fe62076ad3be7318a80e9d3109a20b186dbbcdc`.
It binds the retained campaign/D4 identity, exact analysis sources and snapshot,
input hashes, validation image groups and environment. The
[reviewed D6 result](../experiment_results/phase3_d1_v3_d6_profile.md) owns the
resulting counts, membership changes, sensitivity and verification limitations.
P6 publishes the named projection in the separate freeze below.

#### P6 shared projection freeze — 2026-10-05

The user authorized P6 after the D6 policy/report were completed and committed.
`V_selected` is the versioned
[`ingredients_selected_v5_d6_v1`](../../src/ingredient_selection/resources/ingredients_selected_v5_d6_v1.json)
definition, containing exactly the labels marked `included` in the approved D6
report. Keep `uncertain` and `below_quality_floor` out of the primary projection
while preserving their independent reasons. There are no manual exceptions,
new thresholds, sensitivity-derived alternatives or category-specific lists.
The [reviewed publication](../experiment_results/phase3_d1_v3_d6_profile.md#p6-publication--2026-10-05)
records membership counts and artifact identities; the
[implementation contract](../implementation_details/ingredient_selection.md#p6-frozen-projection)
owns serialization and reproduction.

The selected output order is the subsequence of the frozen `V_base` class order,
with original indices retained for projecting full-model predictions. Selection
changes output columns, not the image population: retain the same splits and
all records, including recipes whose projected target vector is all zero.
Do not filter such recipes independently in selected and full-task runs.
Runtime integration is P7, and the 165-label task remains the default and Q1
anchor. Matching control vocabularies and training remain Phase 6 work.

This freeze fixes a common task; it does not eliminate reference-selector bias,
establish seed stability or validate the subset on independent test outcomes.
Do not use its selection-validation AP as unbiased final selected-task accuracy.
Any later policy or membership change needs a new explicit version and record,
not an edit to this artifact or a per-model reselection.

### 3. Tune each model category once on the full common task

For every approved model category m, Macro-section 6 performs one bounded HPO
campaign on V_base and freezes H_base(m) from validation data. The search
objective, budget, transforms, loss policy, early-stopping rule, and one declared
seed per configuration must be fixed before trials begin. The HPO objective must
be compatible with the paired D9 evaluation policy; the current implementation's
val_loss optimization is not by itself a final methodological choice.

These full-vocabulary selected configurations answer Q1. They also provide the
common starting point for the vocabulary-reduction ablation. No model category
is tuned separately for each ingredient, and no test result selects an
architecture, hyperparameter, threshold, or vocabulary.

### 4. Measure vocabulary reduction with transferred hyperparameters

For every approved model category, train one new V_selected run using its
unchanged H_base(m). Compare it with the corresponding full-vocabulary model
after restricting the full-vocabulary evaluation to the same labels in
V_selected. This is the required transfer ablation for Q2 and, for M_ref, the
selected-vocabulary side of Q3.

For each headline metric q, the vocabulary effect for a vocabulary V is:

\[
\Delta_q(V) =
Q_q\!\left[\operatorname{train}(M_{ref}, V, H_{base}(M_{ref})); V\right]
-
Q_q\!\left[\operatorname{train}(M_{ref}, V_{base}, H_{base}(M_{ref})); V\right].
\]

The notation after the semicolon denotes that both models are evaluated only on
the same vocabulary V. For each model category, the analogous restricted-set
comparison is reported. The random-vocabulary control below is run only for
M_ref unless a later, separately budgeted study expands it.

Transferred-hyperparameter results answer a deliberately narrow causal question:
under the same training procedure, does changing the output vocabulary change
performance on the retained labels? They do not establish that H_base(m) is
optimal for the smaller vocabulary.

### 5. Control for arbitrary label removal

Macro-section 6 generates several deterministic V_random^(r) controls before
their runs. Each has the cardinality of V_selected and is stratified by train and
validation positive-support tiers; prevalence matching and semantic-type matching
are added when the final selected set makes either necessary for a credible
comparison. The number of draws and the training budget are frozen in the Phase
6 plan before any control outcome is inspected.

Using M_ref and the unchanged H_base(M_ref), run the same transfer ablation for
every V_random^(r). Report Δ_q(V_selected) alongside every Δ_q(V_random^(r)), for
both mAP and micro F1, rather than comparing absolute scores from different
label sets. This asks whether removing the selected labels helps more than
removing an equally large, comparably supported set of labels by chance.

The controls are an empirical reference distribution, not replicated
stochastic-training runs. With a small number of draws they support a descriptive
comparison, not a formal significance claim. They also do not prove that every
retained ingredient is directly visible.

### 6. Keep the selected-task local adaptation separate

Removing outputs can alter the optimization problem: in this repository,
BCEWithLogitsLoss uses mean reduction by default, and optional pos_weight values
are recomputed from the active output labels. The gradient balance and the best
learning rate, weight decay, momentum, or scheduler can therefore change when
moving from about 165 labels to a selected set of roughly 50.

After the transfer ablation is frozen, Macro-section 6 may run a small local
adaptation panel around H_base(m) on V_selected. Its candidates and equal
per-category budget must be declared in advance; typical dimensions are a small
multiplicative learning-rate neighbourhood and a bounded weight-decay
neighbourhood. It is not an unrestricted second Optuna study. If an optimized
selected-task headline is reported for several model categories, every category
must receive the same declared local-adaptation opportunity.

H_local(m) answers Q4 only. It must never be used to calculate the Q3
selected-versus-random effect, because it changes both the vocabulary and the
training procedure. If the schedule cannot accommodate the local panel, the
project may report the transferred result but must not describe it as the best
attainable selected-task configuration.

### 7. Preserve selection isolation and report the single-run limit

All vocabulary selection, HPO, local adaptation, control design, thresholds, and
calibration use training and validation data only. Test data is accessed only
after these choices and their report schemas are frozen; it is never used to
revise a vocabulary, selector, model category, hyperparameter, or control.

The project has a resource constraint of one declared seed per configuration.
This applies to selection, full-vocabulary HPO confirmation, transferred
ablation, random controls, and local-adaptation runs. Temporal-window checks,
configuration sensitivity, and resampling of held-out predictions may quantify
limited evidence, but they do not estimate run-to-run stochastic variability.
Every final table and claim must state this limitation and classify borderline
vocabulary decisions as uncertain.

## Rejected primary designs

| Design | Why it is not the primary methodology |
| --- | --- |
| Freeze one selected vocabulary using an arbitrary model without Subphase 4B review | The learnability conclusion is model-conditional; the selector must be explicitly justified and frozen first. |
| Select a different vocabulary for each model category after full-vocabulary tuning | It changes the prediction task by category, so model rankings cannot be interpreted as a like-for-like comparison. |
| Aggregate model-specific selected vocabularies after every category has trained, then use the aggregate as the primary benchmark | It delays the shared task until after model outcomes influence it and doubles the campaign before a comparable benchmark exists. It may be a separately labelled exploratory study later. |
| Perform a full second HPO on every selected and random vocabulary | It is too costly for the thesis schedule and confounds the primary vocabulary-effect ablation. |
| Transfer H_base(m) and call the result optimized for V_selected | Output reduction can alter loss scaling and class weights; transferred results are causal ablations, not selected-task optima. |
| Use test results to choose the selector, selected vocabulary, HPO values, local panel, or random controls | It leaks final evaluation information into research decisions. |

## Evidence and rationale

| Evidence | Consequence for this design |
| --- | --- |
| [label_learnability/learnability_assessment.md](../research/topics/label_learnability/learnability_assessment.md) | Learnability is conditional on the declared learner and protocol, so the project must name and freeze a reference selector. Train AP, validation AP, mechanism controls, and observability are separate evidence. |
| [Cawley and Talbot (2010)](https://jmlr.org/papers/v11/cawley10a.html) | Performance estimates with finite-sample variance can be overfit during model selection; selection criteria, vocabulary decisions, and final evaluation must be separated. |
| [Varma and Simon (2006)](https://doi.org/10.1186/1471-2105-7-91) | Reusing the data that selected an optimized configuration as its final error estimate is optimistic; keep test data isolated from all vocabulary and HPO decisions. |
| [Ambroise and McLachlan (2002)](https://doi.org/10.1073/pnas.102102699) | Selecting a subset based on observed predictive evidence is analogous to feature selection and requires its own validation boundary. |
| [PyTorch BCEWithLogitsLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) and the current [loss setup](../../src/lightning/lgn_models.py) | Mean reduction and active per-class positive weights make a reduced output space a changed optimization problem, motivating the transfer-ablation/local-adaptation separation. |

## Execution dependencies and open decisions

| Owner | Required decision or artifact | Status |
| --- | --- | --- |
| Subphase 4A | Define and justify two established model families and one custom attention architecture to compare. | Done: established pair and custom P2-S adopted in the [portfolio](experimental_model_portfolio.md); the [completed 4A plan](../plans/experimental_model_research.md) hands implementation gates to Phase 5. Q1–Q4 are unchanged. |
| Subphase 4B | Choose, verify, and freeze M_ref. | Done: 4B-D1 freezes the EfficientNetV2-S model-side selector and [`reference_selector_research.md`](../plans/reference_selector_research.md) records the completed evidence and handoff. |
| Macro-section 3 | Implement the frozen selector workflow and produce versioned V_selected evidence. | Done through P7: D6 uncertainty, shared projection, opt-in runtime integration and retention checks are complete. Original D4 evidence and the full default remain preserved; the [completed plan](../plans/recognizable_ingredient_selection.md) owns the execution record. |
| Macro-section 5 | Implement and technically qualify the adopted experiment-model protocols. | [Implementation plan](../plans/additional_model_implementation.md) prepared; execution is Pending. Data 2.4/P7 prerequisites are complete, but each model's runtime/resource gates and the later benchmark-policy/evaluation gates remain. |
| Macro-section 6 | Freeze HPO objectives/budgets, random-control count and matching rules, transfer runs, and any equal local-adaptation panel. | Deferred until the selected vocabulary and models are available. |
| Macro-section 7 | Freeze report schemas, evaluate the already selected configurations on test, and keep Q1–Q4 result statements separate. | Deferred until Macro-section 6 completes. |

## Related documentation

- [experimental_model_portfolio.md](experimental_model_portfolio.md)
- [problem_definition.md](problem_definition.md)
- [benchmark_decisions.md](benchmark_decisions.md)
- [general_plan.md](../general_plan.md)
- [experimental_model_research.md](../plans/experimental_model_research.md)
- [reference_selector_research.md](../plans/reference_selector_research.md)
- [recognizable_ingredient_selection.md](../plans/recognizable_ingredient_selection.md)
- [label_learnability/README.md](../research/topics/label_learnability/README.md)
