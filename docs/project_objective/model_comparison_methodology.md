# Comparative model and vocabulary-reduction methodology

**Created:** 2026-08-12
**Last updated:** 2026-09-24
**Status:** Active and binding design; Subphase 4B has frozen the EfficientNetV2-S reference-selector protocol, Phase 3-D1 has frozen its campaign and measurement contract, and the later benchmark also requires the independent Subphase 4A model portfolio.

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
campaign-side protocol under Phase 3-D1. It does not prescribe the numerical
ingredient promotion gates, Phase 6 HPO budget, or final selected vocabulary.

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
| V_selected | A versioned, shared projection of V_base produced by Macro-section 3 with M_ref, the learnability decision profile, semantic evidence, and observability review. |
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

The R2 smoke established only that a direct 165-logit adapter can load,
differentiate, and fit the development GPU. The current
[model implementation inventory](../implementation_details/models.md) does not
yet include the EfficientNetV2 wrapper, the exact full-frame transform, per-label AP
trajectories, or the complete provenance manifest. Macro-section 3 must
implement and validate them; 4B-D1 must not be described as current runtime
support until those gates pass.

### 2. Freeze the Phase 3 campaign and measurement protocol

#### Phase 3-D1 — Frozen selector campaign and measurement protocol

**Adopted:** 2026-09-24. This decision completes P1 without inspecting a new
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

The single 20-epoch run emits all 165 logits because the frozen head is shared,
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

The campaign must start from a clean tracked worktree. Large checkpoints, raw
scores, and generated reports remain outside `docs/`; durable documentation
records only reviewed decisions and results. P2 owns schema tests, class-order
and hash validation, test-split access denial, deterministic rerun checks, and
the complete instrumented 8 GB resource gate.

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
| Macro-section 3 | Implement the frozen 4B-D1/Phase 3-D1 workflow and produce versioned V_selected evidence. | P1 is done; P2 implementation is next under [`recognizable_ingredient_selection.md`](../plans/recognizable_ingredient_selection.md). |
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
