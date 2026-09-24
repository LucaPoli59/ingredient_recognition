# Reference-selector research and decision plan

**Created:** 2026-08-12
**Last updated:** 2026-09-24
**Linked macro-section and subphase:** [Subphase 4B, Reference-selector research](../general_plan.md#4b-reference-selector-research)
**Overall status:** Done

## Objective

Choose and freeze one reference selector, `M_ref`, before Macro-section 3
starts the new `v5` ingredient-selection study. The selector defines the
model-conditional meaning of “image-learnable”; it is a measurement instrument,
not the automatically preferred final benchmark model.

The binding cross-phase design remains in
[model_comparison_methodology.md](../project_objective/model_comparison_methodology.md).
This plan owns only Subphase 4B and the bounded decision needed to release Macro-section 3.

## Relationship to Subphase 4A

Subphases 4A and 4B may cite the same broad discoveries, primary sources, model-family descriptions, implementation audits, and resource evidence. Reusable evidence remains in the research records rather than being copied into both plans.

This plan applies selector-specific criteria only. A 4B exclusion does not remove a model from the 4A experiment shortlist, a 4A shortlist decision does not select `M_ref`, and the final outputs remain independently justified.

## Why the remaining work is intentionally small

R0 already surveyed the relevant model and pretraining families and mapped them
to the repository, the frozen data contract, and the 8 GB constraint. Repeating
that survey as one dossier per candidate would add work without changing the
decision.

The remaining question is narrower: which credible protocol is sufficiently
sensitive, interpretable, reproducible, and affordable to act as the common
selector? The plan therefore limits the decision to at most three finalist
protocols, checks only uncertainties that can change the choice, and combines
the decision record with the handoff.

## Scope

This subphase will:

- reduce the completed R0 inventory to at most three meaningfully different
  finalist protocols;
- compare those finalists using a short set of mandatory gates and an explicit
  decision priority;
- run only bounded technical checks needed to confirm the selected protocol on
  the current WSL environment and 8 GB GPU; and
- freeze the exact `M_ref` protocol, its interpretation boundary, and the
  requirements handed to Macro-section 3.

## Non-goals

This subphase does not:

- reopen broad model discovery unless every credible R0 path fails a mandatory
  gate;
- produce a systematic review or a separate research document for every model
  family considered in R0;
- train or tune candidates as a performance tournament;
- run the `v5` learnability campaign or generate `V_selected`;
- choose the final benchmark winner or replace Subphase 4A;
- implement a new architecture solely to keep it in the selector comparison;
- access the test split or use selected-vocabulary outcomes to choose
  `M_ref`; or
- claim seed-level stability.

## Progress tracker

**Overall status:** Done
**Current task:** Complete — 4B-D1 is frozen and Macro-section 3 has accepted the handoff.
**Next action:** Maintain 4B-D1 while Macro-section 3 executes P2 in the [recognizable-ingredient plan](recognizable_ingredient_selection.md). P1 has frozen the complementary campaign settings under Phase 3-D1.

| # | Task | Status | Evidence or result |
| --- | --- | --- | --- |
| R0 | Discover candidate families and map them to the frozen `v5` task, repository, instrumentation needs, and compute boundary. | **Done** | The dated [discovery and integration inventory](../research/discovery/2026-08-22/README.md) retain the broad landscape and repository-backed intake tiers. No selector was chosen. |
| R0.1 | Conduct the broad candidate-landscape discovery. | **Done** | The [2026-08-22 discovery](../research/discovery/2026-08-22/README.md) covers supervised, visual self-supervised, vision-language, food-domain, and structured multi-label families with explicit interpretation boundaries. |
| R0.2 | Map credible candidates to the current `v5` task and integration path. | **Done** | The [candidate and instrumentation inventory](../research/discovery/2026-08-22/candidate_integration_inventory.md) records verified, conditional, and deferred paths plus the common observability gap. |
| R1 | Freeze a shortlist of at most three distinct protocols and the decision priority. | **Done** | Retained supervised ResNet-50 full fine-tuning, supervised EfficientNetV2-S full fine-tuning, and frozen DINOv2 ViT-B/14-register linear transfer. The [R1 checkpoint](#r1-completion-checkpoint--2026-09-15) records gate outcomes, grouped exclusions, claim boundaries, and the R2 priority without selecting `M_ref`. |
| R2 | Verify the finalists only as needed, compare them, and choose `M_ref`. | **Done** | Selected supervised EfficientNetV2-S with full-backbone fine-tuning and an independent 165-logit pooled head. The [R2 checkpoint](#r2-completion-checkpoint--2026-09-15) records the source audit, bounded synthetic smoke, interpretation boundary, exclusions, and remaining R3 freeze items; no candidate training or accuracy comparison was run. |
| R3 | Freeze the selected protocol and hand it to Macro-section 3. | **Done** | [4B-D1](../project_objective/model_comparison_methodology.md#4b-d1--frozen-reference-selector-protocol) freezes the exact EfficientNetV2-S model-side protocol, interpretation, resource boundary, and provenance requirements. The handoff remains accepted; Phase 3 has since completed P1 and is ready for P2. |

## Dependencies and fixed constraints

| Dependency or constraint | Status | Consequence |
| --- | --- | --- |
| FoodOn-first `v5` vocabulary and split | Available | Every finalist targets the same 165-label task and class order. |
| Phase 3 decision profile | Available | `M_ref` must be able to provide named-label train and validation AP evidence after the shared instrumentation is added. |
| One declared seed per configuration | Binding | The later campaign cannot claim seed-level stability. This does not require candidate training during Subphase 4B. |
| Test isolation | Binding | No test result, selected-vocabulary size, or downstream ranking may influence the choice. |
| 8 GB development GPU and thesis schedule | Binding | A protocol that requires disproportionate integration or campaign cost is not eligible. |
| Final benchmark shortlist | Independently completed by 4A | Selecting `M_ref` does not declare the final model winner or alter the 4A portfolio. |

## Simplified decision protocol

### R0. Completed evidence base

R0 established three useful intake groups:

- verified paths: torchvision ResNet-50, repairable frozen DINOv2 B/14-register,
  and a maintained-library pool containing ConvNeXt Tiny, EfficientNetV2-S,
  and Swin V2 Tiny;
- conditional paths: compact DINOv3 and SigLIP 2, only if their access,
  dependency, checkpoint, interpretation, and 8 GB issues are resolved without
  a one-off pipeline; and
- deferred paths: broken or redundant DenseNet, unavailable food-domain
  checkpoints, structured/dependency heads, and disproportionate generative
  models.

These groups are retained as discovery evidence, not carried forward as a
requirement to compare every entry.

### R1. Bounded shortlist and decision priority

A finalist must pass all of these mandatory gates:

1. **Task fit:** consume the canonical `v5` images and produce 165 independent
   continuous label scores without changing the split or using downstream
   label-text prompts or dependency reasoning.
2. **Evidence path:** have a credible maintained route to named-label train and
   validation AP, reproducible configuration, and run provenance. The common
   instrumentation may be implemented once in Macro-section 3; it need not be
   duplicated for each finalist now.
3. **Operational fit:** fit the 8 GB GPU and available schedule using one
   declared configuration and seed.
4. **Reproducibility:** use traceable weights, transforms, dependencies, and
   trainability state, with no test access or downstream selected-vocabulary
   feedback.

Retain at most three protocols that represent genuinely different measurement
choices, for example supervised continuity, a modern supervised visual model,
and a visually self-supervised pretrained representation. A conditional
candidate enters only if it removes a clear limitation of the verified paths
and its prerequisites can be satisfied proportionately.

Compare finalists in this priority order:

1. scientific meaning for image-based ingredient learnability;
2. capacity to expose useful per-label visual signal under the declared
   protocol;
3. reproducibility and interpretability of pretraining and limitations; and
4. campaign cost and integration risk.

Do not invent a numeric score. If two finalists remain effectively equivalent,
prefer the maintained, lower-cost, easier-to-audit protocol instead of opening
another experiment campaign solely to break the tie.

#### R1 completion checkpoint — 2026-09-15

R1 retains exactly three protocols from the R0 intake. They cover the smallest
set of measurement choices needed to decide between historical supervised
continuity, a maintained modern supervised learner, and visual
self-supervised linear accessibility. This is an eligibility shortlist, not a
ranking or an `M_ref` decision. Candidate accuracy, selected-vocabulary size,
Subphase 4A portfolio membership, and test evidence were not inspected.

At this stage a gate pass means that R0 and current repository evidence expose
no known disqualifying condition and that any remaining uncertainty has a
bounded R2 verification path. R2 must reject a finalist if its exact protocol
cannot satisfy that verification; plausible parameter counts or historical
runs do not by themselves certify current 8 GB feasibility.

| ID | Protocol carried into R2 | Distinct measurement and rationale | R1 gate outcome | Bounded R2 uncertainty |
| --- | --- | --- | --- | --- |
| `R1-SC` | Torchvision ResNet-50 with traceable ImageNet-1K supervised weights, full-backbone fine-tuning (`P0 + A3`), and one independent 165-logit pooled head (`H0`). | Supervised continuity control. It most closely preserves the historical question “can an end-to-end image learner begin learning this label?” while removing the old F1-only decision rule. | **Eligible.** The current wrapper and retained experiments provide the lowest-risk task and integration path. The shared AP/provenance layer is still required. | Pin the exact weight enum instead of `DEFAULT`; verify transforms, output shape, a backward pass, peak VRAM and elapsed time on current `v5`, and confirm the configuration fields that Macro-section 3 must capture. |
| `R1-MS` | Torchvision EfficientNetV2-S with traceable ImageNet-1K supervised weights, full-backbone fine-tuning (`P0 + A3`), and one independent 165-logit pooled head (`H0`). | Single modern supervised representative. It keeps the prior, adaptation, and head comparable to `R1-SC` while testing whether the instrument should prioritize historical continuity or a more recent efficiency-oriented architecture. It has the lowest published parameter count in the R0 maintained-library pool and needs no new third-party stack; neither fact is treated as accuracy or memory evidence. | **Eligible for bounded verification.** The canonical image/logit contract is direct and the integration path is proportionate; official scale metadata is not treated as measured memory evidence. | Confirm the exact weight-native input policy, instantiate the minimal 165-logit protocol, and measure full-fine-tuning memory/time with the common head and artifacts. |
| `R1-VS` | DINOv2 ViT-B/14 with registers, a frozen visual-self-supervised backbone (`P1 + A0`), and a newly learned independent 165-logit linear head (`H0`). Downstream ingredient text is forbidden. | Distinct representation-accessibility instrument. It asks whether each label is linearly accessible in a broad visual prior at much lower adaptation cost; it does **not** establish that the local end-to-end learner can acquire the representation. | **Eligible only through the bounded reproducibility repair already identified by R0.2.** Earlier project execution supports integration plausibility, not current provenance or resource certification. | Pin upstream source revision and checkpoint checksum, make pretrained/frozen state truthful, freeze preprocessing and register policy, verify 165-score output and head-only gradients, and measure current resource use. |

EfficientNetV2-S is retained independently of its separate Subphase 4A role.
For R1 it represents only the parameter-efficient modern supervised route
selected from the verified torchvision pool; the 4A portfolio decision is
neither evidence for nor against selecting it as `M_ref`.

The four mandatory gates were applied as follows. **Pass-to-R2** means that a
credible bounded path exists, not that R2's current-environment confirmation
has already happened.

| Mandatory gate | `R1-SC` | `R1-MS` | `R1-VS` | R2 release condition |
| --- | --- | --- | --- | --- |
| Task fit | **Pass:** canonical images, 165 independent logits, no label text or dependencies. | **Pass:** same canonical independent-logit contract. | **Pass:** canonical images and learned independent logits; downstream text is prohibited. | Reject any implementation that changes the split, label order, or `H0` output semantics. |
| Evidence path | **Pass-to-R2:** shared Lightning instrumentation can attach named train/validation AP and run provenance. | **Pass-to-R2:** the same shared layer applies after minimal wrapper integration. | **Pass-to-R2:** the shared layer applies to the learned head and its scores. | Confirm a common attachment and serialization path for the required artifacts; implementing that shared layer remains a Macro-section 3 prerequisite. |
| Operational fit | **Pass-to-R2:** strongest historical integration evidence, but no current `v5` measurement. | **Pass-to-R2:** compact maintained model with a direct adapter; full-training memory is unmeasured. | **Pass-to-R2:** historical frozen-backbone execution exists; current resource state is unmeasured. | Record current load/forward/backward compatibility, peak VRAM, physical batch/accumulation, and elapsed time. |
| Reproducibility | **Pass-to-R2:** exact torchvision enum must replace the moving `DEFAULT` alias. | **Pass-to-R2:** exact enum, transform, and new wrapper configuration must be serialized. | **Pass-to-R2 after named repair:** upstream revision, checkpoint checksum, preprocessing, register and freeze state must be explicit. | A finalist becomes selectable only when every identifier and trainability state reconstructs without test or downstream feedback. |

##### Grouped exclusions

- **Redundant supervised representatives:** DenseNet remains broken in the
  model-owned transform contract and adds no distinct measurement principle.
  ConvNeXt Tiny and Swin V2 Tiny are credible maintained alternatives, but
  carrying them alongside EfficientNetV2-S would turn R2 into a supervised
  architecture tournament without changing the `P0 + A3 + H0` claim. If
  `R1-MS` fails a mandatory gate, reopen only this maintained-library slot and
  choose a replacement by the same scientific-fit, auditability, and resource
  rules.
- **Unresolved external priors:** compact DINOv3 and SigLIP 2 do not remove a
  clear limitation of the verified DINOv2 path proportionately. Their access,
  licence/dependency, checkpoint, memory, and interpretation conditions remain
  unresolved; CLIP would duplicate the generic vision-language question.
  Downstream text-prototype scoring is outside the task-fit gate.
- **Unsupported or disproportionate pretraining paths:** MAE/FCMAE, I-JEPA,
  Food2K, Recipe1M+/VLPCook, and FoodSeg/ReLeM lack the complete compact,
  maintained, traceable checkpoint and overlap/taxonomy path required here.
  They remain research evidence, not R2 finalists.
- **Confounded heads and interfaces:** spatial label-query heads,
  dependency-aware heads, graph/set decoders, and generative food VLMs change
  the meaning of learnability or exceed the output/compute contract. They may
  become separately controlled diagnostics, but cannot replace the independent
  `H0` selector in this decision.
- **Subphase 4A-only designs:** MaxViT-T and the custom P2-S topology do not
  enter through portfolio membership. They lack a verified current selector
  path, and P2-S deliberately changes the head measurement. Their experiment
  roles remain untouched by this exclusion.

##### Decision priority handed to R2

R2 must first choose the scientific claim, not the architecture with the most
promising reputation:

1. decide whether the primary instrument should measure end-to-end label
   acquisition under local supervised adaptation (`R1-SC` or `R1-MS`) or
   linear accessibility under a fixed visual self-supervised prior (`R1-VS`);
2. within that claim, prefer the protocol able to expose interpretable
   per-label train/validation AP without downstream text or label-dependency
   information;
3. require exact, auditable pretraining, transform, trainability, data, seed,
   and environment provenance; and
4. only after the scientific and evidence questions are effectively tied,
   prefer the lower-cost maintained protocol with lower integration risk.

The different claim made by `R1-VS` must not be collapsed into an accuracy
comparison with the two full-fine-tuning protocols. R2 may use source
inspection and technical smoke evidence to resolve the choice, but not a
candidate training or AP tournament.

### R2. Decision-relevant verification and selection

Create one concise table for the finalists containing:

- exact architecture and initialization/pretraining;
- frozen, partially trained, or fully trained backbone policy;
- input transform and independent multi-label head;
- what “learnable” would mean under that protocol;
- current integration and provenance gaps;
- licence/checkpoint constraints; and
- current 8 GB evidence.

Inspect current source for every finalist. Run a bounded technical smoke only
where code inspection or existing evidence cannot settle a decision-relevant
question. A smoke may confirm loading, forward/backward compatibility, output
shape, and peak resource use; it is not comparative accuracy evidence.

Choose one `M_ref` from this table. Briefly group excluded alternatives by the
gate or trade-off that mattered. If no finalist passes, reopen only the failed
gate or intake category rather than repeating the broad discovery.

#### R2 completion checkpoint — 2026-09-15

R2 selects **`M_ref = R1-MS`**, namely Torchvision EfficientNetV2-S with
supervised ImageNet-1K initialization, full-backbone adaptation (`P0 + A3`),
and an independent pooled 165-logit head (`H0`). The architecture and
adaptation identity were decided at this checkpoint; R3 still owned the exact
protocol freeze and binding Macro-section 3 handoff at that point.

The choice was made from source inspection at repository commit `861abff` and
a bounded engineering smoke on the current development environment. The smoke
used synthetic inputs and targets only: one warm-up Adam step followed by three
timed forward, BCE-with-logits backward, and optimizer steps in FP32, physical
batch 8, no gradient accumulation, and candidate-native canvas sizes. It did
not load `v5` examples, inspect train/validation AP, access the test split, or
compare predictive quality.

| ID | Exact inspected route | Adaptation, head, input, and meaning of “learnable” | Integration, provenance, and licence boundary | Current 8 GB evidence |
| --- | --- | --- | --- | --- |
| `R1-SC` | Torchvision ResNet-50; the [current wrapper](../../src/models/resnet.py) resolves `ResNet50_Weights.DEFAULT` to `IMAGENET1K_V2`, whose official 0.23 contract uses a 224 crop and 232 resize. | Full-backbone training; global pooled 2048 → 165 linear head; current model-owned train/validation transforms use a 224 canvas. A positive result would mean that this historically continuous supervised learner can acquire the label under local end-to-end adaptation. | Already wrapped, but `DEFAULT` and the wrapper's aspect-ratio-changing tuple resize are not a sufficient exact protocol. R3 would have to pin the enum and transform semantics. Torchvision code is BSD-licensed; inspected checkpoint URL ends in `resnet50-11ad3fa6.pth`, SHA-256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`. | **Pass.** 23,846,117 total/trainable parameters; 1,015.2 MiB peak allocated, 1,174.0 MiB peak reserved; 0.0663 s/step in the bounded 224 smoke. |
| `R1-MS` | Torchvision EfficientNetV2-S with exact `EfficientNet_V2_S_Weights.IMAGENET1K_V1`; the maintained 0.23 weight contract uses a 384 canvas and ImageNet normalization. | Full-backbone training; global pooled 1280 → 165 linear head. A positive result means that a maintained modern supervised image learner can acquire the label under the local end-to-end protocol. At R2, R3 still had to choose between the official center crop and a full-frame aspect-preserving fit/pad policy; 4B-D1 later froze the latter because ingredient evidence may occur near image borders. | No repository wrapper exists yet, but the direct Torchvision adapter loaded and differentiated without a new dependency. Exact enum, transform, head, and trainability must be serialized. Torchvision code is BSD-licensed; inspected checkpoint URL ends in `efficientnet_v2_s-dd5fe13b.pth`, SHA-256 `dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74`. | **Pass.** 20,388,853 total/trainable parameters; 3,744.1 MiB peak allocated, 4,250.0 MiB peak reserved; 0.1435 s/step in the bounded 384 smoke. This is the highest measured cost but remains proportionate to the 8 GB boundary at physical batch 8. |
| `R1-VS` | Official DINOv2 Hub route `dinov2_vitb14_reg_lc`; the [current wrapper](../../src/models/dinov2.py) freezes the ViT-B/14-register visual backbone and replaces its classifier with a 3840 → 165 linear head on a 224 canvas. | Head-only training. A positive result would mean linear accessibility in a fixed visual self-supervised representation, not acquisition of ingredient evidence by local end-to-end learning. | Current wrapper follows an unpinned Hub `main`, always requests pretrained weights even when its serialized flag says otherwise, and first loads then replaces the upstream ImageNet linear head. The cached source has no recoverable Git revision. The two cached files were identified by SHA-256 as `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71` (backbone) and `d046c4caca798f721394e4bf19e2b434061ea61fa1dea729229195ed746a1cab` (upstream head). Official code and weights are Apache-2.0. | **Pass for the frozen protocol.** 87,217,317 total and 633,765 trainable parameters; 439.6 MiB peak allocated, 508.0 MiB peak reserved; 0.0651 s/step in the bounded 224 smoke. The low cost follows from freezing the backbone and is not comparable to full fine-tuning. |

The environment was PyTorch `2.8.0+cu129`, Torchvision `0.23.0+cu129`, CUDA
12.9, and an NVIDIA RTX 4060 reporting 8,187 MiB. All candidates returned
finite `[8, 165]` outputs and finite gradients in the intended trainable head.
The short timings are operational smoke observations, not throughput
benchmarks: they use different resolutions and trainability policies and omit
the data-loader and complete campaign overhead.

The decision follows the R1 priority rather than a model reputation ranking:

1. the primary selector should answer whether a local supervised image learner
   begins to acquire evidence for each ingredient; therefore full end-to-end
   adaptation is a closer match than frozen linear accessibility;
2. within the two supervised protocols, the existing
   [EfficientNetV2 evidence dossier](../research/topics/experimental_model_candidates/efficientnet_v2.md)
   supports the stronger sensitivity hypothesis through a maintained modern
   training-aware CNN and direct food/multi-label precedent, while preserving
   the same independent-head claim;
3. its exact maintained-library initialization is auditable and the direct
   adapter passed the output, gradient, and 8 GB gates; and
4. its greater measured cost is acceptable because scientific sensitivity is
   ordered ahead of cost, and no extra candidate campaign is needed.

This is **not evidence that EfficientNetV2-S is more accurate on `v5`**. No
local predictive comparison was performed, and the separate Subphase 4A
portfolio decision did not select `M_ref`. At the R2 checkpoint, ResNet-50
remained the historical continuity control and bounded fallback in case R3
could not freeze a valid EfficientNetV2-S input/provenance contract; 4B-D1
subsequently passed that gate. Frozen DINOv2 remains a useful
representation-accessibility diagnostic, but it is not the primary selector
because it measures a different estimand and currently has avoidable source
and configuration provenance defects.

Primary implementation contracts inspected for this checkpoint are the
[Torchvision ResNet-50 weights](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.resnet50.html),
[Torchvision EfficientNetV2-S weights](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.efficientnet_v2_s.html),
[Torchvision BSD licence](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE),
and the [official DINOv2 repository and model card](https://github.com/facebookresearch/dinov2).

### R3. Freeze and hand off

Record the selected protocol with enough precision for Macro-section 3 to use
it without reinterpretation:

- model variant and exact weight/checkpoint identity;
- pretraining source and the resulting claim boundary;
- trainability policy, input resolution/transforms, independent output head,
  loss family, and resource boundary;
- known biases and limitations; and
- the shared per-label AP, label-manifest, score-audit, seed, configuration,
  code/environment, and data-provenance artifacts that Phase 3 must implement.

Update
[model_comparison_methodology.md](../project_objective/model_comparison_methodology.md),
[benchmark_decisions.md](../project_objective/benchmark_decisions.md),
[recognizable_ingredient_selection.md](recognizable_ingredient_selection.md),
and the [general plan](../general_plan.md) at this completion checkpoint.
Macro-section 3 then resumes and owns instrumentation, the bounded pilot,
selection thresholds, campaign execution, and `V_selected`.

#### R3 completion checkpoint — 2026-09-15

Subphase 4B is complete. The binding 4B-D1 record freezes:

- Torchvision `efficientnet_v2_s` with exact
  `EfficientNet_V2_S_Weights.IMAGENET1K_V1` weights, verified artifact size and
  SHA-256, and the pinned Torch/Torchvision environment boundary;
- supervised ImageNet pretraining, end-to-end adaptation from the first step,
  and the resulting model-conditional interpretation of learnability;
- an exact RGB 384-pixel full-frame fit/pad transform, primary horizontal-flip
  augmentation, ImageNet normalization, and deterministic validation path;
- stock global pooling and classifier dropout with a freshly initialized
  independent biased `Linear(1280, 165)` logit layer;
- mean-reduced train-positive-weighted BCE and the FP32 physical-batch-8,
  no-accumulation execution target validated by R2's bounded resource smoke;
  and
- the label manifest, weight/hash, trainability, transform, loss, seed,
  configuration, code/environment, data, score-audit, and per-label
  train/validation AP provenance required from Macro-section 3.

The detailed protocol is authoritative in
[4B-D1](../project_objective/model_comparison_methodology.md#4b-d1--frozen-reference-selector-protocol),
and [D12](../project_objective/benchmark_decisions.md#d12-frozen-reference-selector-boundary)
records the benchmark-level decision. The Phase 3 plan accepts these choices as
incoming constraints. Its completed P1 decision now freezes the complementary
optimizer, learning-rate/scheduler, epoch-budget, evaluation-cadence,
single-configuration boundary, and measurement-policy values under
[Phase 3-D1](../project_objective/model_comparison_methodology.md#phase-3-d1--frozen-selector-campaign-and-measurement-protocol).

R3 changes documentation and methodology only. EfficientNetV2-S, the exact
transform, per-label AP trajectories, and the complete manifest are not yet
implemented in the repository. No training campaign, validation AP comparison,
selected-vocabulary outcome, or test access informed the freeze.

## Expected artifacts

- this plan with the R1 shortlist rationale and R2 finalist comparison;
- bounded smoke evidence only when required for the choice;
- one binding `M_ref` decision with an explicit interpretation boundary; and
- a synchronized Macro-section 3 handoff.

New topic-research documents are created only when R1 or R2 produces a reusable
finding not already owned by the R0 discovery. They are not mandatory
per-candidate paperwork.

## Validation and completion criteria

This plan is complete only when:

- the completed R0 discovery and inventory remain linked as the evidence base;
- no more than three scientifically distinct finalists were carried into R2;
- the chosen protocol passes every mandatory gate and any decision-relevant
  technical uncertainty has been checked on the current environment;
- the exact `M_ref` protocol and its model-conditional limitation are recorded
  in the binding methodology;
- Macro-section 3 receives the instrumentation and execution handoff; and
- no test outcome, selected-vocabulary result, or comparative candidate-tuning
  campaign influenced the choice.

## Decision and change log

| Date | Change | Rationale |
| --- | --- | --- |
| 2026-08-12 | Created the plan as former Work package 4.6, now Subphase 4B. | Macro-section 3 is deferred until a research-supported reference selector is frozen independently from the final model shortlist. |
| 2026-08-22 | Opened R0.1 broad discovery and R0.2 technical inventory. | The selector search must not be limited to existing ResNet, DenseNet, and DINO implementations; pretraining is part of the selector protocol and changes its interpretation. |
| 2026-08-22 | Completed R0.1. | The discovery found no universal selector; architecture, pretraining, adaptation, downstream label text, and head structure define different measurements. |
| 2026-08-22 | Completed R0.2 and handed intake tiers to the decision stage. | The inventory identified credible and conditional paths, deferred disproportionate ones, and isolated a shared instrumentation/provenance gap without selecting `M_ref`. |
| 2026-08-27 | Compressed the remaining R1–R5 sequence into R1–R3. | R0 already provides broad evidence. The decision only needs a bounded shortlist, decision-relevant verification, one frozen selector, and its Phase 3 handoff; exhaustive candidate dossiers, a numeric rubric, and separate decision/synchronization stages do not advance the objective. |
| 2026-08-27 | Reclassified the plan as Subphase 4B and separated it from Subphase 4A. | Shared discoveries and technical evidence may support both streams, but this plan owns only the selector criteria, `M_ref` decision, and Phase 3 handoff. |
| 2026-09-15 | Completed R1 with three protocol-level finalists and an ordered decision rule. | ResNet-50 full fine-tuning preserves supervised continuity, EfficientNetV2-S is the sole modern supervised representative, and frozen DINOv2 B/14-register exposes the distinct linear-accessibility question. Grouped exclusions avoid an architecture tournament; no `M_ref` was selected and no candidate was trained. |
| 2026-09-15 | Completed R2 and selected EfficientNetV2-S full fine-tuning as `M_ref`. | The supervised end-to-end protocol best matches local label acquisition; the modern maintained CNN supplies the preferred sensitivity hypothesis and passed bounded 384-pixel output, gradient, provenance-path, and 8 GB checks. ResNet-50 remains a continuity fallback and frozen DINOv2 a differently interpreted diagnostic. No candidate training or accuracy comparison informed the decision. |
| 2026-09-15 | Completed R3, froze 4B-D1, and handed the selector to Macro-section 3. | Exact weights, pretraining boundary, trainability, full-frame transform, independent head, weighted BCE, resource target, limitations, and required provenance are binding. The remaining campaign settings belong to Phase 3 P1; no model implementation or outcome inspection occurred. |
