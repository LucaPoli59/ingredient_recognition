# Experimental model portfolio

**Created:** 2026-09-07
**Last updated:** 2026-09-07
**Status:** Active and binding family selection; implementation unverified
**Decision:** 4A-D1 — established families

## Decision and scope

Select **EfficientNetV2** and **MaxViT**, starting with **EfficientNetV2-S**
and **MaxViT-T**, as the two literature-derived experiment families. They test
efficient convolutional representation and hybrid local/global attention,
respectively. Their expected value is a research hypothesis, not a prediction
of which model will win on Yummly.

ResNet and DINOv2 remain the already-used comparison anchors. The third new
category, a custom attention architecture, remains to be designed and selected
in 4A.4. Swin V2 is the first alternative to reconsider if MaxViT fails its
implementation gate; it is not a third selected family or an automatic swap.
This decision does not choose the reference selector `M_ref` or the ingredient
subset, which retain their separate 4B and Phase 3 owners.

This document owns portfolio decisions. The
[benchmark](benchmark_decisions.md) and
[comparative methodology](model_comparison_methodology.md) still govern data,
metrics, HPO, vocabulary controls, and final evaluation. Detailed literature
evidence remains in the [candidate collection](../research/topics/experimental_model_candidates/README.md).

## Evidence and decision method

The decision uses the five dossiers and their
[comparative synthesis](../research/topics/experimental_model_candidates/comparative_synthesis.md),
with literature cutoff 2026-08-28 and a bounded source/implementation review on
2026-09-07. The repository baseline is commit `e918393`. Inputs re-read were
the [data audit](yummly_data_audit.md) and [problem definition](problem_definition.md)
(2026-08-02), [vocabulary audit](ingredient_vocabulary_audit.md) (2026-08-04),
benchmark decisions (2026-08-27), comparative methodology (2026-08-28),
[model contract](../implementation_details/models.md) (2026-08-27), and
[requirements matrix](../research/discovery/2026-08-28/problem_model_requirements.md)
(2026-08-28). Older audit counts are historical; the binding task is the
165-label `v5` generation, without `<UNK>`.

Eligibility is checked before portfolio value. A research-level pass requires
a concrete, plausible access and execution path; it does not certify a passed
GPU or checkpoint test. The priority is problem fit and evidence, then
complementarity and feasible implementation under the thesis budget. Recency
does not settle this choice. No local candidate run, HPO, or test outcome was
used.

### Eligibility outcome

All five have a documented image-only route to independent logits and a
falsifiable comparison. Remaining conditions and dispositions are explicit:

| Candidate and evidence | Relevance and evidence | Access, provenance, and resource gate | Outcome |
| --- | --- | --- | --- |
| [C1 EfficientNetV2](../research/topics/experimental_model_candidates/efficientnet_v2.md) | Efficiency mechanism; direct ingredient-task precedent, with dataset-transfer limits | Traceable TorchVision ImageNet route; plausible compact fine-tuning path; transform and measured memory pending | Eligible; selected |
| [C2 Swin V2](../research/topics/experimental_model_candidates/swin_v2.md) | Hierarchical attention; direct ingredient-task precedent | Maintained TorchVision route; Tiny is plausible; exact transform and memory pending | Eligible; reserve for the spatial-backbone slot |
| [C3 SigLIP2](../research/topics/experimental_model_candidates/siglip2.md) | Distinct language-supervised visual prior; direct evidence for this exact ingredient protocol is weaker | Public image-tower path; acceptable scope of WebLI overlap uncertainty and concrete adaptation mode remain unresolved | Conditional; not selected for this portfolio |
| [C4 query/set head](../research/topics/experimental_model_candidates/structured_multilabel_head.md) | Strong multi-label mechanism and food attention-readout precedent | Reproducible initialization/reference code; feasibility and provenance depend on a named backbone and paired pooled-head control | Eligible as a paired protocol, not an independent backbone; retained for 4A.4 |
| [C5 MaxViT](../research/topics/experimental_model_candidates/maxvit.md) | Hybrid local/global mechanism; direct ingredient-task precedent | Maintained TorchVision ImageNet route avoids reference-code conversion; 224-square path plausible; memory pending | Eligible; selected |

Source licences and public weight routes support research intake. Phase 5 must
preserve applicable notices and pin exact artifacts. ImageNet provenance is
more bounded than WebLI, but it is not proof of zero overlap with web food
images. None of the selected models solves recipe-label ambiguity or removes
the need for shortcut diagnostics.

### Qualitative portfolio comparison

Each judgment below is a project interpretation of the linked dossier, relative
to the current two-family-plus-custom design. `Strong`, `moderate`, `weak`, and
`uncertain` are qualitative assessments, not measured scores or a weighted sum.

| Criterion | [C1](../research/topics/experimental_model_candidates/efficientnet_v2.md) | [C2](../research/topics/experimental_model_candidates/swin_v2.md) | [C3](../research/topics/experimental_model_candidates/siglip2.md) | [C4](../research/topics/experimental_model_candidates/structured_multilabel_head.md) | [C5](../research/topics/experimental_model_candidates/maxvit.md) |
| --- | --- | --- | --- | --- | --- |
| Fit to local cues, context, and weak supervision | Moderate | Strong | Moderate | Strong | Strong |
| Complementarity in the proposed portfolio | Strong: efficient CNN | Moderate: overlaps C5 | Strong: semantic pretraining | Moderate: overlaps custom-design opportunity | Strong: hybrid attention |
| Task relevance and empirical evidence | Strong | Moderate | Weak for exact protocol | Moderate | Moderate |
| Information value of minimal comparison | Strong | Strong | Strong | Strong if paired | Strong |
| Implementation/training plausibility | Strong | Moderate | Uncertain | Moderate | Moderate |
| Reproducibility and maintenance | Strong | Strong | Moderate | Moderate | Strong via TorchVision |
| Resource efficiency evidence | Strong | Moderate | Uncertain | Uncertain until paired | Moderate |
| Useful architectural recency | Moderate | Moderate | Strong | Moderate | Moderate |

These judgments do not imply that C1 has proven superior food-task accuracy or
that C5 has measured lower memory than C2. The direct Recipe1M comparison is
one study, not multiple independent replications for each exact family.

## Why this pair and why not the alternatives

**EfficientNetV2** adds an efficiency question beyond the existing ResNet
anchor. Its block design and training-aware scaling are substantively different
from changing ResNet depth. It supplies a practical convolutional reference
against which attention complexity must earn its cost. The selection does not
automatically adopt the paper's progressive-learning schedule.

**MaxViT** adds explicit local block and global grid interactions within a
convolutional hierarchy. This is useful both as an established spatial model
and as a reference for the later custom design. The combination with C1 asks
whether richer spatial interaction improves recipe-label ranking enough to
justify its measured cost.

**Swin V2** is a credible close alternative. MaxViT is preferred because its
hybrid structure directly connects the efficient CNN reference to the custom
attention research, and it has a compact released 224-square path. This is a
portfolio judgment, not evidence that Swin is inferior. Both would consume
two slots testing overlapping spatial-backbone questions. The Recipe1M study
supports MaxViT's candidacy but does not decide the choice by an imported score.

**SigLIP2** would provide a distinct pretraining question. It is deferred
because the exact image-only ingredient protocol has less direct evidence and
adds provenance/adaptation work to a portfolio that already has a DINOv2 visual
pretraining anchor. DINOv2 does not duplicate language supervision. The Base
checkpoint's total parameter display is not the vision-tower count and is not
used to assert that image-only fine-tuning cannot fit 8 GB.

**The structured head** is valuable evidence for 4A.4, particularly ML-Decoder
and Query2Label. Allocating it a family slot now would commit a backbone/head
pair and a control before the custom design is understood. Retain it as a
potential reusable component and, if relevant to the chosen custom hypothesis,
an established head control. A custom model that merely reproduces such a
head must be described as an adaptation, not an unsupported novelty claim.

## Selected protocols and fallback

The following is the adopted starting contract for Phase 5. Engineering checks
must pass before it becomes a runnable benchmark configuration.

| Field | EfficientNetV2-S | MaxViT-T |
| --- | --- | --- |
| Maintained route | TorchVision `efficientnet_v2_s` | TorchVision `maxvit_t` |
| Explicit weight identity | `EfficientNet_V2_S_Weights.IMAGENET1K_V1` | `MaxVit_T_Weights.IMAGENET1K_V1` |
| Representation | Final convolutional feature map | Final feature map after stem and MaxViT blocks |
| Common readout | Global average pooling, flatten, newly initialized linear projection to `L` logits | Same readout |
| Initial adaptation | Full backbone fine-tuning plus learned head | Same adaptation |
| Within-family resource fallback | Same S checkpoint and head, frozen encoder/linear probe | Same T checkpoint and head, frozen encoder/linear probe |
| Initial input | Shared aspect-preserving resize and centered padding to 224×224 | Same input |

`L=165` for the full task. Later vocabulary projections use their saved class
order; the model must not hard-code label names. The common readout intentionally
replaces MaxViT's complete stock classifier, including its intermediate
normalization/projection/Tanh, and EfficientNet's classifier dropout. It uses
the same linear initialization policy for both. Backbone-internal operations
remain family-specific. This explicitly resolves the C5 dossier's earlier
default suggestion to retain its pooled projection.

The shared input policy preserves the entire image geometry: resize to fit the
canvas, then center-pad without stretching or center-cropping away content.
Use common interpolation/antialiasing and ImageNet normalization; Phase 5 must
serialize exact rounding, pad fill, and transform ordering, and Phase 6 must
freeze a common augmentation policy. A typical 3:2 image occupies roughly
224×149 pixels of this canvas. Padding preserves framing; the reduced content
resolution loses spatial detail and the policy shifts the pretrained input
distribution. In particular,
224 is below the released EfficientNetV2-S evaluation transform of 384.

This is a bounded common-input experiment, not each checkpoint at its native
optimum. Changes to resolution or readout require an explicit protocol revision
before comparative training; they must not emerge as silent family-specific
defaults. No source FLOP value at a different resolution ranks the new models.

The frozen fallback reduces training state while retaining the same family,
weights, and head. It is an alternative if measured full tuning is infeasible,
not an extra required campaign. Freeze stochastic layers and normalization state
with the encoder. Before HPO, record whether the comparison is full tuning or
frozen representation; if only one family requires freezing, either use the
common frozen mode for the pair or revise the claim explicitly to a comparison
of different adaptation protocols. Do not call that an isolated architecture
effect. Neither fallback has yet been measured locally.

## Falsifiable comparisons and interpretation

| Hypothesis | Minimal later comparison | Result that weakens or fails to support it |
| --- | --- | --- |
| H-E: C1 offers a useful AP/resource trade-off beyond ResNet | C1 versus the current-benchmark ResNet anchor, common input/head/loss policy and declared comparable HPO budgets | No AP gain and no useful reduction in measured time/memory, or a gain explained by an unmatched policy |
| H-M: C5's spatial representation adds useful evidence beyond C1 | C5 versus C1 on the same `v5` task; examine macro AP, micro F1, per-label support/cuisine slices and cost | No ranking benefit or cost incompatible with the declared budget; gains explained only by priors leave the spatial-mechanism claim unsupported |

H-M specializes the C5 dossier's C2-or-C1 comparison to the actually selected
pair; it does not require training the reserve Swin model. Predefine any
observability groups independently of model outcomes; if annotations are not
ready, leave that mechanism conclusion open. Frequency/cuisine controls and
image-dependence diagnostics help assess shortcuts but do not prove direct
ingredient visibility. Evaluation-time image shuffling can be a diagnostic
without another trained model; define its pairing and interpretation in the
evaluation plan. Contextual prediction can still be useful for recipe inference;
it must be reported as such rather than as direct visual recognition.

The shared data, head, and downstream policy reduce confounding. Different
checkpoint training recipes, backbone regularizers, and later family-specific
HPO still mean the headline comparison concerns **model protocols**, not a
causal isolation of attention alone. One seed per configuration remains binding.
Finite-sample uncertainty does not measure stochastic training variability.
Vocabulary ablation, matched random subsets, and local adaptation retain the
separate Q1–Q4 rules in the comparative methodology.

## Handoff to custom research and implementation

| Recipient | Reusable evidence or gap | Required next outcome |
| --- | --- | --- |
| 4A.4 problem/evidence synthesis | Both chosen models ultimately pool away spatial locations; weak labels and cuisine priors remain unresolved | A bounded custom objective that names what changes and what stays controlled |
| 4A.4 component research | C1 efficient convolution/SE; C5 local/grid attention; C2 shifted windows; C4 query/group decoding; C3 pretraining/geometry lessons | Source-backed component choices and incompatibilities; no obligation to combine all of them |
| 4A.4 design comparison | Class-specific readout or multi-scale evidence may be useful, but small images and co-occurrence limit interpretation | Three distinct topology proposals with S/M/L scaling, then one selected topology and necessary ablations |
| Phase 5 artifact gate | Explicit weight enums and maintained constructors | Pinned package/source version, exact weight URL/hash, licence notices, offline load and checkpoint round trip |
| Phase 5 interface gate | Common logits/readout/input contract | Shape and gradient checks, preprocessing verification on representative train images, saved normalization state, working visualization hooks |
| Phase 5 resource gate | 8 GB, initial full tuning, bounded frozen fallback | Peak allocated/reserved memory and step time including loss, optimizer state and backward; feasible physical batch, precision and normalization policy |
| Phase 6 protocol freeze | Common full-task comparison and existing Q1–Q4 methodology | HPO spaces/budgets, augmentation/loss/stopping policies, one declared seed, calibration and transferred-vocabulary runs |

Gradient accumulation may help memory but does not reproduce the BatchNorm
statistics of a larger physical batch. The MaxViT checkpoint's documented
BatchNorm training-momentum peculiarity must be retained in its provenance and
the chosen fine-tuning normalization policy tested. No automatic change to its
saved running statistics is implied by this decision.

If either family has no useful executable route after its fallback, reopen
4A-D1 with the failed gate and its evidence. Reconsider Swin first for MaxViT's
spatial slot; any replacement remains a recorded two-family decision. Phase 5
engineering work may begin for these families when its DataModule prerequisite
passes; the custom-model decision and overall 4A completion remain pending.

## Source verification and limitations

The bounded 2026-09-07 review rechecked the
[EfficientNetV2 paper](https://proceedings.mlr.press/v139/tan21a.html),
[MaxViT paper](https://arxiv.org/abs/2204.01697), and versioned TorchVision
[EfficientNet source](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/efficientnet.py),
[MaxViT source](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/maxvit.py),
[Swin source](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/swin_transformer.py),
and [licence](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE).
Local source inspection confirmed the square `BaseModel` interface and the two
selected classifiers. The dated findings are retained in the
[research addendum](../research/topics/experimental_model_candidates/comparative_synthesis.md#bounded-source-review--2026-09-07).

The [Recipe1M study](https://doi.org/10.3390/foods14244269) remains direct
task precedent with a different vocabulary, objective, and preprocessing.
Its indexed primary text was available during this review; direct full-text
retrieval was rate-limited. The review does not add new quantitative claims from
it. Existing papers and rejected candidates remain retained research evidence.

No weights were downloaded or models run. Hashes, resource measurements,
training stability, and Yummly effectiveness remain unverified. Completing
4A.3 means that the portfolio decision and handoff are explicit, not that
4A.4, Phase 5, or the benchmark have been completed.

## Decision history

| Date | Decision | Basis |
| --- | --- | --- |
| 2026-09-07 | Adopted 4A-D1: EfficientNetV2-S and MaxViT-T, shared readout/input starting contract, explicit within-family fallback; retained Swin as first spatial reserve | Completed five-candidate research, current problem constraints, bounded official-source inspection, and portfolio complementarity |
