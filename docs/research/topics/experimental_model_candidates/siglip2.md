# C3 — SigLIP2 candidate dossier

**Created:** 2026-08-28
**Last updated:** 2026-08-28
**Candidate:** C3 — Vision-language visual encoder
**Status:** Research dossier; no local training performed

## Research question and boundary

Can a language-supervised visual representation provide useful semantic
features for recipe-level ingredient inference beyond image-only supervised or
self-supervised anchors? C3 is intentionally split into a canonical image-only
downstream protocol and optional text-conditioned interventions. The canonical
claim tests the visual encoder plus a learned 165-logit head; it does not use
ingredient names, recipe text, or prompts at inference.

SigLIP2 Base, native-aspect variants, and older SigLIP/CLIP checkpoints are
nested representation options. They do not create separate candidate slots
unless a later decision identifies a different scientific mechanism.

## Candidate identity and mechanism

| Tuple field | Proposed C3 value |
| --- | --- |
| Family | SigLIP2 vision-language encoder |
| Representative | `google/siglip2-base-patch16-224`; `google/siglip2-base-patch16-naflex` is a native-aspect alternative to audit |
| Pretraining | Image-text training on WebLI, extending SigLIP with captioning, self-distillation, masked prediction, and online curation |
| Representation | Vision Transformer image tower with patch tokens and pooled image features; text tower is not needed by the canonical downstream head |
| Adaptation | Image-only linear/multi-layer head over visual features; frozen, partial, and full tuning remain open |
| Head | Independent 165-output logits for the canonical comparison |
| Input policy | Use the checkpoint processor or a documented equivalent; fixed-square and native-aspect modes are separate variants |

The original SigLIP objective applies a pairwise sigmoid loss to image-text
pairs rather than a batch-wide softmax. SigLIP2 retains that image-text basis
and adds captioning, self-supervised losses, curation, and variants that handle
multiple resolutions and native aspect ratios. The mechanism may encode
fine-grained semantic concepts, but it also imports a web-language prior and
cannot by itself demonstrate that a hidden ingredient is visually observable.

## Feature flow and adaptation boundary

The official Transformers path loads an `AutoProcessor` and `AutoModel`, then
exposes image features through `get_image_features` or the vision tower output.
The canonical downstream adapter maps the image representation to 165 logits.
It must return raw logits so that the repository's Lightning module continues
to own loss, sigmoid, calibration, and metric policy.

Using `candidate_labels` and image-text logits is a valid semantic-prior
experiment, but it is not the canonical C3 result: it gives the model the
target names at inference and changes the question. Text-initialised queries,
prompt ensembling, and joint image-text fine-tuning therefore require separate
named protocols and provenance statements.

## Pretraining and task-relevant evidence

| Source | Evidence type | Finding used here | Transfer boundary |
| --- | --- | --- | --- |
| [Sigmoid Loss for Language Image Pre-Training, Zhai et al., ICCV 2023](https://arxiv.org/abs/2303.15343) | Primary objective evidence | Pairwise sigmoid image-text loss avoids global softmax normalisation and supports scalable image-text pretraining. | Generic zero-shot/transfer results do not establish recipe-ingredient set inference. |
| [SigLIP 2, Tschannen et al., 2025](https://arxiv.org/abs/2502.14786) | Primary representation evidence | The unified recipe adds captioning, self-distillation, masked prediction, curation, multilingual data, and dense/native-aspect variants; released sizes include ViT-B, L, So400m, and g. | More semantic and dense features may improve transfer, but the web-language prior and source overlap are part of the treatment, not free evidence. |
| [SigLIP2 Base model card](https://huggingface.co/google/siglip2-base-patch16-224) and [native-aspect model card](https://huggingface.co/google/siglip2-base-patch16-naflex) | Official checkpoint/interface evidence | Transformers processors and image-feature extraction are documented; the model cards list Apache-2.0 and WebLI pretraining. | Exact checkpoint revision, hashes, processor settings, and total-vs-vision parameter accounting require a Phase 5 manifest. |
| [Vision-Language Models for Image-Based Dietary Assessment benchmark](https://www.biorxiv.org/content/10.64898/2026.07.26.740845v1.full) | Adjacent recent food/VLM evidence | Nutrition5K ingredient-overlap evaluation illustrates both the promise and the ambiguity of language-capable models on food images. | Preprint, Nutrition5K labels, prompt policy, and overlap differ from the image-only Yummly benchmark. |
| [Visual Food Ingredient Prediction Using Deep Learning with Direct F-Score Optimization](https://www.mdpi.com/2304-8158/14/24/4269) | Direct food multi-label control evidence | A simple image-only head over modern visual encoders remains a meaningful baseline for food ingredients. | It does not evaluate SigLIP2 and uses Recipe1M/F-score optimisation; it supports the need for a clean image-only control. |

## Project fit and transfer limits

| Requirement | Assessment | Reason and limitation |
| --- | --- | --- |
| R1 fixed 165-label output | Strong for image-only adapter | A learned linear head can produce the required logits without text at inference. |
| R2 partial observability/local evidence | Moderate | Semantic pretraining may capture dish context and patch features; it cannot turn unobservable preparation steps into visual truth. |
| R3 sparse positives/long tail | Moderate | Pretrained features may help rare labels, but the head/loss still face severe imbalance. |
| R4 label co-occurrence/shortcuts | Uncertain | Language and web data can encode strong co-occurrence and cuisine priors; non-visual controls and provenance audit are mandatory. |
| R5 small 3:2 inputs | Moderate-high in native-aspect variant, uncertain in fixed-square variant | SigLIP2 explicitly releases native-aspect/resolution variants, but the exact token budget and useful resolution require measurement. |
| R6 provenance/leakage | Weak-moderate | WebLI scale and content are not fully auditable against Yummly; source overlap and label-name priors must be disclosed. |
| R7 ranking/calibration | Strong for learned head; separate for text scoring | Image-only logits support common metrics; image-text logits need their own calibration and cannot be merged silently. |
| R8 reproducibility/8 GB | Uncertain | The model card lists 0.4B parameters for Base and F32 tensors; full fine-tuning may exceed 8 GB even if a frozen/adapter path fits. |
| R9 fair comparison | Moderate | A clean image-only adapter is comparable; prompt/text variants would confound the category comparison. |
| R10 one declared seed | Strong | No repeated-seed evidence is needed for intake. |
| R11 falsifiability | Strong | The hypothesis is semantic visual transfer versus provenance/shortcut risk, not an unconditional claim of superiority. |

## Canonical protocol and alternatives

The recommended canonical C3 protocol is:

1. pin one SigLIP2 Base checkpoint and processor;
2. use only the vision tower/image features;
3. attach an independently initialised 165-logit head;
4. keep all target names, recipes, and label graphs out of inference; and
5. compare the image-only protocol under the common split and metrics.

Frozen visual features with a linear head, partial fine-tuning, and full
fine-tuning are adaptation alternatives. `naflex` native-aspect processing,
fixed 224-square processing, prompt-based zero-shot scoring, and text-query
initialisation are separate variants. The 4A.3 decision must state which one is
being compared; otherwise C3 combines pretraining, input, and label semantics
into an uninterpretable treatment.

## Access, checkpoint, licence, and provenance

The maintained access path is Hugging Face Transformers with the Google
`siglip2-*` model cards and Safetensors checkpoints. The Base cards currently
declare the [Apache-2.0 licence](https://huggingface.co/google/siglip2-base-patch16-224).
The paper describes WebLI pretraining and large TPU training; neither implies
that the data or near-duplicate image overlap with Yummly is fully auditable.

The paper labels the visual family sizes as ViT-B (86M), L (303M), So400m
(400M), and g (1B), while the current Base model card reports a 0.4B total
checkpoint. This apparent accounting difference is deliberately unresolved:
Phase 5 must record the exact configuration's total and vision-tower parameter
counts instead of mixing paper and model-card numbers.

The Transformers documentation provides both fixed-square and native-aspect
examples, including a `max_num_patches` control for native-aspect inputs. Those
settings affect activation memory and effective resolution and must be frozen
with the checkpoint hash for reproducibility.

## Resource envelope

| Variant/protocol | Known metadata | Interpretation |
| --- | --- | --- |
| Base patch16-224 | Model card: 0.4B parameters, F32; 224-style processor | Weight storage is substantial; training memory cannot be inferred from the parameter count. |
| Base patch16-naflex | Same Base family; processor exposes `max_num_patches` | Native aspect can avoid a forced crop but token count is an explicit memory/compute variable. |
| L/So400m/g | Paper family sizes 303M/400M/1B | Research references only; not plausible 8 GB starting points without freezing/quantisation and a separate justification. |

There is no local peak-memory or throughput measurement in this dossier. A
frozen image tower plus a small head may be a feasible probe, while full
fine-tuning is a high-risk path on the 8 GB development GPU. Quantisation or
parameter-efficient adapters would change the adaptation hypothesis and must
be recorded separately.

## Repository integration path

Phase 5 would add a wrapper under `src/models/` that loads a pinned local
Transformers checkpoint, exposes a model-owned transform/processor compatible
with the DataModule, maps image features to 165 logits, and serialises the
checkpoint and processor revisions. It must not call the text tower or access
ingredient names in the canonical image-only protocol. A dashboard target
should point to a real vision block or patch representation if interpretability
is claimed.

Required smoke checks are offline cache loading, processor determinism on the
project's 3:2 images, output shape `(B, 165)`, checkpoint reconstruction,
feature-target traversal, one forward/backward pass for the declared tuning
mode, and peak-memory measurement. Provenance and overlap checks must be
written before using any food-domain or text-conditioned checkpoint.

## Risks, go/no-go conditions, and open questions

**Go for 4A.3 comparison:** C3 tests a scientifically distinct semantic visual
prior and has a maintained checkpoint/interface path with an explicit Apache-2.0
model card.

**Go/no-go questions:**

- Can a frozen or parameter-efficient Base protocol fit the 8 GB boundary and
  still expose useful image features?
- What exact WebLI/checkpoint provenance and possible Yummly overlap can be
  documented?
- Does an image-only head gain on validation AP remain after comparing against
  DINOv2 and matched non-visual controls?
- If native-aspect processing is used, what patch budget is fair against the
  square-input families?

**Invalidating evidence:** inability to cache and reload a checkpoint under
  acceptable terms, unresolvable provenance that prevents a defensible claim,
  or a memory path requiring hidden text/prompt inputs would demote C3 from the
  common image-only shortlist. The evidence remains useful for a clearly
  labelled semantic-prior ablation.

## Falsifiable local hypothesis and minimal comparison

**Hypothesis H-C3:** an image-only SigLIP2 visual adapter will improve
validation AP on semantically distinctive and contextual ingredient groups over
the existing DINOv2/ResNet anchors, but any gain will shrink when the protocol
uses no label text and is evaluated with provenance/shortcut controls.

The minimal test is a frozen-vision or declared fine-tuning C3 run with the
same 165-label head contract, paired with DINOv2 and ResNet controls. Report
macro/micro AP, per-label support/observability slices, calibration, and
resource cost. Prompted text scoring may be reported only as a separate
semantic-prior experiment, never as the canonical C3 comparison.

## Comparison anchors

| Anchor | Role |
| --- | --- |
| [ResNet local contract](../../../implementation_details/models.md) | Existing image-only supervised control. |
| [DINOv2 local deep dive](../../../models_deepdive/dinov2.md) | Existing self-supervised visual control; C3's semantic prior is not equivalent to DINOv2. |

## Handoff to 4A.3

Reuse the C3 image-only/text-conditioned boundary, exact checkpoint accounting
issue, provenance risks, and H-C3 hypothesis. 4A.3 must choose whether the
semantic prior offers enough information value to justify its access and
overlap burden; it must not treat text-conditioned scores as image-only model
evidence.

## References

- [SigLIP paper](https://arxiv.org/abs/2303.15343)
- [SigLIP2 paper](https://arxiv.org/abs/2502.14786)
- [SigLIP2 Base model card](https://huggingface.co/google/siglip2-base-patch16-224)
- [SigLIP2 native-aspect model card](https://huggingface.co/google/siglip2-base-patch16-naflex)
- [Transformers SigLIP2 documentation](https://huggingface.co/docs/transformers/model_doc/siglip2)
- [Food VLM dietary-assessment benchmark](https://www.biorxiv.org/content/10.64898/2026.07.26.740845v1.full)
- [Visual Food Ingredient Prediction Using Deep Learning with Direct F-Score Optimization](https://www.mdpi.com/2304-8158/14/24/4269)
