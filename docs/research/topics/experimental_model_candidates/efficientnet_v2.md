# C1 — EfficientNetV2 candidate dossier

**Created:** 2026-08-28
**Last updated:** 2026-08-28
**Candidate:** C1 — Efficient convolutional network
**Status:** Research dossier; no local training performed

## Research question and boundary

Can the training-aware efficiency and compound scaling of EfficientNetV2 give
the project a stronger parameter/compute trade-off than the existing ResNet
anchor for sparse, recipe-level multi-label prediction? The candidate is an
image-only visual encoder with a learned 165-logit head. The original evidence
is primarily ImageNet and other single-label transfer tasks; the food evidence
is direct in task formulation but uses a different dataset and vocabulary.

This dossier evaluates a family-level hypothesis. EfficientNetV2-S, -M, and
-L remain nested variants rather than separate candidates.

## Candidate identity and mechanism

| Tuple field | Proposed C1 value |
| --- | --- |
| Family | EfficientNetV2 |
| Representative | `efficientnet_v2_s` with ImageNet weights when the checkpoint path passes the Phase 5 audit |
| Pretraining | ImageNet-1K/21K supervised weights; exact checkpoint is a protocol variable |
| Representation | Convolutional stages using Fused-MBConv in early stages and MBConv with squeeze-and-excitation later |
| Adaptation | Replace the single-label classifier with a fixed-vocabulary multi-label head; frozen, partial, or full fine-tuning remains open |
| Head | Independent sigmoid logits; no sigmoid inside the model wrapper |
| Input policy | Aspect-preserving policy to be defined against the square ImageNet weight transform; no silent crop/warp is adopted here |

EfficientNetV2 was obtained with training-aware neural architecture search and
scaling. Its distinguishing mechanism is the joint optimisation of accuracy,
training speed, and parameter size, with Fused-MBConv and progressive learning
that adjusts regularisation as image size changes. These are different from a
plain residual CNN's optimisation and block composition, so C1 is a genuinely
new candidate rather than another ResNet depth.

## Feature flow and adaptation boundary

The TorchVision implementation exposes a convolutional `features` sequence,
adaptive global average pooling, and a classifier. For the current S variant,
the final pooled representation is 1,280 channels; the stock 1,000-way linear
layer is therefore replaced by `Linear(1280, 165)` for the canonical protocol.
The training module receives logits and applies `BCEWithLogitsLoss`, sigmoid,
metrics, and calibration according to the project contract.

The backbone can preserve local texture through its convolutional stages, but
the default pooled head discards explicit spatial coordinates. A query/set head
from C4 could be paired with its feature map as a separate head hypothesis; it
must not be folded into the C1 backbone claim.

## Pretraining and task-relevant evidence

| Source | Evidence type | Finding used here | Transfer boundary |
| --- | --- | --- | --- |
| [EfficientNetV2, Tan and Le, ICML 2021](https://proceedings.mlr.press/v139/tan21a.html) | Primary architecture/training evidence | Fused-MBConv, training-aware search/scaling, and progressive learning target efficiency and faster training. | The reported ImageNet/CIFAR/Cars/Flowers results are not recipe-level multi-label evidence. Progressive learning must not be imported without an explicit input/augmentation protocol. |
| [TorchVision EfficientNetV2 documentation](https://docs.pytorch.org/vision/0.27/models/efficientnetv2.html) and [weight metadata](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py) | Official maintained implementation | Constructors and weight enums for S/M/L provide a traceable PyTorch path. | The repository's installed version, checkpoint hash, transform, and licence notices must be frozen before training. |
| [Food Ingredients Recognition through Multi-label Learning, Ismail and Yuan](https://arxiv.org/abs/2210.14147) | Direct food multi-label evidence | A food ingredient study evaluates EfficientNet-family encoders with global-pooling and attention decoders on Nutrition5K. | EfficientNet is not EfficientNetV2, and Nutrition5K labels/splits differ from Yummly; this establishes task relevance, not expected score. |
| [Visual Food Ingredient Prediction Using Deep Learning with Direct F-Score Optimization](https://www.mdpi.com/2304-8158/14/24/4269) | Direct food multi-label evidence | A 2025 Recipe1M study compares EfficientNetV2-S with MaxViT-T, SwinV2-T, and ResNet-50 under an image-only ingredient head. | The source uses Recipe1M, a different vocabulary and F-score optimisation; the reported comparison cannot rank C1 on Yummly. |

## Project fit and transfer limits

| Requirement | Assessment | Reason and limitation |
| --- | --- | --- |
| R1 fixed 165-label output | Strong | A linear multi-label head is direct and produces comparable logits. |
| R2 partial observability/local evidence | Moderate | Convolutional locality is useful, but global pooling does not expose class-specific evidence by itself. |
| R3 sparse positives/long tail | Moderate | Efficient capacity and per-label logits are compatible; the architecture does not solve imbalance without the declared loss/analysis policy. |
| R4 label co-occurrence/shortcuts | Moderate | Shared features can learn co-occurrence, but C1 has no explicit label-dependency module and requires the project's non-visual controls. |
| R5 small 3:2 inputs | Uncertain | The released S weights use a 384-square crop; aspect-preserving resizing/padding and useful resolution remain to test. |
| R6 provenance/leakage | Moderate | ImageNet provenance is easier to document than food-domain pretraining, but exact checkpoint and overlap audits remain required. |
| R7 ranking/calibration | Strong | Independent logits allow AP, calibrated probabilities, fixed-policy F1, and per-label trajectories. |
| R8 reproducibility/8 GB | Moderate-high | TorchVision access and the compact S variant are practical; peak training memory is not inferred from parameter count. |
| R9 fair comparison | Strong | It can use the common split, head, transforms, and metrics with the existing anchors. |
| R10 one declared seed | Strong | No repeated-seed evidence is required at dossier stage. |
| R11 falsifiability | Strong | Efficiency and local convolutional evidence yield a clear comparison against ResNet. |

The strongest C1 transfer claim is therefore efficiency under a fixed image-only
protocol, not improved visual observability of hidden ingredients.

## Canonical protocol and alternatives

The canonical candidate for 4A.3 is:

1. load one declared TorchVision EfficientNetV2 checkpoint;
2. remove its ImageNet classifier and attach a 165-output independent linear
   head;
3. use the same target metadata, split, loss family, metric instrumentation,
   and selection boundary as every other candidate; and
4. declare one aspect-preserving input transform and record its relation to the
   checkpoint's square transform.

Frozen-linear-probe, partial fine-tuning, and full fine-tuning are legitimate
   adaptation alternatives, but 4A.2 does not choose among them. The original
   progressive-learning augmentation schedule is not part of the canonical
   protocol unless a later plan adopts it explicitly, because it would change
   the comparison's augmentation axis.

## Access, checkpoint, licence, and provenance

The maintained path is TorchVision's `efficientnet_v2_s/m/l` constructors and
their ImageNet weight enums. TorchVision source is distributed under the
[BSD 3-Clause licence](https://github.com/pytorch/vision/blob/main/LICENSE).
The checkpoint URL, package version, SHA-256, and any upstream weight-specific
terms must be recorded in the Phase 5 manifest; the paper alone is not a
checkpoint provenance record. The supervised ImageNet pretraining corpus is an
external prior, but does not contain Yummly recipe labels as an explicit
downstream input.

The current WSL TorchVision inspection (version 0.23.0) reported the following
metadata for the available weights:

| Variant | Parameters | Published operations | Weight transform |
| --- | ---: | ---: | --- |
| S | 21,458,488 | 8.366 GFLOPs | 384×384 crop/resize, ImageNet mean/std, bilinear |
| M | 54,139,356 | 24.582 GFLOPs | 480×480 crop/resize, ImageNet mean/std, bilinear |
| L | 118,515,272 | 56.08 GFLOPs | 480×480 crop/resize, mean/std 0.5, bicubic |

These are library metadata, not measured Yummly training memory. The original
paper and a 2025 food study report slightly different parameter counts for S;
the exact local constructor and checkpoint must be the implementation source
of truth rather than mixing those tables.

## Repository integration path

Phase 5 would add a maintained wrapper under `src/models/` implementing the
existing `BaseModel` contract: `num_classes`, model-owned transforms,
`conv_target_layer`, `classifier_target_layer`, serialisable configuration, and
one logit per target. The wrapper should reuse the TorchVision constructor and
avoid copying the training pipeline into a launcher. The current
`implementation_details/models.md` and model index would be updated only after
the wrapper and smoke tests exist.

Required Phase 5 checks are weight loading offline, output shape `(B, 165)`,
checkpoint reconstruction, Grad-CAM target traversal, transform provenance,
one forward/backward pass, and peak-memory measurement on the 8 GB GPU.

## Risks, go/no-go conditions, and open questions

**Go for 4A.3 comparison:** the family has direct food multi-label precedent,
maintained access, a compact representative, and a mechanism distinct from
ResNet.

**Go/no-go questions:**

- Can a 3:2 image policy preserve enough local evidence without violating the
  ImageNet checkpoint assumptions?
- Does the S variant remain useful when the project input is not the released
  384-square crop?
- Does efficiency matter for the sparse-label optimisation bottleneck, or only
  for throughput?
- If C4 is paired with this backbone, are observed gains attributable to C1 or
  to the query head?

**Invalidating evidence:** failure to load a reproducible checkpoint under
  acceptable terms, an unresolvable transform contract, or a measured memory
  path that cannot execute the declared representative would remove C1 from
  the implementation shortlist, but not erase this research record.

## Falsifiable local hypothesis and minimal comparison

**Hypothesis H-C1:** at a matched image policy, loss, target vocabulary, and
training budget, EfficientNetV2-S will improve validation average precision per
unit of measured compute/memory over the existing ResNet anchor, with the
largest benefit on labels requiring mid-level texture/colour cues rather than
pure frequency priors.

The minimal test is one declared-seed C1 run paired with the same-protocol
ResNet control. Compare macro/micro AP, per-label AP by support band, calibration,
and measured resources. A gain that appears only after a different crop,
augmentation schedule, or class-weight policy is not evidence for the C1
architecture alone.

## Comparison anchors

| Anchor | Role |
| --- | --- |
| [ResNet local contract](../../../implementation_details/models.md) | Existing supervised-CNN continuity and simple head control. |
| [DINOv2 local deep dive](../../../models_deepdive/dinov2.md) | Existing visual self-supervised representation anchor; not a new C1 alternative. |

## Handoff to 4A.3

Reuse the C1 tuple, evidence boundaries, S/M/L resource metadata, and H-C1
hypothesis. 4A.3 must choose the concrete adaptation and transform policy only
after comparing all dossiers. It must not infer Yummly performance from
ImageNet or Recipe1M scores.

## References

- [EfficientNetV2 paper](https://proceedings.mlr.press/v139/tan21a.html)
- [TorchVision model documentation](https://docs.pytorch.org/vision/0.27/models/efficientnetv2.html)
- [TorchVision source and weight metadata](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)
- [Food Ingredients Recognition through Multi-label Learning](https://arxiv.org/abs/2210.14147)
- [Visual Food Ingredient Prediction Using Deep Learning with Direct F-Score Optimization](https://www.mdpi.com/2304-8158/14/24/4269)
- [TorchVision licence](https://github.com/pytorch/vision/blob/main/LICENSE)
