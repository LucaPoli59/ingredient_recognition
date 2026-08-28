# C2 — Swin Transformer V2 candidate dossier

**Created:** 2026-08-28
**Last updated:** 2026-08-28
**Candidate:** C2 — Hierarchical/windowed transformer
**Status:** Research dossier; no local training performed

## Research question and boundary

Can a hierarchical transformer with shifted local windows preserve small
ingredient cues while still integrating dish-level context more effectively
than a pooled CNN? C2 is an image-only Swin Transformer V2 visual backbone
with a fixed 165-logit head. Its strongest evidence is general vision and
adjacent food vision; the direct ingredient evidence currently comes from a
separate Recipe1M study and must remain in that context.

Tiny and Small are nested size variants of one Swin V2 family. They do not fill
separate candidate slots.

## Candidate identity and mechanism

| Tuple field | Proposed C2 value |
| --- | --- |
| Family | Swin Transformer V2 |
| Representative | `swin_v2_t` or `swin_v2_s`, with the final size chosen only after the 4A.3 feasibility comparison |
| Pretraining | ImageNet-1K or ImageNet-22K supervised checkpoints; exact source remains a protocol variable |
| Representation | Four-stage hierarchical patch representation with shifted window attention, patch merging, and multi-scale features |
| V2 mechanism | Residual post-normalisation, scaled cosine attention, and log-spaced continuous relative-position bias; SimMIM is an available pretraining path |
| Adaptation | Replace the single-label head with a 165-output independent sigmoid head; frozen/partial/full tuning remains open |
| Input policy | A declared aspect-preserving square-compatible policy; the released weights use 256-square evaluation transforms |

Swin's shifted windows restrict attention to non-overlapping local regions while
shifting the partition between blocks to connect neighbouring windows. Swin V2
adds stability and resolution-transfer mechanisms rather than changing the
candidate into a generic ViT. This is distinct from C5 MaxViT's convolutional
plus block/grid attention.

## Feature flow and adaptation boundary

The TorchVision implementation applies patch embedding and hierarchical stages,
normalises the final feature map, permutes it into channel-first layout, pools
spatially, flattens, and applies a linear `head`. The T and S variants expose a
768-dimensional pooled representation; the stock 1,000-way head can therefore
be replaced with `Linear(768, 165)`.

The hierarchy is valuable for local evidence, but the canonical pooled head
still removes an explicit per-label spatial readout. Pairing Swin with C4 is a
separate protocol and must be analysed as a head change, not as evidence that
the Swin backbone alone localises ingredients.

## Pretraining and task-relevant evidence

| Source | Evidence type | Finding used here | Transfer boundary |
| --- | --- | --- | --- |
| [Swin Transformer V2, Liu et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html) | Primary architecture/pretraining evidence | V2 uses residual post-norm, scaled cosine attention, continuous relative-position bias, and SimMIM to scale capacity/resolution and transfer. | The paper's ImageNet/detection/segmentation results are not recipe-level ingredient recognition. |
| [Official Microsoft Swin repository](https://github.com/microsoft/Swin-Transformer) | Official implementation/checkpoints | ImageNet-1K/22K checkpoints, classification code, and downstream integrations provide a traceable path; the repository describes shifted-window efficiency. | Exact revision, checkpoint hash, transform, and dependency compatibility still require a Phase 5 manifest. |
| [FoodSeg103 / ReLeM](https://arxiv.org/abs/2105.05409) | Adjacent food-vision evidence | Ingredient-level food imagery and multi-scale localisation motivate preserving local and hierarchical structure. | Segmentation masks and 103 visible categories are not recipe-level labels or proof of hidden-ingredient inference. |
| [Visual Food Ingredient Prediction Using Deep Learning with Direct F-Score Optimization](https://www.mdpi.com/2304-8158/14/24/4269) | Direct food multi-label evidence | SwinV2-T is compared with EfficientNetV2-S, MaxViT-T, and ResNet-50 using an image-only ingredient head on Recipe1M. | Recipe1M, F-score optimisation, and its preprocessing differ from Yummly; source scores cannot rank C2 here. |

## Project fit and transfer limits

| Requirement | Assessment | Reason and limitation |
| --- | --- | --- |
| R1 fixed 165-label output | Strong | The pooled representation accepts a standard independent linear head. |
| R2 partial observability/local evidence | Moderate-high | Windowed hierarchy can preserve local evidence and global context, but the pooled head does not prove label-specific localisation. |
| R3 sparse positives/long tail | Moderate | Multi-label logits and multi-scale features are compatible; imbalance remains a loss and measurement problem. |
| R4 label co-occurrence/shortcuts | Moderate | Context can encode dish/cuisine priors; no explicit graph is introduced, so non-visual controls remain necessary. |
| R5 small 3:2 inputs | Uncertain | Official weights use 256×256 crop/260 resize; square partitioning and 3:2 aspect preservation need an explicit policy. |
| R6 provenance/leakage | Moderate | ImageNet pretraining is auditable, while any 22K or SimMIM route requires separate data/checkpoint provenance. |
| R7 ranking/calibration | Strong | The head exposes independent logits and can use the common AP/calibration instrumentation. |
| R8 reproducibility/8 GB | Moderate | TorchVision provides a maintained path, but T/S activation memory at useful resolution is unmeasured. |
| R9 fair comparison | Strong | Same split/head/metrics are straightforward once input policy is declared. |
| R10 one declared seed | Strong | The dossier does not require repeated-seed training. |
| R11 falsifiability | Strong | The testable mechanism is local-window plus hierarchical context, not a generic “transformer is newer” claim. |

## Canonical protocol and alternatives

The canonical C2 protocol is an ImageNet-pretrained Swin V2-T or S, the
original classification head replaced by 165 independent logits, and one
declared square-compatible transform that does not silently use recipe text or
metadata. Frozen linear probing, partial fine-tuning, and full fine-tuning are
adaptation alternatives for the later benchmark; 4A.2 does not choose one.

The SimMIM checkpoint path is a potentially useful alternative but should not
be mixed with ImageNet-supervised weights in the same family claim. A C4 query
head, a different patch/window size, or a new augmentation schedule is a
separate protocol axis and must be labelled as such.

## Access, checkpoint, licence, and provenance

The official Microsoft repository is under the [MIT
licence](https://github.com/microsoft/Swin-Transformer/blob/main/LICENSE) and
contains classification code and released checkpoints. TorchVision also
provides maintained `swin_v2_t/s` constructors and weights under the
[TorchVision BSD 3-Clause source licence](https://github.com/pytorch/vision/blob/main/LICENSE).
The Phase 5 manifest must record which implementation is used, because the
repository and TorchVision can differ in parameter naming, transforms, and
checkpoint format.

Inspection of the installed TorchVision 0.23.0 metadata reported:

| Variant | Parameters | Published operations | Weight transform |
| --- | ---: | ---: | --- |
| T | 28,351,570 | 5.940 GFLOPs | 256×256 crop, 260 resize, ImageNet mean/std, bicubic |
| S | 49,737,442 | 11.546 GFLOPs | 256×256 crop, 260 resize, ImageNet mean/std, bicubic |

These are library metadata at the released square input, not a local 8 GB
training result. A 3:2 resize-and-pad policy may change the feature-map token
count and memory; that effect must be measured rather than inferred.

## Repository integration path

Phase 5 would add a wrapper under `src/models/` that converts the TorchVision
model to the `BaseModel` contract, replaces `head`, preserves logits, exposes a
real final attention/block target for the dashboard, and serialises the exact
checkpoint and transform policy. The implementation should reuse the common
training and DataModule APIs, not the Microsoft repository's launcher.

Required smoke checks are offline checkpoint loading, output shape `(B, 165)`,
feature-map traversal, checkpoint reconstruction, transform/resize behaviour on
3:2 images, one forward/backward pass, and peak memory at the declared scale.

## Risks, go/no-go conditions, and open questions

**Go for 4A.3 comparison:** C2 has a distinct spatial mechanism, official and
maintained access paths, and direct recent food-ingredient comparison evidence.

**Go/no-go questions:**

- Does the square partition and 256-pixel checkpoint preserve more useful
  information than an aspect-preserving 224/256 policy on Yummly images?
- Can Tiny or Small execute with the selected head and batch policy within the
  8 GB GPU once activations, not just weights, are counted?
- Does a C2 gain persist against C1 and C5 when all use matched transforms?
- Do window features improve rare/local labels, or do they mainly amplify
  cuisine and presentation shortcuts?

**Invalidating evidence:** an unresolvable checkpoint/dependency path, an
unacceptable memory footprint at the smallest useful variant, or a transform
that destroys the local cues motivating C2 would remove it from the
implementation shortlist while preserving the evidence record.

## Falsifiable local hypothesis and minimal comparison

**Hypothesis H-C2:** at matched input area, head, loss, and training budget,
Swin V2-T will improve validation AP for labels requiring both local texture
and dish context relative to the ResNet anchor, without a corresponding gain
on image-shuffled or non-visual co-occurrence controls.

The minimal comparison is one declared-seed C2 run against the same-protocol
ResNet and, if both survive 4A.3, C1 or C5. Report macro/micro AP, per-label
support bands, calibration, and resource measurements. A result obtained only
with a different crop or a different pretrained-data regime is not an isolated
C2 architecture result.

## Comparison anchors

| Anchor | Role |
| --- | --- |
| [ResNet local contract](../../../implementation_details/models.md) | Existing supervised-CNN continuity control. |
| [DINOv2 local deep dive](../../../models_deepdive/dinov2.md) | Existing visual self-supervised anchor; not a new Swin alternative. |

## Handoff to 4A.3

Reuse the C2 tuple, V2 mechanism, implementation alternatives, resource facts,
and H-C2 hypothesis. 4A.3 must choose whether the windowed hierarchy is
complementary to C1/C5 and freeze the concrete input/adaptation protocol only
after the common qualitative comparison.

## References

- [Swin Transformer V2 paper](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html)
- [Swin Transformer V2 arXiv record](https://arxiv.org/abs/2111.09883)
- [Official Microsoft repository](https://github.com/microsoft/Swin-Transformer)
- [TorchVision Swin implementation and weight metadata](https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py)
- [FoodSeg103 / ReLeM](https://arxiv.org/abs/2105.05409)
- [Visual Food Ingredient Prediction Using Deep Learning with Direct F-Score Optimization](https://www.mdpi.com/2304-8158/14/24/4269)
