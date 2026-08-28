# C5 — MaxViT candidate dossier

**Created:** 2026-08-28
**Last updated:** 2026-08-28
**Candidate:** C5 — Hybrid multi-axis attention network
**Status:** Research dossier; no local training performed

## Research question and boundary

Can a hybrid convolutional and multi-axis attention backbone combine local
ingredient fragments with global dish context more efficiently than a purely
windowed transformer? C5 is an image-only MaxViT backbone with a standard
165-logit head. The family is distinct from C2 because its mechanism alternates
convolutional blocks with blocked local and dilated/global grid attention,
rather than shifted windows alone.

MaxViT-T is the compact representative; MaxViT-S and larger variants remain
nested options. The official research repository is archived, so the maintained
TorchVision implementation is the preferred access path subject to Phase 5
verification.

## Candidate identity and mechanism

| Tuple field | Proposed C5 value |
| --- | --- |
| Family | MaxViT (Multi-Axis Vision Transformer) |
| Representative | TorchVision `maxvit_t` with ImageNet weights |
| Pretraining | ImageNet-1K/21K checkpoints in the original work; exact maintained checkpoint is a protocol variable |
| Representation | Hierarchical convolutional stages followed by MBConv, blocked local attention, and grid attention |
| Core mechanism | Block attention captures local interactions; grid attention provides global interactions with linear complexity in image size |
| Adaptation | Replace the single-label classifier with independent 165 logits; tuning mode remains open |
| Input policy | Square-compatible checkpoint transform must be audited against the project's 3:2 images |

The MaxViT paper claims global-local interaction throughout the hierarchy by
combining local blocked attention and dilated global grid attention with
convolutions. This may preserve local evidence earlier than a pooled CNN while
avoiding full quadratic global attention. It is not interchangeable with C2's
shifted-window/patch-merging mechanism.

## Feature flow and adaptation boundary

The TorchVision model applies a convolutional stem, repeated MaxViT blocks, and
a classifier consisting of adaptive average pooling, flattening, layer norm,
a projection, `Tanh`, and a final linear layer. For `maxvit_t`, the stock final
linear output is 1,000 classes; the canonical adapter replaces it with a
165-output independent logit layer while retaining the preceding pooled
projection unless 4A.3 explicitly defines a different head.

The block/grid attention is spatially structured, but the default classifier
still pools before output. A C4 query head can use an earlier feature map as a
separate protocol; it must not be credited to MaxViT's backbone alone.

## Pretraining and task-relevant evidence

| Source | Evidence type | Finding used here | Transfer boundary |
| --- | --- | --- | --- |
| [MaxViT, Tu et al., ECCV 2022](https://arxiv.org/abs/2204.01697) | Primary architecture evidence | Multi-axis attention blends blocked local and dilated global interactions with convolution and reports linear-complexity global-local modelling. | ImageNet and generic downstream results do not establish recipe-level ingredient inference. |
| [Official Google Research MaxViT repository](https://github.com/google-research/maxvit) | Official implementation/checkpoint evidence | Tiny/Small checkpoints and TensorFlow reference code are documented; the repository states that MaxViT sees globally in earlier high-resolution stages. | The repository was archived on 2024-08-05; maintenance, TensorFlow/PyTorch conversion, and checkpoint terms require review. |
| [TorchVision MaxViT implementation and weight metadata](https://docs.pytorch.org/vision/main/_modules/torchvision/models/maxvit.html) | Maintained implementation evidence | `maxvit_t` is available with a 30.9M-parameter checkpoint, 224-square transform, and reproducible download URL. | The released weight contract is square; 3:2 adaptation and exact local memory remain unmeasured. |
| [Visual Food Ingredient Prediction Using Deep Learning with Direct F-Score Optimization](https://www.mdpi.com/2304-8158/14/24/4269) | Direct food multi-label evidence | A Recipe1M comparison evaluates MaxViT-T alongside SwinV2-T, EfficientNetV2-S, and ResNet-50 with a simple image-only ingredient head; MaxViT-T is reported as the strongest of those source conditions. | The paper uses a different dataset, labels, split, fixed-threshold/F-score objective, and source-specific comparisons; it cannot rank C5 on Yummly. |
| [FoodSeg103 / ReLeM](https://arxiv.org/abs/2105.05409) | Adjacent food localisation evidence | Ingredient-level images and local segmentation motivate a backbone that retains spatial structure. | Pixel-level visible ingredients differ from recipe-level weak labels. |

## Project fit and transfer limits

| Requirement | Assessment | Reason and limitation |
| --- | --- | --- |
| R1 fixed 165-label output | Strong | Standard classifier replacement exposes one independent logit per target. |
| R2 partial observability/local evidence | Moderate-high | Local block and global grid interactions are a plausible mechanism for visible fragments plus dish context; hidden ingredients remain ambiguous. |
| R3 sparse positives/long tail | Moderate | Compact capacity and spatial features are compatible; no dedicated imbalance solution is built in. |
| R4 label co-occurrence/shortcuts | Moderate | Global attention may amplify dish/cuisine priors; non-visual controls are required. |
| R5 small 3:2 inputs | Uncertain | TorchVision weights use 224×224; partition sizes and positional assumptions need an aspect-preserving policy. |
| R6 provenance/leakage | Moderate | ImageNet and TorchVision paths are auditable, but the archived reference and any converted checkpoint require a precise provenance record. |
| R7 ranking/calibration | Strong | Independent logits support the project's AP/calibration instrumentation. |
| R8 reproducibility/8 GB | Moderate-high | 30.9M parameters and 5.558 GFLOPs are compact on paper; activation memory at non-square input is unverified. |
| R9 fair comparison | Strong if transform is matched | MaxViT can share the common head/split/metrics, but C2 must use a comparable input area. |
| R10 one declared seed | Strong | No repeated-seed claim is required at intake. |
| R11 falsifiability | Strong | The local-plus-global attention hypothesis can be contrasted directly with C2 and C1. |

## Canonical protocol and alternatives

The recommended canonical C5 protocol is:

1. use the maintained TorchVision `maxvit_t` checkpoint and record its hash;
2. replace the stock ImageNet classifier with a 165-output independent head;
3. choose one square-compatible, aspect-preserving input policy and document
   any padding or interpolation; and
4. keep the same loss, metrics, split, and one-seed budget as the other
   candidates.

Frozen/partial/full tuning, an earlier spatial head, a C4 query head, and
TensorFlow-to-PyTorch checkpoint conversion are alternatives, not silent parts
of the canonical family claim. The MaxViT paper's arbitrary-resolution
complexity does not automatically mean the released TorchVision weights are
safe for arbitrary aspect ratios.

## Access, checkpoint, licence, and provenance

The original Google Research repository is archived but provides an
[Apache-2.0 licence](https://raw.githubusercontent.com/google-research/maxvit/main/LICENSE).
TorchVision supplies a maintained PyTorch implementation and a public
`maxvit_t-bc5ab103.pth` download URL; the TorchVision source is under the [BSD
3-Clause licence](https://github.com/pytorch/vision/blob/main/LICENSE). The
Phase 5 manifest must state whether weights are taken from TorchVision or
converted from the archived TensorFlow repository and must preserve the
corresponding attribution and checkpoint hash.

The installed TorchVision 0.23.0 metadata reported:

| Variant | Parameters | Published operations | Weight transform |
| --- | ---: | ---: | --- |
| Tiny | 30,919,624 | 5.558 GFLOPs | 224×224 crop/resize, ImageNet mean/std, bicubic |

The table names Tiny because the constructor is `maxvit_t`; the parameter and
operation values are library metadata at the released square input, not a
local peak-memory result. The original paper reports larger family variants,
but they are not plausible starting points until the compact path is measured.

## Repository integration path

Phase 5 would add a wrapper under `src/models/` that replaces the classifier,
implements `BaseModel` transforms and serialization, exposes a real MaxViT
block for visualization, and preserves raw logits. If the current TorchVision
constructor enforces a square partition, the wrapper must make the padding or
resize policy explicit rather than silently distorting 3:2 images.

Required smoke checks are offline weight loading, output shape `(B, 165)`,
checkpoint reconstruction, transform behaviour on representative aspect
ratios, attention/block target traversal, one forward/backward pass, and peak
memory/throughput at the declared scale.

## Risks, go/no-go conditions, and open questions

**Go for 4A.3 comparison:** C5 has a distinct hybrid attention mechanism,
direct recent food-ingredient evidence, a compact maintained TorchVision path,
and an explicit comparison against Swin/C1.

**Go/no-go questions:**

- Can the maintained TorchVision model consume the benchmark's 3:2 images
  without a destructive crop or partition failure?
- Is the archived official repository's checkpoint provenance needed, or is the
  TorchVision weight sufficient and legally/technically auditable?
- Does hybrid local/global attention add information beyond C2 at a matched
  token area and memory budget?
- Does the default pooled head hide the very spatial mechanism motivating C5?

**Invalidating evidence:** an unfixable square-input restriction, inability to
  verify a checkpoint under acceptable terms, or no plausible 8 GB path at the
  smallest variant would demote C5. The architecture evidence remains relevant
  to the custom attention design.

## Falsifiable local hypothesis and minimal comparison

**Hypothesis H-C5:** at matched input area and independent head, MaxViT-T will
improve validation AP for labels requiring both local texture and global dish
context over Swin V2-T, while preserving a comparable or lower resource cost.

The minimal test is one declared-seed C5 run against the same-protocol C2 (or
C1 if C2 is not retained), with identical target metadata, transforms, loss,
budget, and metrics. Report macro/micro AP, support and observability slices,
calibration, and measured memory/throughput. The Recipe1M/F-score result is
context only and cannot substitute for this like-for-like Yummly test.

## Comparison anchors

| Anchor | Role |
| --- | --- |
| [ResNet local contract](../../../implementation_details/models.md) | Existing pooled supervised-CNN control. |
| [DINOv2 local deep dive](../../../models_deepdive/dinov2.md) | Existing visual self-supervised comparison anchor; not a MaxViT variant. |

## Handoff to 4A.3

Reuse the C5 mechanism boundary, TorchVision/archived access distinction,
resource metadata, aspect-ratio risks, and H-C5 hypothesis. 4A.3 must decide
whether MaxViT is complementary to Swin or should be replaced by a lower-risk
family; no source benchmark score may settle that decision alone.

## References

- [MaxViT paper](https://arxiv.org/abs/2204.01697)
- [Official Google Research repository](https://github.com/google-research/maxvit)
- [TorchVision MaxViT implementation](https://docs.pytorch.org/vision/main/_modules/torchvision/models/maxvit.html)
- [TorchVision MaxViT source](https://github.com/pytorch/vision/blob/main/torchvision/models/maxvit.py)
- [Visual Food Ingredient Prediction Using Deep Learning with Direct F-Score Optimization](https://www.mdpi.com/2304-8158/14/24/4269)
- [FoodSeg103 / ReLeM](https://arxiv.org/abs/2105.05409)
