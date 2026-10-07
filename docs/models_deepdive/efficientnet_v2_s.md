# EfficientNetV2-S experimental representation

**Created:** 2026-10-07
**Last updated:** 2026-10-07

## Variant and scope

This explains the actual [224-pixel experiment adapter](../../src/models/experimental_efficientnet.py), initialized from TorchVision `EfficientNet_V2_S_Weights.IMAGENET1K_V1`. Its [runtime owner](../implementation_details/experimental_model_contract.md) owns configuration, exact artifact hashes, qualification and persistence. The [candidate dossier](../research/topics/experimental_model_candidates/efficientnet_v2.md) retains literature evidence. The separate 384-pixel Phase 3 selector is a different protocol.

## Representation and readout

The intact feature stack mixes early fused convolutional expansion blocks with later mobile inverted bottlenecks, depthwise spatial convolution and squeeze-excitation. Residual/stochastic-depth operations remain those of the released implementation. Squeeze-excitation recalibrates channels; it is not an ingredient-specific spatial prediction.

At the adopted input, the traversed eight-stage stack is:

| Stage | Output excluding batch dimension |
| --- | --- |
| Stem / `features[0]` | 24 × 112 × 112 |
| `features[1]` | 24 × 112 × 112 |
| `features[2]` | 48 × 56 × 56 |
| `features[3]` | 64 × 28 × 28 |
| `features[4]` | 128 × 14 × 14 |
| `features[5]` | 160 × 14 × 14 |
| `features[6]` | 256 × 7 × 7 |
| `features[7]` | 1,280 × 7 × 7 |

Global average pooling discards explicit spatial locations and produces `[B,1280]`. A new biased linear head produces independent raw `[B,L]` logits. The whole stock classifier, including dropout, is replaced; the ImageNet classifier is not mapped to ingredient names. The backbone has 20,177,488 parameters; the new head adds `1281L`, giving 20,388,853 at L=165. Changing L changes only output rows, not the encoder.

## Adaptation and consequences

Original supervised ImageNet weights supply a visual prior; recipe targets remain weak labels that can be predicted from context. Shared full-frame fit/pad 224 preserves framing but reduces image content resolution and differs from the released 384-pixel center-crop evaluation transform. Published ImageNet accuracy/FLOPs therefore do not describe this ingredient protocol. See [official weight/transform documentation](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.efficientnet_v2_s.html).

Full adaptation trains encoder and head with native BatchNorm epsilon 1e-3/momentum 0.1. Frozen adaptation keeps every encoder operation in eval, including BatchNorm and stochastic depth; the new head trains. Accumulation changes gradient grouping, not physical-batch normalization. The actual supported modes and measured resource conditions belong to the runtime owner, rather than parameter-count estimates here.

The final feature stage supplies Grad-CAM. The new linear head also scores factorization concept vectors. These summaries explain output sensitivity/representation; they do not certify ingredient visibility or localization. This pooled model is DICA-Net's head comparator only when data, adaptation and downstream policy are actually matched.

## Evidence and references

The [source](../../src/models/experimental_efficientnet.py), [real-graph tests](../../tests/test_experimental_efficientnet.py) and canonical runtime tests verify structure, state and output identity. Original [EfficientNetV2 research](https://proceedings.mlr.press/v139/tan21a.html) explains the architectural motivation; the local implementation follows the pinned TorchVision realization. No benchmark effectiveness follows from this explanation or a bounded smoke.
