# MaxViT-T experimental representation

**Created:** 2026-10-07
**Last updated:** 2026-10-07

## Variant and scope

This explains [MaxViTTExperiment](../../src/models/experimental_maxvit.py), the TorchVision `maxvit_t` / `MaxVit_T_Weights.IMAGENET1K_V1` path adopted by 4A-D1. Exact configuration, artifact/normalization provenance, restoration and measurements belong to the [runtime owner](../implementation_details/experimental_model_contract.md). The [candidate dossier](../research/topics/experimental_model_candidates/maxvit.md) owns retained research evidence.

## Representation and readout

The stem maps a 224 canvas to `[B,64,112,112]`. Four hierarchical blocks downsample to feature maps `[B,64,56,56]`, `[B,128,28,28]`, `[B,256,14,14]` and `[B,512,7,7]`. Their layer counts are 2/2/5/2. Each layer combines an MBConv-style convolutional operation with block and grid attention, giving 22 partition-attention modules in total.

Block attention groups nearby spatial locations. Grid attention groups positions sampled across spatial blocks, allowing interactions over larger regions while retaining the hierarchical feature representation. The released learned relative-position biases and their indexing buffers are preserved. Partition size seven and stage grids are validated; this adapter does not advertise arbitrary-resolution operation.

The stock classifier is GAP → flatten → LayerNorm → 512-to-512 biased linear → Tanh → biasless ImageNet projection. The experiment replaces this **entire** readout with GAP → flatten → seeded biased `Linear(512,L)`. This aligns the established pair's output operation but intentionally changes the released classifier. Backbone parameters total 30,143,944; the new head adds `513L`, yielding 30,228,589 at L=165.

## Adaptation and interpretation

The original supervised ImageNet artifact supplies encoder parameters/buffers. Full adaptation trains stem, blocks and new head. The same-topology frozen alternative holds stem/blocks persistently in eval while the head trains; input-gradient diagnostics remain possible.

The chosen installed BatchNorm policy is epsilon 1e-3/momentum 0.01, preserving original running statistics. The [official model documentation](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.maxvit_t.html) describes historical pretraining momentum 0.99; the runtime owner explains why that is not silently substituted for the current constructor's 0.01. Effective batch accumulation does not reproduce larger-physical-batch statistics.

The shared 224 fit/pad input retains the frame but changes the image distribution relative to the original crop transform. Local/global interactions motivate the experiment; they do not establish better ingredient prediction. Parameter/FLOP estimates and pretrained ImageNet scores cannot replace measured execution or a controlled ingredient benchmark.

Grad-CAM hooks the real final 512-channel 7×7 block. The final new linear scores factorization concepts and pooled image features. Neither diagnostic proves direct ingredient presence. Different pretrained recipes, internal regularization and later HPO mean a comparison concerns model protocols, not attention as an isolated causal effect.

## Evidence and references

The [source](../../src/models/experimental_maxvit.py), [graph/state tests](../../tests/test_experimental_maxvit.py) and [runtime tests](../../tests/test_experimental_maxvit_runtime.py) verify this realization. See the [original MaxViT paper](https://arxiv.org/abs/2204.01697) and [pinned TorchVision source](https://github.com/pytorch/vision/blob/824e8c8726b65fd9d5abdc9702f81c2b0c4c0dc8/torchvision/models/maxvit.py). Research hypotheses and fallbacks remain in the [binding portfolio](../project_objective/experimental_model_portfolio.md).
