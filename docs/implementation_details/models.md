# Vision models

This page describes the implementation of the models available in `src/models` and their contract with the training pipeline. The problem remains a multi-label classification task: each model outputs a vector of `num_classes` **logits**, with no final sigmoid. Converting logits to probabilities and applying `BCEWithLogitsLoss` are responsibilities of the Lightning module.

The documentation focuses on the integration aspects and architectural decisions of the repository; it does not repeat the introductory theory of CNN, residual connection or transformer. Sections marked *to be expanded* are intentionally an initial outline only.

## Architectural insights

For machine-learning details, network internals, and the relevant research, see [`docs/models_deepdive/`](../models_deepdive/). A deep dive on [DINOv2 ViT-B/14](../models_deepdive/dinov2.md) is currently available; deep dives on the remaining models will be added to the same directory.

## Common contract: `BaseModel`

`BaseModel` is the common interface for all vision models. It stores `num_classes`, the square input size, and the transform builders; it also exposes `transform_aug` and `transform_plain`, used by the DataModule for training and validation/inference respectively. The transforms are therefore part of the model's serializable configuration rather than an external detail of the run.

Each subclass must expose `conv_target_layer` and `classifier_target_layer`. These hooks are consumed by the visualization dashboard (e.g. Grad-CAM) and must refer to modules actually traversed by the `forward`.

### Serialization and reconstruction

`to_config()` records the concrete type and common parameters; `load_from_config()` validates that the requested type matches the class that is rebuilding the object. Pretrained wrappers extend this payload with their own options. Non-standard transformation callables remain Python objects in the configuration: their persistence therefore requires the project's normal checkpointing/configuration mechanism, not stand-alone portable JSON serialization.

### Layer-wise pretraining in custom ResNets

The optional layer-wise pretraining (LP) protocol is implemented in `BaseModel` and concretized only by the custom ResNet family. With `lp_phase >= 0`, the `layer1`–`layer4` blocks are initially replaced by `Identity`; the available trunk and a head compatible with its number of channels remain trainable. Each call to `lp_phase_step()` freezes the last trained stage, installs the next stage, and recreates the classifier. After the last phase, all parameters are thawed again and `lp_phase` changes to `-1`.

This mechanism modifies the effective topology during training: it is not a simple learning rate scheduler. Checkpoint and resume must therefore keep `lp_phase` consistent; DenseNet families and torchvision wrappers do not support it.

## ResNet custom

The `ResnetLikeV1`, `ResnetLikeV1LVariant` and `ResnetLikeV2` classes share the `7×7, stride 2` stem followed by batch normalization, ReLU and max-pooling. The head is always `AdaptiveAvgPool2d(1) → Flatten → Linear`, so the number of classes can change independently of the final spatial resolution.

### Blocks and channel progression

`ResnetLikeV1` replicates the depth of ResNet-18: two `BasicBlocks` for each of the four stages. A `BasicBlock` uses two `3×3` convolutions; when strides or channels do not match, the identity branch becomes a `1×1` projection with batch normalization. The channels follow `64 → 64 → 128 → 256 → 512`, with downsampling at the input of the last three stages.

`ResnetLikeV1LVariant` keeps the same configuration but replaces the block activation with `LeakyReLU`. The stem remains unchanged, so the variant only changes the non-linearity of the residual branches.

`ResnetLikeV2` follows the ResNet-50 structure (`3, 4, 6, 3` blocks), with a fourfold-expansion `BottleneckBlock`: `1×1` for compression, `3×3` for feature extraction, and `1×1` for expansion. In the constructors the head is initially created with 512 features, but `_make_classifier()` applies `LAYER_EXPANSION = 4`; it therefore receives the 2,048 features produced by the final stage.

### Operational implications

Custom models use the project's generic transformation builders, not those tied to ImageNet weights. Their `conv_target_layer` is the final block of the last stage (or the block below, also in LP), while the classifier's target is the last `Linear`. They are therefore directly usable by the dashboard interpretability tools.

## DenseNet custom

`DensenetLikeV1` and `DensenetLikeV2` share the custom ResNet stem, but replace residual composition with feature concatenation. Each `DenseLayer` applies the pre-activation sequence `BN → ReLU → 1×1 → BN → ReLU → 3×3` and concatenates the original input with its output. The growth rate is 32: each layer adds exactly 32 channels to the tensor in that stage.

### Compression and size

The internal `1×1` convolution operates on 128 channels (`growth_rate × 4`), limiting the cost of the subsequent `3×3`. After each of the first three dense blocks, `TransitionLayer` runs `BN → ReLU → 1×1 → AvgPool2d(2)` and halves both the channel count (`reduction_factor = 0.5`) and the resolution. A final normalization and ReLU precede global average pooling and the linear classifier.

| Model | Layer for dense blocks | Channels after dense blocks | Channels after the transition |
| --- | --- | --- | --- |
| `DensenetLikeV1` | 6, 12, 24, 16 | 256, 512, 1024, 1024 | 128, 256, 512 |
| `DensenetLikeV2` | 6, 12, 48, 32 | 256, 512, 1792, 1920 | 128, 256, 896 |

The use of concatenation preserves features from all previous layers, but increases memory pressure, especially in the third and fourth blocks of V2. There is no LP provided for these models; the received parameter is neutralized in the base constructor.

## DINOv2 ViT-B/14 with registers

`DinoV2B14` loads the `dinov2_vitb14_reg_lc` model from the `facebookresearch/dinov2` repository through `torch.hub`. The backbone is a base Vision Transformer with `14×14` patches and register tokens; the suffix `_lc` selects the variant equipped with a linear classifier. The class replaces `model.linear_head` with a new `Linear` whose output size is `num_classes`, so the upstream checkpoint head is not reused for ingredient prediction.

### Freezing and fine-tuning

By default `freeze_backbone=True`: `freeze_backbone()` sets `requires_grad=False` on the backbone parameters, leaving the new linear head trainable. `unfreeze_backbone()` enables full fine-tuning later. The declared batch limit is 32 via `max_allowed_batch_size`, useful for the pipeline to avoid configurations that are too large for the GPU.

The `pretrained` parameter is preserved in the configuration, but the current implementation still calls `torch.hub.load(...)` without using it to choose weights or architecture: the backbone loading is therefore always the one defined by torch.hub. This is an important detail if you want a true start from random weights.

### Preprocessing and interpretability

DINOv2 uses dedicated builders (`transform_*_dino`). If an augmentation function is passed, training enables `random_crop=True`, while validation/inference uses `random_crop=False`; without an override, the configured builders are used directly. For Grad-CAM the target is `backbone.blocks[-1].norm1`: its token activations are converted back to the patch grid, removing CLS and register tokens. The classifier remains `linear_head`.

## Torchvision ResNet wrapper

*To be expanded.* `Resnet18` and `Resnet50` build their respective torchvision architectures, optionally with `DEFAULT` weights, and replace `model.fc` with a projection to `num_classes`. They also set up transformations compatible with ImageNet weights and publish `layer4[-1]` as the visual target.

## DenseNet torchvision wrapper

*To be expanded.* `Densenet121` and `Densenet201` replace the torchvision classifier after the DenseNet feature extractor. The interpretability targets are the last module of `model.features` and the linear classifier. This section will be extended with the implications of constructor variants and pretrained transformations.

## Dummy models

*To be expanded.* `DummyModel` and `DummyBNModel` are test networks with three convolutional blocks and max-pooling; the second inserts batch normalization. They are useful for validating training, shapes and dashboards without the cost of the main backbones, not as a competitive architectural baseline.

## Schedulers present in `src/models`

*To be expanded.* `WarmStartReduceOnPlateau` and `ConstantStartReduceOnPlateau` derive from `ReduceLROnPlateau` and work around the historical incompatibility between `SequentialLR` and Lightning. The first interpolates the learning rate from `warm_start` to `warm_stop` (linear or with `tanh`) before delegating to the plateau logic; the second keeps the initial LR in the waiting phase.

## Code references

- `src/models/commons.py`: common contract, transformations and LP.
- `src/models/resnet.py`: Custom ResNet and torchvision wrapper.
- `src/models/densenet.py`: DenseNet custom and torchvision wrapper.
- `src/models/dinov2.py`: DINOv2 wrapper and frozen backbone management.
- `src/models/dummy.py`: minimal models for testing.
- `src/models/custom_schedulers.py`: custom scheduler.
