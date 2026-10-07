# Vision models

**Created:** 2026-08-02
**Last updated:** 2026-10-07

This page describes the implementation of the models available in `src/models` and their contract with the training pipeline. The problem remains a multi-label classification task: each model outputs a vector of `num_classes` **logits**, with no final sigmoid. Converting logits to probabilities and applying `BCEWithLogitsLoss` are responsibilities of the Lightning module.

The documentation focuses on the integration aspects and architectural decisions of the repository; it does not repeat the introductory theory of CNN, residual connection or transformer. Sections marked *to be expanded* are intentionally an initial outline only.

## Architectural insights

For machine-learning details, network internals, and the relevant research, see [`docs/models_deepdive/`](../models_deepdive/). A deep dive on [DINOv2 ViT-B/14](../models_deepdive/dinov2.md) is currently available; deep dives on the remaining models will be added to the same directory.

## Common contract: `BaseModel`

The [experimental-model contract](experimental_model_contract.md) owns opt-in 224 preprocessing/identity, deterministic head initialization, exact accumulation and strict offline persistence. `EfficientNetV2SExperiment`, `MaxViTTExperiment`, `DICANetSExperiment` and `ExperimentalLGNM` are implemented. Custom diagnostic consumer integration and all measured CUDA/real-consumer qualification remain pending. Legacy/default vocabulary and selector behavior are unchanged; a 4A adapter must use the explicit experimental Lightning path.

`BaseModel` is the common interface for vision models integrated with training. `DICANetSExperiment` wraps its plain tensor core in this interface. `BaseModel` stores `num_classes`, the square input size, and the transform builders; it also exposes `transform_aug` and `transform_plain`, used by the DataModule for training and validation/inference respectively. The transforms are therefore part of the model's serializable configuration rather than an external detail of the run.

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

By default `freeze_backbone=True`: `freeze_backbone()` sets `requires_grad=False` on the backbone parameters, leaving the new linear head trainable. `unfreeze_backbone()` enables full fine-tuning later. `max_allowed_batch_size` currently returns `None`, so this wrapper does not enforce a model-level physical batch cap; feasible batch size must be established by the measured resource smoke test for the concrete protocol.

The `pretrained` parameter is preserved in the configuration, but the current implementation still calls `torch.hub.load(...)` without using it to choose weights or architecture: the backbone loading is therefore always the one defined by torch.hub. This is an important detail if you want a true start from random weights.

### Preprocessing and interpretability

DINOv2 uses dedicated builders (`transform_*_dino`). If an augmentation function is passed, training enables `random_crop=True`, while validation/inference uses `random_crop=False`; without an override, the configured builders are used directly. For Grad-CAM the target is `backbone.blocks[-1].norm1`: its token activations are converted back to the patch grid, removing CLS and register tokens. The classifier remains `linear_head`.

## EfficientNetV2-S Phase 3 selector

`EfficientNetV2SSelector` is the maintained, role-specific implementation of
the frozen 4B-D1 reference learner. It always loads
`EfficientNet_V2_S_Weights.IMAGENET1K_V1`, retains the stock adaptive pooling
and `Dropout(p=0.2, inplace=True)`, and replaces only the classifier linear
layer with a biased `Linear(1280, num_classes)`. The constructor rejects random
initialization, non-384 inputs, and layer-wise pretraining, and asserts that
every backbone and head parameter remains trainable.

Its model-owned transforms preserve the full frame through an exact
long-side-384 round-half-up resize and ImageNet-mean center padding. Training
adds only a horizontal flip; validation and audit transforms are deterministic.
This wrapper is deliberately distinct from the implemented 4A EfficientNetV2-S
comparison implementation, whose role and input contract differ.

The exact loss, optimizer, scheduler, audit cadence, blind-pilot gate, and
resource result are documented in
[`ingredient_selection.md`](ingredient_selection.md).
The wrapper exposes `MAX_ALLOWED_BATCH_SIZE = 8` for the measured 384-pixel
true-FP32 full-fine-tuning contract on the RTX 4060. The specialized Phase 3
launcher requests effective 128 and resolves physical 8 with Lightning
accumulation 16; the cap is hardware/protocol-specific, not an architecture
constant that guarantees memory feasibility on every device.

## EfficientNetV2-S 4A experiment

[`EfficientNetV2SExperiment`](../../src/models/experimental_efficientnet.py) retains TorchVision's intact features and global pool, replacing its entire stock readout with seeded biased `Linear(1280,L)` and no classifier dropout. The shared full-frame input is 224 pixels, not the selector's 384. Full and persistent-eval frozen-encoder modes preserve input-gradient diagnostics; configuration records the exact original ImageNet enum, primitive contract and adaptation. Widths 1/50/59/165 and the 20,388,853-parameter count at165 are verified with no-network fixtures.

[`ExperimentalLGNM`](../../src/lightning/experimental_lgn.py) integrates exact batching, sample-weighted final/truncated accumulation groups, ordered classes and full/light checkpoint guards through canonical construction. The [owning contract](experimental_model_contract.md#experimental-efficientnet-and-canonical-runtime--52) records interfaces, tests and limits. CPU fixture evidence is not actual ImageNet artifact verification, a CUDA cap or completed real dashboard qualification; these remain 5.5.

## MaxViT-T 4A experiment

[`MaxViTTExperiment`](../../src/models/experimental_maxvit.py) retains TorchVision's intact stem and MaxViT blocks with a new GAP/flatten/biased `Linear(512,L)` readout. The whole stock normalization/projection/Tanh classifier is removed. It shares EfficientNet's 224 full-frame transform and seeded head policy; the saved contract explicitly validates partition-seven geometry and BatchNorm epsilon 1e-3/momentum 0.01, preserving pretrained running statistics. Full/frozen behavior covers both stem and blocks, and frozen state remains eval while input gradients work.

The traversed final block provides `[B,512,7,7]` diagnostic features; the final linear supplies the complete classifier for pooled image/concept vectors. Counts, class identity, strict full/light offline restoration and production Grad-CAM/factorization helpers are verified with synthetic CPU fixtures. The actual approved artifact and retained backbone state are also verified. The [owning record](experimental_model_contract.md#experimental-maxvit-and-artifact-provenance--53) records exact hashes, notices, normalization rationale and limits. Actual CUDA capacity and real dashboard acceptance remain 5.5.

## DICA-Net-S 4A experiment

[`DICANetSCore` and `DICAReadoutS`](../../src/models/dica_net.py) implement the user-named **Dual-scale Ingredient-query and Context Attention Network**, historically P2-S. The core consumes an intact EfficientNetV2-S feature stack once and combines ingredient-query attention over 14×14 and 7×7 maps with a pooled context branch. It returns raw ordered logits; S fixes width 128, four attention heads and one query/FFN block at every vocabulary size.

CPU tensor tests verify both-path gradients, reference attention and label-row permutation/subsetting, including the published 59-label projection. The [tensor contract](experimental_model_contract.md#dica-net-s-tensor-core--541) records counts and numerical tolerances.

[`DICANetSExperiment`](../../src/models/experimental_dica_net.py) loads the approved original ImageNet encoder, verifies its complete artifact hash and retains every feature weight/buffer through custom initialization. It provides common 224 preprocessing, full or persistent-eval frozen adaptation, native BatchNorm epsilon 1e-3/momentum 0.1 with preserved statistics, strict configuration/provenance and complete offline full/light restoration through `ExperimentalLGNM`. The serialized identity remains `p2_s`. The [initialization/persistence owner](experimental_model_contract.md#dica-net-s-initialization-and-persistence--542) records original-artifact evidence, tests and limits.

The diagnostic classifier is the complete two-scale readout, never the context-only linear. Standalone-concept factorization raises an explicit unsupported-operation error. Capability-aware dashboard handling and Grad-CAM acceptance remain 5.4.3; measured CUDA/real-runtime qualification remains 5.5. Interface/persistence completion does not authorize benchmark training.

## Torchvision ResNet wrapper

*To be expanded.* `Resnet18` and `Resnet50` build their respective torchvision architectures, optionally with `DEFAULT` weights, and replace `model.fc` with a projection to `num_classes`. They also set up transformations compatible with ImageNet weights and publish `layer4[-1]` as the visual target.

## DenseNet torchvision wrapper

*To be expanded.* `Densenet121` and `Densenet201` replace the torchvision classifier after the DenseNet feature extractor. The interpretability targets are the last module of `model.features` and the linear classifier. This section will be extended with the implications of constructor variants and pretrained transformations.

The wrappers are not currently usable through the normal model-owned transform
contract: both constructors keep the torchvision weight enum in a local
`weights` variable, while `_BaseDensenet.transform_aug` and
`transform_plain` read `self.tr_weights`, which is never initialized. Accessing
either transform therefore raises `AttributeError`. This must be fixed and
smoke-tested before a torchvision DenseNet is treated as a maintained training
or reference-selector path.

## Dummy models

*To be expanded.* `DummyModel` and `DummyBNModel` are test networks with three convolutional blocks and max-pooling; the second inserts batch normalization. They are useful for validating training, shapes and dashboards without the cost of the main backbones, not as a competitive architectural baseline.

## Schedulers present in `src/models`

`WarmStartReduceOnPlateau` and `ConstantStartReduceOnPlateau` derive from `ReduceLROnPlateau` and work around the historical incompatibility between `SequentialLR` and Lightning. The first interpolates the learning rate from `warm_start` to `warm_stop` (linear or with `tanh`) before delegating to the plateau logic; the second keeps the initial LR during the waiting phase.

Both schedulers own their optional `verbose` flag instead of relying on an attribute of the PyTorch parent class. This is required by PyTorch 2.8, whose `ReduceLROnPlateau` no longer creates that legacy attribute. Existing scheduler states and Lightning checkpoints that predate this field remain loadable: the constructor establishes the default before the stored scheduler fields are restored. The reduction path and this compatibility case are covered by [`tests/test_custom_schedulers.py`](../../tests/test_custom_schedulers.py).

## Code references

- `src/models/commons.py`: common contract, transformations and LP.
- `src/models/resnet.py`: Custom ResNet and torchvision wrapper.
- `src/models/densenet.py`: DenseNet custom and torchvision wrapper.
- `src/models/dinov2.py`: DINOv2 wrapper and frozen backbone management.
- `src/models/dummy.py`: minimal models for testing.
- `src/models/custom_schedulers.py`: custom scheduler.
