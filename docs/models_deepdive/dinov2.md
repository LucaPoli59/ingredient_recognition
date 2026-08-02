# Deep dive: DINOv2 ViT-B/14 for ingredient prediction

**Creation date:** 2026-07-28

## Purpose and scope

`DinoV2B14` (`src/models/dinov2.py`) adapts the upstream checkpoint `dinov2_vitb14_reg_lc` from `facebookresearch/dinov2` to the multi-label problem. In this pipeline it is not a generative model nor is the DINO self-supervised objective re-executed: it is an already pre-trained Vision Transformer extractor, to which the repository applies a supervised linear head to estimate the ingredients.

The forward returns a tensor `[B, C]`, where `B` is the batch and `C = num_classes` is the number of ingredients encoded by the DataModule. Values ​​are independent logits; the sigmoid is applied in the Lightning loss/valuation, not within the model. As a result, the model remains compatible with `BCEWithLogitsLoss` and multi-hot targets.

## The DINOv2 model: architecture and representation

### Which variant is loaded

The project uses `dinov2_vitb14_reg_lc`: **ViT-Base**, `14×14` patch, four register tokens and a later replaced upstream linear head for the ingredients task. The ViT-B/14 variant distributed by DINOv2 has approximately 86 million parameters; the distilled backbone has hidden dimension `D = 768`, 12 encoder blocks and 12 attention heads per block. The internal MLP is standard for distilled ViTs, with typically `4D` expansion (3072 hidden units) before projecting back to 768.

| Component | ViT-B/14 configuration | Practical consequence |
| --- | --- | --- |
| Patch embedding | conv/linear projection on `14×14` patches | transforms local regions into tokens, without subsequent convolutions |
| Visual tokens at `224×224` | `16×16 = 256` | spatial grid at 1/14 resolution of the image |
| Token Size | 768 | size of each feature propagated by the transformer |
| Encoders | 12 blocks | global context built iteratively |
| Multi-head attention | 12 head, 64 size/head | relations between patches in different subspaces |
| Feed-forward network | 768 → 3072 → 768 | token-wise nonlinear transformation after attention |
| Parameters | ~86 M in the ViT-B backbone | larger than the project's custom CNNs; the cost grows with the number of tokens |

The `768 / 12 / 12` details and the use of an MLP for the distilled ViT-B variant are given in the architectural table of the DINOv2 work. The `B/14` choice is a compromise: smaller patches increase feature resolution but roughly quadruple the attention cost when the number of tokens doubles per side.

### From image to token sequence

Given an input $x \in \mathbb{R}^{B \times 3 \times H \times W}$ with $H$ and $W$ divisible by 14, the patch embedding divides the image into non-overlapping patches. The number of patches is:

$$N = \frac{H}{14}\frac{W}{14}.$$

Each patch $p_i \in \mathbb{R}^{3 \cdot 14 \cdot 14}$ is projected into $e_i \in \mathbb{R}^{768}$. Operationally this is equivalent to a `Conv2d(3, 768, kernel_size=14, stride=14)`, followed by flattening the grid. At `224×224` the result is a sequence of 256 embeddings; at `518×518` it is a grid of `37×37 = 1369` patch tokens. Interpolatable positional embeddings provide the transformer with positional information that attention alone does not contain.

To the sequence the model adds a learned **class token**, intended to collect image-level information, and four learned **register tokens**. Registers do not represent regions of the image: they are global memory slots without fixed spatial semantics. The elaborated sequence then takes the form:

$$Z_0 = [c; r_1; \ldots; r_R; e_1; \ldots; e_N] + P,$$

where $c$ is the class token, $r_j$ are the four registers and $P$ indicates the positional components applicable to visual tokens. At `224×224` the model then processes `1 + 4 + 256 = 261` tokens. Registers have a small attention cost at usual resolutions but not zero.

### What a transformer block does

Each ViT-B block follows a **pre-normalization** structure with two residual branches. For a sequence $Z_{l-1}$:

$$U_l = Z_{l-1} + \operatorname{MSA}(\operatorname{LN}(Z_{l-1})),$$
$$Z_l = U_l + \operatorname{MLP}(\operatorname{LN}(U_l)).$$

In DINOv2 there are also training stabilization mechanisms, such as LayerScale and stochastic depth in the pretraining recipe; in the ViT-B distilled variant the declared drop rate is zero. The normalization before each subblock leaves a direct residual path, favorable for the propagation of the gradient in depth.

Multi-head self-attention calculates, for each head, query, key and value:

$$Q = ZW_Q, \qquad K = ZW_K, \qquad V = ZW_V,$$
$$\operatorname{Attn}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.$$

For ViT-B, $d_h = 768/12 = 64$. Each token can assign weight to all other tokens — distant patches, class tokens, and registers — and not just a local neighborhood as in a convolution. The 12 heads learn distinct $W_Q,W_K,W_V$ matrices and their outputs are concatenated and projected back to 768 dimensions. Attention has quadratic complexity $O(N^2D)$ with respect to the number of patches: going from 224 to 448 pixels, $N$ goes from 256 to 1024 and the quadratic part of attention grows by about 16 times.

After attention, the MLP operates independently on each token: `Linear(768, 3072) → GELU → Linear(3072, 768)`. Attention mixes information between tokens; MLP increases the nonlinear capacity of the representation of each token after such aggregation. Repetition of these two mechanisms allows patches to build contextualized features, not simple local descriptors.

### Why register tokens

Registers were introduced in the *Vision Transformers Need Registers* job. The authors observe that some low-information patch tokens, often in the background, take on abnormally high norms and are reused by the ViT as global computation space; this produces artifacts in attention maps and dense features. Adding dedicated learned tokens to this feature separates working memory from the image, making patch features and attention maps smoother.

For ingredient classification, registers are not directly “ingredient tokens” nor do they replace the class token. Their contribution is indirect: they improve the quality of the features with which class tokens and patch tokens encode the objects, textures and contextual relationships of the dish. This is particularly relevant for interpretability applications: a cleaner attentional map does not demonstrate causality, but reduces an important structural source of backbone artifacts.

## What was learned in the pretraining

### Student–teacher without labels

DINOv2 is pre-trained in a self-supervised discriminative manner: it does not use class labels to provide the target. A **student** receives different crops of the same image and is optimized; a **teacher** generates targets on its own crops and is updated as an exponential moving average (EMA) of the student's weights, not by backpropagation. In schematic terms:

$$\theta_t \leftarrow m\theta_t + (1-m)\theta_s,$$

where the momentum $m$ grows with a cosine schedule during training. The teacher is therefore a temporally more stable version of the student and makes a label-free distillation objective possible.

### DINO global target and iBOT local target

The DINO loss at the image level compares the distributions on the *prototype scores* obtained from the class token of the student and the teacher on different views of the same image. After softmax and teacher centering, the form is a cross-entropy:

$$\mathcal{L}_{\mathrm{DINO}} = -\sum_k p_t(k)\log p_s(k).$$

This part pushes the class token to remain consistent between different crops: it is the component useful for a global classification.

The iBOT loss instead operates on masked patches: some patches are hidden from the student, while the teacher observes the unmasked view; the distributions of the corresponding patches are compared. In compact form:

$$\mathcal{L}_{\mathrm{iBOT}} = -\sum_{i \in \mathcal{M}}\sum_k p_{t,i}(k)\log p_{s,i}(k).$$

The result is important: the backbone must not only summarize the image in the class token, but must preserve dense features at the patch level. DINOv2 uses separate MLP heads for the DINO and iBOT objectives, unlike some previous training recipes that shared the projection.

The **KoLeo** regularizer is added to the loss, based on the distance from the nearest neighbor between normalized features in the batch. Maximizing feature dispersion prevents the representation space from collapsing or excessively concentrating many examples in the same region. The actual pretraining objective is therefore a weighted combination of global consistency, local patch prediction and geometric dispersion.

### Scale, curation and distillation

The DINOv2 result does not derive only from the loss: the paper attributes an essential role to LVD-142M, an automatically curated dataset of 142 million images, to the scalable training recipe and to distillation. The large ViT-g/14 is trained from scratch; the smaller models, including ViT-B/14, are distilled from the larger teacher. For this reason, the ViT-B used here brings knowledge transferred from a much larger model while remaining manageable as a downstream backbone.

It is useful to separate this phase from the repository training: no DINO loss, iBOT or EMA are reapplied here. The current pipeline uses the resulting weights as a feature extractor and only optimizes supervised multi-label loss on the ingredients.

## Relation to the ingredients task

The `_lc` variant does not classify using only the class token. With the hub default `layers=4`, it concatenates the normalized class tokens from the final four blocks with the mean of the normalized patch tokens from the final block. The feature that enters the head therefore has size `5 × 768 = 3840`:

$$h = [c_{9}; c_{10}; c_{11}; c_{12}; \operatorname{mean}(E_{12})].$$

The upstream ImageNet head is `Linear(3840, 1000)`. The repository wrapper reads the input size and replaces it with `Linear(3840, num_classes)`, thus preserving the multi-layer and global aggregation already achieved by the DINOv2 implementation. In the multi-label case, each row of the new head matrix corresponds to an ingredient and learns a direction in the feature space that increases its logit. There is no softmax between ingredients: “tomato”, “basil” and “oil” can simultaneously receive high logits.

Global pretraining promotes robustness to crop, style, background, and domain variations; the patch-level component preserves useful detail for small or localized ingredients. However, a photo of a dish does not guarantee complete observability of the recipe: some ingredients may be mixed, hidden or inferable only from the context. DINOv2 can learn visual and semantic correlations of the training set, but it does not transform non-visible information into certain evidence. Decision thresholds, class imbalances and co-occurrences therefore remain the responsibility of the supervised part of the pipeline.

## Object construction and actual graph

The concrete class sets `weights = "dinov2_vitb14_reg"`; the base constructor adds the suffix `_lc` and calls:

```python
torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg_lc")
```

The variant is a ViT-Base with `14×14` patches and *register tokens*. With the usual `224×224` input format, patch embedding produces a `16×16` grid, i.e. 256 visual tokens, to which the backbone adds the special tokens. Attention therefore works globally on the token sequence rather than within a local convolutional receptive field. Register tokens are part of the upstream backbone and are not managed individually by the project code.

Immediately after loading, the repository replaces the upstream head:

```python
self.model.linear_head = nn.Linear(
    self.model.linear_head.weight.shape[1], num_classes
)
```

The number of input features is read from the loaded head rather than hard-coded. This makes the adaptation robust to changes in the embedding size of the selected hub model, but assumes that the model exposes the `linear_head` attribute.

The graph used by `forward` is therefore:

```text
normalized image
  → DINOv2 patch embedding and transformer
  → global representation produced by the upstream wrapper
  → new linear_head [feature_dim → num_classes]
  → logits for each ingredient
```

The class delegates forwarding entirely to `self.model(x)`: it does not extract patches, CLS tokens, or register tokens explicitly. Changes in the DINOv2 upstream wrapper contract can therefore directly impact integration.

## Pretraining, freezing and fine-tuning modes

### What `pretrained` really controls

`DinoV2B14` accepts and serializes the `pretrained` parameter, but the value does not affect the call to `torch.hub.load`. The code always loads the hub model identified by `dinov2_vitb14_reg_lc`; `pretrained=False` does not construct a random weight ViT. It should therefore be interpreted as configuration metadata, not as a functional switch.

### Linear probing (default)

With `freeze_backbone=True` — `None` is also converted to `True` — `freeze_backbone()` visits `self.model.backbone.named_parameters()` and sets `requires_grad=False`. The new `linear_head`, which is external to `model.backbone`, remains trainable. This is a *linear probing* setup: during the backward pass the gradients do not update the backbone, but the optimizer is still built with `self.model.parameters()` and also includes the frozen parameters; PyTorch ignores them because they don't receive gradient.

It is an appropriate choice when the dataset is small or you want to isolate the quality of DINOv2 features, but it reduces the ability to adapt generic representations to the semantics of the ingredients, which are often visually subtle or partially occluded.

### Full fine-tuning

`unfreeze_backbone()` re-enables `requires_grad=True` on the backbone parameters. There is no callback or automatic policy that invokes it after N epochs: the transition from linear probing to fine-tuning must be explicitly orchestrated by the configuration/training code.

The method does not rebuild the optimizer. If it is called after Lightning has already created the optimizer, the parameters are already present in its param groups and can start updating; However, it remains the responsibility of the experiment to choose the learning rate, weight decay and scheduler suitable for the regime change.

### Layer-wise pretraining not supported

The constructor preserves `lp_phase`, but DINOv2 does not implement `_lp_init_layers`, `_lp_step_phase`, or the other LP protocol hooks of `BaseModel`. The argument must remain at `-1`: an active value does not construct a progressive version of the transformer and using `lp_phase_step()` would lead to the basic behavior not being implemented. It should not be confused with freezing the backbone.

## Preprocessing: declared pipeline and actually executed pipeline

### Transformations declared by the model

Without overriding, the `transform_aug` property returns `transform_aug_dino`:

1. `RandomResizedCrop(input_shape)` with `scale=(0.2, 1.0)`, `ratio=(0.75, 1.3333)` and bicubic interpolation;
2. horizontal flip with probability 0.5;
3. conversion to `float32` and scaling to `[0, 1]`;
4. normalization with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

`transform_plain` instead returns short side resize to 256 (keeping the ratio), center crop to `input_shape`, conversion/scaling and the same normalization. For the default value `input_shape=224`, the resolution is compatible with patch size 14 (`224 / 14 = 16`).

With custom `trns_aug` the model uses `transform_core_dino`: train uses `RandomResizedCrop`, validation uses `Resize(256) → CenterCrop`; then inserts custom augmentations before conversion and DINO normalization. If you pass `trns_bld_aug` and/or `trns_bld_plain` instead, they directly override the default builders.

### Normalization added by DataModule: current state

It is important to distinguish the model builder from the transformation consumed by the dataset. In `BaseDataModule.prepare_data()`, `_init_transform()` treats a list of transformations as input to `transformations_wrapper()`, which always adds:

```text
ToImage → supplied transforms → ToDtype(float32, scale=True)
        → Normalize(mean=train_images_stats, std=train_images_stats)
```

Since the default DINO builders return a **list** that already contains `ToDtype` and `Normalize(DINO_MEAN, DINO_STD)`, the actual pipeline applies two consecutive normalizations: first the DINO statistics, then the statistics calculated on the training set. The same happens in both train and validation/test/predict.

This composition does not coincide with standard DINO preprocessing and alters the distribution expected from the pre-trained backbone. This is a behavior of the current implementation, not a recommendation for new runs. A custom builder that directly returns a `v2.Transform` (for example a `v2.Compose`) avoids wrapping the DataModule and can therefore explicitly define only one normalization. In any case `images_stats_path` remains required by `prepare_data()` even when the transform already handles the normalization.

## Batch size, accumulation and memory

`_BaseDinoV2.MAX_ALLOWED_BATCH_SIZE = 32`. `BaseLGNM` uses this limit to calculate real batch and gradient accumulation before building the DataModule and Trainer:

```text
target_batch_size ≤ 32  → real batch = target, accumulo = 1
target_batch_size > 32  → accumulo = ceil(target / 32)
                         real batch = ceil(target / accumulo)
```

For example, a configuration with target batch 128 runs as micro-batch 32 and `accumulate_grad_batches=4`. For non-divisible values, the `real_batch × accumulation` product may be slightly larger than the target; the comment in the code indicates exact divisibility, but the implementation uses upward rounding.

The limit is an application convention, not a guarantee of absence of OOM: it depends on resolution, precision, optimizer state, head size, number of workers and GPU. The fast/Optuna trainers set `16-mixed` precision, which can reduce memory usage, while the basic trainer uses its own precision setting.

## Checkpoint, setup and resume

`to_config()` adds the `pretrained` and `freeze_backbone` flags to `BaseModel.to_config()`. The rebuild uses `load_from_config()`, then runs `torch.hub.load` again before loading the checkpoint state dict. To resume a run, the hub repository must therefore be reachable or already present in the local cache of torch.hub; project checkpointing does not eliminate this dependency when building the model.

The configuration also saves transformation callables when they are passed as overrides. These are Python objects of the process, not a portable JSON definition: reliable recovery requires that the corresponding functions and imports remain available.

## Interpretability and limitations of hooks

`classifier_target_layer` returns `self.model.linear_head`, so it correctly represents the final mapping to the ingredients. For Grad-CAM and factorization, `conv_target_layer` returns `self.model.backbone.blocks[-1].norm1`: it is the final pre-attention normalization, whose activations directly affect the last block and the head. DINOv2 does not have a final convolutional layer; `gradcam_reshape_transform` removes CLS and the four register tokens, then converts the remaining patch tokens from `[B, 256, 768]` to `[B, 768, 16, 16]` for standard `224×224` input.

This reshape is essential because Grad-CAM and Deep Feature Factorization work on `[B, C, H, W]` spatial maps, while ViT produces sequences. The Grad-CAM wrapper also makes the input differentiable: without this step, a frozen backbone does not produce gradients at the target layer and Grad-CAM cannot compute map weights.

When interpreting predictions by ingredient, the resulting heatmap also depends on how the visualization utility retranslates tokens and features; choosing `backbone.norm` alone does not guarantee a semantically equivalent spatial map to Grad-CAM on a CNN.

## Checklist for a new run

- Use `DinoV2B14` and supply the number of classes from the run encoder, never a manually set number.
- Check availability/caching of `facebookresearch/dinov2` in the execution environment before a long job.
- Consciously choose between linear probing (`freeze_backbone=True`) and fine-tuning (`False` or explicitly orchestrated `unfreeze_backbone()`).
- Keep `lp_phase=-1` for DINOv2.
- Check the final pipeline of the DataModule: with the default builders today the double normalization described above is present.
- For `224×224` input, keep crop and resize multiples of 14; Different sizes require verification of the hub model contract and number of tokens.
- Treat `max_allowed_batch_size=32` as a starting point and check the memory on the actual device and accuracy chosen.

## References in the repository

- `src/models/dinov2.py`: wrapper, head, freezing, visualization hooks and batch limit.
- `src/data_processing/transformations.py`: `transform_*_dino` builder and normalization statistics.
- `src/data_processing/common.py`: wrapping of transformation lists and dependency on `train_images_stats.csv`.
- `src/lightning/lgn_models.py`: Actual batch, accumulation and optimizer construction.
- `src/training/commons.py`: Propagate transformations and batches from the Lightning model to the DataModule.

## Primary sources

- Oquab et al., [*DINOv2: Learning Robust Visual Features without Supervision*](https://arxiv.org/abs/2304.07193): ViT-B architecture, DINO+iBOT pretraining, KoLeo, LVD-142M and distillation.
- Darcet et al., [*Vision Transformers Need Registers*](https://arxiv.org/abs/2309.16588): motivation and functioning of register tokens.
- [Official repository `facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2): published model names and loading with PyTorch Hub; the [linear head implementation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/classifiers.py) defines the four registers and the aggregation of the last four blocks used by `_lc`.
