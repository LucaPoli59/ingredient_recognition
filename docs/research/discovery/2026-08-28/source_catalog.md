# Primary-source catalog — 4A.1

**Created:** 2026-08-28
**Last updated:** 2026-08-28
**Cutoff:** 2026-08-28
**Use:** Evidence for broad candidate intake; not a cross-dataset ranking

## Search protocol

Sources were retained when they are an original paper, publisher proceedings
record, official research page, official repository, maintained library, or
official checkpoint/model card. The catalog records what a source establishes
and where transfer stops. Search snippets, benchmark aggregators, and secondary
surveys were used only to locate primary material.

The project has one image-only, fixed-vocabulary target. A source that uses
ingredient names as prompts, recipe text, segmentation masks, retrieval labels,
or a different food taxonomy is therefore evidence for a mechanism or an
adjacent task, not direct proof of performance on Yummly.

## Existing baseline anchors (papers retained, not selection candidates)

| Baseline | Source | Evidence retained | Boundary |
| --- | --- | --- | --- |
| B1 — ResNet | He et al., [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), CVPR 2016 | Residual supervised CNN and the historical continuity point for the repository. | ImageNet single-label evidence does not establish recipe-ingredient recognition; local ResNet is already used. |
| B1 — implementation | [Torchvision model documentation](https://docs.pytorch.org/vision/main/models) | Maintained constructors, weight metadata, and transform contract. | Exact local transform and memory still belong to the implementation gate. |
| B2 — DINOv2 | [Official DINOv2 repository](https://github.com/facebookresearch/dinov2) and [model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) | Label-free pretrained global/patch features, public S/B/L/g variants, and downstream use guidance. | Generic visual features are not proof of any ingredient's observability; the model is already implemented and used locally. |

## New candidate evidence

### C1 — EfficientNetV2 efficient convolutional network

| Source | Evidence type | Claim used in 4A.1 | Transfer boundary |
| --- | --- | --- | --- |
| Tan and Le, [EfficientNetV2: Smaller Models and Faster Training](https://proceedings.mlr.press/v139/tan21a.html), ICML 2021 | M — original architecture/training | Training-aware architecture search, Fused-MBConv, and progressive learning target parameter/compute efficiency. | ImageNet/CIFAR/Cars/Flowers evidence is not recipe-level multi-label evidence. |
| [Torchvision model documentation](https://docs.pytorch.org/vision/master/models.html) | Official implementation/library | EfficientNetV2-S/M/L constructors and pretrained weights offer a maintained PyTorch access path. | Weight terms, transform choices, and local memory must be frozen before implementation. |
| Ismail and Yuan, [Food Ingredients Recognition through Multi-label Learning](https://arxiv.org/abs/2210.14147), 2022 | D/A — direct food multi-label | A food ingredient study evaluates EfficientNet-family encoders with pooled and attention decoders. | Nutrition5K labels, images, and split differ; it supports mechanism relevance, not score transfer. |

### C2 — Hierarchical/windowed transformer

| Source | Evidence type | Claim used in 4A.1 | Transfer boundary |
| --- | --- | --- | --- |
| Liu et al., [Swin Transformer V2: Scaling Up Capacity and Resolution](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html), CVPR 2022 | M — original architecture | Hierarchical shifted windows, scaled cosine attention, and multi-scale representations provide a distinct alternative to pooled CNN features. | Published benchmarks are not recipe-level multi-label evidence. |
| Microsoft, [Swin Transformer repository](https://github.com/microsoft/Swin-Transformer) | Official implementation | Code and checkpoint paths are available for a traceable implementation route. | Repository/checkpoint version, licence, and exact memory at local resolution must be checked. |
| Wu et al., [FoodSeg103 / ReLeM](https://arxiv.org/abs/2105.05409), 2021 | A — food vision | Ingredient-level food imagery demonstrates that local and multi-scale visual structure is relevant in an adjacent task. | Segmentation masks and 103 visible categories are not recipe-level labels. |

### C3 — Vision-language visual encoder

| Source | Evidence type | Claim used in 4A.1 | Transfer boundary |
| --- | --- | --- | --- |
| Radford et al., [CLIP](https://openai.com/index/clip/), 2021 | M — original pretraining | Web image-text supervision can transfer visual concepts and offers a continuity fallback for VLM encoders. | Text-derived concepts and possible source overlap change the claim relative to image-only learning. |
| Zhai et al., [SigLIP](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV2023_paper.pdf), ICCV 2023 | M — original objective | Pairwise sigmoid image-text loss is a distinct VLM pretraining mechanism. | Generic image-text benchmarks do not establish ingredient-set performance. |
| Tschannen et al., [SigLIP 2](https://arxiv.org/abs/2502.14786), 2025 | M — updated pretraining | Captioning, self-distillation, masked prediction, curation, and dense/native-aspect variants motivate a contemporary candidate. | Richer semantic priors, recent checkpoint access, and resource behavior require explicit audit. |
| Google, [SigLIP2 model card](https://huggingface.co/google/siglip2-base-patch16-224) | Official checkpoint/model card | A maintained checkpoint path and downstream Transformers interface are available for investigation. | Exact licence, dependency versions, checkpoint provenance, and 8 GB behavior must be verified locally. |

### C4 — Structured multi-label query/set head

| Source | Evidence type | Claim used in 4A.1 | Transfer boundary |
| --- | --- | --- | --- |
| Liu et al., [Query2Label: A Simple Transformer Way to Multi-Label Classification](https://arxiv.org/abs/2107.10834), 2021 | M — multi-label mechanism | Transformer decoders use label queries and cross-attention to pool class-related features from a visual feature map. | Benchmarks are generic and the label-query mechanism must be separated from word/text priors. |
| Query2Label, [official repository](https://github.com/SlongLiu/query2labels) | Official implementation path | Supplies a concrete reference implementation to inspect and adapt. | Current dependency/API compatibility and memory are not yet verified. |
| Ridnik et al., [ML-Decoder](https://openaccess.thecvf.com/content/WACV2023/html/Ridnik_ML-Decoder_Scalable_and_Versatile_Classification_Head_WACV_2023_paper.html), WACV 2023 | M — efficient multi-label head | Learned queries and group decoding provide an efficient alternative to a full decoder for large label sets. | MS-COCO/other generic multi-label scores are not Yummly evidence. |
| Alibaba-MIIL, [ML-Decoder repository](https://github.com/Alibaba-MIIL/ML_Decoder) | Official implementation path | Provides code and examples for a head implementation. | Head integration, licence, and output-contract checks remain open. |
| Salvador et al., [Inverse Cooking](https://openaccess.thecvf.com/content_CVPR_2019/papers/Salvador_Inverse_Cooking_Recipe_Generation_From_Food_Images_CVPR_2019_paper.pdf), CVPR 2019 | D — food set-decoding precedent | Ingredient prediction can be formulated as an unordered set with dependencies in a food-image/recipe system. | Recipe generation, instructions, and autoregressive decoding are outside the canonical classifier. |

### C5 — MaxViT hybrid multi-axis attention network

| Source | Evidence type | Claim used in 4A.1 | Transfer boundary |
| --- | --- | --- | --- |
| Tu et al., [MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697), ECCV 2022 | M — original architecture | Hybrid convolution, local block attention, and global grid attention provide a distinct local-plus-global mechanism. | ImageNet classification/detection/segmentation results do not establish recipe-level ingredient inference. |
| Google Research, [official MaxViT repository](https://github.com/google-research/maxvit) | Official implementation/checkpoints | Documents Tiny/Small checkpoints and the architecture's global grid-attention path. | Repository is archived; TensorFlow route, checkpoint terms, and maintenance need review. |
| PyTorch, [Torchvision MaxViT implementation](https://docs.pytorch.org/vision/main/_modules/torchvision/models/maxvit.html) | Maintained implementation/library | Provides a `maxvit_t` constructor, public weights, and documented parameter/transform metadata. | The released weight contract assumes square 224-pixel inputs; aspect-preserving adaptation and local memory are open. |

## Adjacent food-domain leads

| Source | Why retained as context | Why not a formal new family |
| --- | --- | --- |
| Min et al., [Food2K](https://arxiv.org/abs/2103.16107) | Large food-image representation-transfer setting. | Dish categories are not ingredient presence; checkpoint/data terms need an explicit audit. |
| Marin et al., [Recipe1M+ project](https://pic2recipe.csail.mit.edu/) and [paper](https://arxiv.org/abs/1810.06553) | Public image-recipe corpus and cross-modal representation evidence. | Retrieval and recipe-text supervision introduce a different target and possible source overlap. |
| Wu et al., [FoodInsSeg repository](https://github.com/jamesjg/FoodInsSeg) | 103-category ingredient-instance masks show local food cues can be annotated. | Segmentation target and pixel visibility differ from recipe-level supervision. |
| [Zero-Shot Ingredient Recognition by Multi-Relational GCN](https://ojs.aaai.org/index.php/AAAI/article/view/6626) | Graph label dependencies address ingredient variation and zero-shot structure. | Different food datasets and zero-shot objective; graph gains need non-visual controls here. |

## Source-quality and transfer notes

- **Primary-source confidence:** high for original architecture/objective and
  official repositories; medium for transfer to this project because no source
  uses the repaired Yummly contract.
- **Access is not reproducibility:** links identify a path to inspect; Phase 5
  must freeze versions, transforms, checkpoint hashes, and licences.
- **Text is a confound:** using label names at downstream time is a separate
  semantic-prior experiment, even when the encoder itself is frozen.
- **Food-domain pretraining is not automatically better:** it may be closer to
  the target while also increasing overlap, vocabulary, and provenance risks.
- **No score transfer:** source metrics remain in the source task and are not
  converted into candidate ranks.
