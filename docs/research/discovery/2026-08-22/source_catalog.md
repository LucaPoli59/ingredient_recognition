# Primary-source catalog for reference-selector candidates

**Created:** 2026-08-22
**Last updated:** 2026-08-22

## Search protocol

**Cutoff:** 2026-08-22.

Sources were retained when they were an original paper, publisher proceedings
record, official research page, or official implementation/checkpoint
repository. The review used sources to establish training objectives,
available model families, output mechanisms, and transfer boundaries. It did
not compare headline scores across datasets.

The existence of a paper or official repository does not establish compatibility
with this project. R0.2 must recheck checkpoint access, exact terms, dependency
versions, memory, output shape, and preprocessing before declaring a candidate
eligible.

## Direct ingredient and food-domain evidence

| Source | Evidence used here | Transfer boundary |
| --- | --- | --- |
| Bolaños et al., [Food Ingredients Recognition through Multi-label Learning](https://arxiv.org/abs/1707.08816), 2017 | Direct image-to-ingredient multi-label formulation and acknowledgement of hidden/transformed ingredients | Different datasets, vocabulary simplification, split, and evaluation |
| Min et al., [Large Scale Visual Food Recognition / Food2K](https://arxiv.org/abs/2103.16107), TPAMI 2023 | Food2K scale and food-domain representation-transfer claim | Dish-category supervision is not ingredient presence; checkpoint and data terms require verification |
| Marin et al., [Recipe1M+ official project](https://pic2recipe.csail.mit.edu/), TPAMI 2019 | Public image-recipe corpus, code, and pretrained cross-modal models | Retrieval target and recipe-text supervision differ; possible source overlap must be audited |
| Shukor et al., [VLPCook official repository](https://github.com/mshukor/VLPCook), 2022/2024 | Structured recipe-language and image pretraining; released code/checkpoint path | Retrieval-centered legacy stack; ingredient semantics enter pretraining directly |
| Wu et al., [FoodSeg103 and ReLeM](https://arxiv.org/abs/2105.05409), 2021 | Ingredient-level food segmentation and Recipe1M+-based multimodal pretraining | Pixel-visible target and taxonomy differ from recipe-level multilabel classification |
| Wu et al., [OVFoodSeg](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_OVFoodSeg_Elevating_Open-Vocabulary_Food_Image_Segmentation_via_Image-Informed_Textual_Representation_CVPR_2024_paper.html), CVPR 2024 | Food-specific open-vocabulary image-text localization evidence | Prompted segmentation is not the canonical selector output |
| Zhu et al., [Food-R1](https://arxiv.org/abs/2606.04986), 2026 preprint; [released checkpoint](https://huggingface.co/zy12123/Food-R1) | Emerging food-specific generative VLM and released multi-task checkpoint | Generative scores and large checkpoint do not satisfy the current per-label trajectory/8 GB contract |

## Supervised and efficient visual backbones

| Source | Evidence used here | Transfer boundary |
| --- | --- | --- |
| He et al., [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), CVPR 2016 | Residual supervised-CNN continuity family | Single-label ImageNet architecture evidence, not ingredient learnability |
| Huang et al., [Densely Connected Convolutional Networks](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html), CVPR 2017 | DenseNet continuity family | Same boundary as other general supervised backbones |
| Tan and Le, [EfficientNetV2](https://proceedings.mlr.press/v139/tan21a.html), ICML 2021 | Efficient supervised convolutional family and progressive-training design | Published training recipe need not match local fine-tuning |
| Woo et al., [ConvNeXt V2](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html), CVPR 2023; [official repository](https://github.com/facebookresearch/ConvNeXt-V2) | Compact-to-large ConvNet family and FCMAE pretraining | Official repository is archived; maintained integration and exact checkpoint type require review |
| Liu et al., [Swin Transformer V2](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html), CVPR 2022; [official repository](https://github.com/microsoft/Swin-Transformer) | Hierarchical/windowed transformer and compact variants | Architecture and pretraining must be separated |

## Visual self-supervision and masked representation learning

| Source | Evidence used here | Transfer boundary |
| --- | --- | --- |
| Oquab et al., [DINOv2 official repository](https://github.com/facebookresearch/dinov2), 2023 | 142M-image label-free pretraining; global/patch features; S/B/L/g families and register variants | Exact local wrapper, transforms, freezing, and licence/checkpoint must be frozen |
| Siméoni et al., [DINOv3 official research page](https://ai.meta.com/research/dinov3/), 2025; [official repository](https://github.com/facebookresearch/dinov3) | 1.7B-image visual SSL, dense features, compact ViT and distilled ConvNeXt releases | Access request, dedicated licence, recent stack, and compute require a gate |
| He et al., [Masked Autoencoders Are Scalable Vision Learners](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html), CVPR 2022; [official repository](https://github.com/facebookresearch/mae) | Masked-pixel reconstruction as an alternative P1 objective | Standard released variants can be costly; objective is not ingredient-specific |
| Assran et al., [I-JEPA official repository and paper](https://github.com/facebookresearch/ijepa), CVPR 2023 | Latent target prediction without handcrafted multi-view augmentations | Official checkpoints are centered on very large ViT-H/g variants |

## Generic vision-language pretraining

| Source | Evidence used here | Transfer boundary |
| --- | --- | --- |
| Radford et al., [CLIP](https://openai.com/index/clip/), 2021; [paper](https://arxiv.org/abs/2103.00020) | Web image-text contrastive representation and zero-shot name scoring | Web-text concepts, prompt choices, and unknown source overlap enter the measurement |
| Zhai et al., [SigLIP](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language-Image_Pre-Training_ICCV_2023_paper.html), ICCV 2023 | Pairwise sigmoid image-text objective and released encoder family | Same semantic-prior boundary as CLIP |
| Tschannen et al., [SigLIP 2](https://arxiv.org/abs/2502.14786), 2025; [official big_vision page](https://google-research.github.io/big_vision/big_vision/configs/proj/image_text/README_siglip2.html) | Captioning, self-distillation, masked prediction, curation, dense/native-aspect variants, and multiple scales | Mixed objectives make the prior richer, not more neutral; downstream text use must be separated |

## Multi-label and structured heads

| Source | Evidence used here | Transfer boundary |
| --- | --- | --- |
| Ridnik et al., [ML-Decoder](https://openaccess.thecvf.com/content/WACV2023/html/Ridnik_ML-Decoder_Scalable_and_Versatile_Classification_Head_WACV_2023_paper.html), WACV 2023; [official repository](https://github.com/Alibaba-MIIL/ML_Decoder) | Efficient learned queries over spatial features and group decoding | General multi-label benchmarks; learned and word queries must be separated |
| Liu et al., [Query2Label](https://arxiv.org/abs/2107.10834), 2021; [official repository](https://github.com/SlongLiu/query2labels) | Full transformer-decoder label-query mechanism | Higher head cost and the same general-benchmark boundary |
| Lanchantin et al., [C-Tran](https://openaccess.thecvf.com/content/CVPR2021/html/Lanchantin_General_Multi-Label_Image_Classification_With_Transformers_CVPR_2021_paper.html), CVPR 2021; [official repository](https://github.com/QData/C-Tran) | Joint visual/label-state transformer and partial-label behavior | Label dependencies can substitute for direct visual evidence |
| Chen et al., [ML-GCN](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Multi-Label_Image_Recognition_With_Graph_Convolutional_Networks_CVPR_2019_paper.html), CVPR 2019 | Graph-based label relationship modeling | Co-occurrence graph must use training data only and needs non-visual controls |
| Salvador et al., [Inverse Cooking](https://openaccess.thecvf.com/content_CVPR_2019/html/Salvador_Inverse_Cooking_Recipe_Generation_From_Food_Images_CVPR_2019_paper.html), CVPR 2019 | Unordered ingredient-set decoding and dependency modeling | Recipe/instruction generation and decoding policy are outside the primary selector |

## Sources not used as ranking evidence

No leaderboard aggregator, general survey, commercial model summary, or
secondary tutorial was used to rank candidates. Sources with segmentation,
retrieval, generation, zero-shot, or single-label targets were retained only
when their transfer boundary was explicit.
