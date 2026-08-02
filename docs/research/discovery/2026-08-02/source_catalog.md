# Primary-Source Catalog

Created: 2026-08-02  
Updated: 2026-08-02

## 1. Search protocol

Cutoff date: **2026-08-02**.

The discovery used publisher proceedings, author papers, official repositories, and official dataset/ontology documentation. Surveys, commercial summaries, and leaderboard aggregators were not used as evidence. Search terms combined the local task with adjacent methods, including:

- food ingredient recognition, recipe ingredient prediction, inverse cooking, recipe retrieval;
- food segmentation, open-vocabulary food segmentation, dietary VLM evaluation;
- multi-label image classification, class queries, label graphs, hierarchy;
- long-tailed multi-label loss, missing/partial/noisy labels, calibration;
- recipe ingredient parsing, food ontology, semantic deduplication, grouped iterative stratification;
- native aspect vision transformers, food augmentation, sample mixing;
- multi-label conformal prediction, interpretability, shortcut learning, dataset documentation.

Selection rules:

1. prefer peer-reviewed primary research;
2. use official dataset/project pages for resource facts;
3. include preprints only for current directions not yet represented by stable proceedings, and mark them;
4. retain sources whose supervision, output, or evaluation differs from the project only when the transfer boundary is explicit;
5. do not interpret incompatible benchmark scores as a ranking for this project.

This is a broad state-of-the-art map, not a formal systematic review or meta-analysis. The literature is too heterogeneous for an aggregate effect size.

## 2. Dataset and project context

| Source | Status | Relevance and boundary |
|---|---|---|
| Min et al., [You Are What You Eat: Exploring Rich Recipe Information for Cross-Region Food Analysis](https://openreview.net/pdf?id=F9oSOeGGkwP), 2018 | peer-reviewed journal paper | Primary description of Yummly-66K and its multi-attribute cross-region task; not the regenerated local benchmark. |
| Authors’ [official Yummly-66K repository](https://github.com/minweiqing/You-Are-What-You-Eat-Exploring-Rich-Recipe-Information-for-Cross-Region-Food-Analysis) | official code/data page | Provenance and dataset structure. |

## 3. Food ingredient recognition and recipe understanding

| Source | Status | Relevance and boundary |
|---|---|---|
| Bolaños et al., [Food Ingredients Recognition through Multi-label Learning](https://arxiv.org/abs/1707.08816), 2017 | peer-reviewed conference paper / author manuscript | Closest foundational multi-label formulation; simplified vocabularies and different datasets/splits. |
| Salvador et al., [Inverse Cooking: Recipe Generation from Food Images](https://openaccess.thecvf.com/content_CVPR_2019/html/Salvador_Inverse_Cooking_Recipe_Generation_From_Food_Images_CVPR_2019_paper.html), CVPR 2019 | peer-reviewed | Ingredient set decoder and dependencies; instruction generation is out of scope. |
| Salvador et al., [Learning Cross-Modal Embeddings for Cooking Recipes and Food Images](https://openaccess.thecvf.com/content_cvpr_2017/html/Salvador_Learning_Cross-Modal_Embeddings_CVPR_2017_paper.html), CVPR 2017 | peer-reviewed | Image–recipe representation learning; retrieval target differs. |
| Marin et al., [Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images](https://arxiv.org/abs/1810.06553), 2019 | peer-reviewed journal paper / author manuscript | Large-scale multimodal transfer context; not approved active data. |
| Ismail and Yuan, [Food Ingredients Recognition through Multi-label Learning](https://arxiv.org/abs/2210.14147), 2022 | preprint | Encoder/pooled-versus-attention decoder evidence using Nutrition5K; controlled-domain differences. |
| Chen et al., [Zero-Shot Ingredient Recognition by Multi-Relational Graph Convolutional Network](https://ojs.aaai.org/index.php/AAAI/article/view/6626), AAAI 2020 | peer-reviewed | Ingredient hierarchy, attributes, and co-occurrence; zero-shot setup differs. |
| Chhikara et al., [FIRE: Food Image to REcipe Generation](https://openaccess.thecvf.com/content/WACV2024/html/Chhikara_FIRE_Food_Image_to_REcipe_Generation_WACV_2024_paper.html), WACV 2024 | peer-reviewed | BLIP title, ViT ingredient decoder, and T5 instruction pipeline; generation remains out of scope. |
| Liu et al., [Retrieval Augmented Recipe Generation](https://openaccess.thecvf.com/content/WACV2025/html/Liu_Retrieval_Augmented_Recipe_Generation_WACV_2025_paper.html), WACV 2025 | peer-reviewed | Future retrieval/generation direction; not a primary closed-vocabulary classifier. |
| Wang et al., [Towards Unbiased Cross-Modal Representation Learning for Food Image-to-Recipe Retrieval](https://arxiv.org/abs/2511.15201), revised 2026 | preprint | Current class-query ingredient module and explicit visibility-bias analysis; retrieval target and causal claims require separate validation. |

## 4. Food segmentation and dietary understanding

| Source | Status | Relevance and boundary |
|---|---|---|
| Wu et al., [FoodSeg103: A New Benchmark for Food Image Segmentation](https://arxiv.org/abs/2105.05409), 2021 | peer-reviewed / author manuscript | Localized food evidence and segmentation difficulty; visible pixels only. |
| [UECFoodPixComplete](https://mm.cs.uec.ac.jp/uecfoodpix/) | official dataset page | Food segmentation resource; taxonomy and pixel target differ. |
| Wu et al., [OVFoodSeg: Elevating Open-Vocabulary Food Image Segmentation via Image-Informed Textual Representation](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_OVFoodSeg_Elevating_Open-Vocabulary_Food_Image_Segmentation_via_Image-Informed_Textual_Representation_CVPR_2024_paper.html), CVPR 2024 | peer-reviewed | Food-specific text-conditioned dense features; open-vocabulary segmentation is outside headline scope. |
| Thames et al., [Nutrition5K: Towards Automatic Nutritional Understanding of Generic Food](https://openaccess.thecvf.com/content/CVPR2021/html/Thames_Nutrition5k_Towards_Automatic_Nutritional_Understanding_of_Generic_Food_CVPR_2021_paper.html), CVPR 2021 | peer-reviewed | Fine-grained food/nutrition and controlled capture; quantities and multi-view data absent locally. |
| Romero-Tapiador et al., [Are Vision-Language Models Ready for Dietary Assessment?](https://openaccess.thecvf.com/content/CVPR2025W/MTF/html/Romero-Tapiador_Are_Vision-Language_Models_Ready_for_Dietary_Assessment_Exploring_the_Next_CVPRW_2025_paper.html), CVPR Workshops 2025 | peer-reviewed workshop | Current empirical limits of VLM food understanding; dietary target differs. |
| Wang et al., [A Comparative Study of Vision–Language Models for Food Ingredient Recognition and Nutrient Estimation](https://doi.org/10.1016/j.crfs.2026.101405), Current Research in Food Science 2026 | peer-reviewed journal paper | Most current direct VLM comparison found; Nutrition5K, API models, and multi-view prompting differ from the local single-image benchmark. |
| Chi et al., [Food Image Segmentation with LLM-Derived Ingredient Labels and Multimodal Fusion](https://arxiv.org/abs/2607.25820), submitted 2026-07-28 | preprint | Very recent semantic feature/query injection direction; segmentation and automatically inferred labels are transfer risks. |

## 5. Multi-label architectures and semantic transfer

| Source | Status | Relevance and boundary |
|---|---|---|
| Chen et al., [Multi-Label Image Recognition with Graph Convolutional Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Multi-Label_Image_Recognition_With_Graph_Convolutional_Networks_CVPR_2019_paper.html), CVPR 2019 | peer-reviewed | Explicit label co-occurrence graph; susceptible to prior shortcuts. |
| Lanchantin et al., [General Multi-Label Image Classification with Transformers](https://openaccess.thecvf.com/content/CVPR2021/html/Lanchantin_General_Multi-Label_Image_Classification_With_Transformers_CVPR_2021_paper.html), CVPR 2021 | peer-reviewed | Label-state transformers and partial observations. |
| Liu et al., [Query2Label: A Simple Transformer Way to Multi-Label Classification](https://arxiv.org/abs/2107.10834), 2021 | peer-reviewed / author manuscript | Label queries over spatial features. |
| Ridnik et al., [ML-Decoder: Scalable and Versatile Classification Head](https://openaccess.thecvf.com/content/WACV2023/html/Ridnik_ML-Decoder_Scalable_and_Versatile_Classification_Head_WACV_2023_paper.html), WACV 2023 | peer-reviewed | Efficient class-query head; preferred first structured-head candidate. |
| Sun et al., [DualCoOp: Fast Adaptation to Multi-Label Recognition with Limited Annotations](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c5169260ef32d1bd3597c14d8c89b034-Abstract-Conference.html), NeurIPS 2022 | peer-reviewed | Positive/negative prompt adaptation for multi-label CLIP. |
| Wang et al., [MuMIC — Multimodal Embedding for Multi-Label Image Classification with Tempered Sigmoid](https://ojs.aaai.org/index.php/AAAI/article/view/26850), AAAI 2023 | peer-reviewed | Deployed vision-language transfer and tempered sigmoid loss; its travel-image domain differs. |
| Wehrmann et al., [Hierarchical Multi-Label Classification Networks](https://proceedings.mlr.press/v80/wehrmann18a.html), ICML 2018 | peer-reviewed | Joint local/global hierarchy modeling; headline metric must remain exact-label. |

## 6. Visual representations and adaptation

| Source | Status | Relevance and boundary |
|---|---|---|
| Oquab et al., [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193), 2023 | peer-reviewed / author manuscript | Existing self-supervised backbone and continuity reference. |
| Siméoni et al., [DINOv3](https://arxiv.org/abs/2508.10104), 2025 | preprint | Current dense SSL direction; gate on access, terms, dependencies, and 8 GB feasibility. |
| Meta AI, [official DINOv3 repository](https://github.com/facebookresearch/dinov3) | official code/checkpoint page | Model variants and loading path; operational facts should be pinned to a revision. |
| Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), 2021 | peer-reviewed / author manuscript | CLIP foundation for image–text transfer. |
| Zhai et al., [Sigmoid Loss for Language Image Pre-Training](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html), ICCV 2023 | peer-reviewed | Pairwise sigmoid image–text pretraining. |
| Tschannen et al., [SigLIP 2](https://arxiv.org/abs/2502.14786), 2025 | preprint | Current multilingual/localization-aware VLM family; evaluate as a representation hypothesis. |
| Cherti et al., [Reproducible Scaling Laws for Contrastive Language-Image Learning](https://arxiv.org/abs/2212.07143), CVPR 2023 | peer-reviewed / author manuscript | OpenCLIP scaling and reproducible pretrained alternatives. |
| Woo et al., [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html), CVPR 2023 | peer-reviewed | Modern compact convolutional control. |
| Dehghani et al., [Patch n’ Pack: NaViT, a Vision Transformer for Any Aspect Ratio and Resolution](https://proceedings.neurips.cc/paper_files/paper/2023/hash/06ea400b9b7cfce6428ec27a371632eb-Abstract-Conference.html), NeurIPS 2023 | peer-reviewed | Evidence for native aspect/variable resolution; not required for first preprocessing test. |
| Beyer et al., [FlexiViT: One Model for All Patch Sizes](https://openaccess.thecvf.com/content/CVPR2023/html/Beyer_FlexiViT_One_Model_for_All_Patch_Sizes_CVPR_2023_paper.html), CVPR 2023 | peer-reviewed | Flexible patch-scale representation. |
| Jia et al., [Visual Prompt Tuning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930696.pdf), ECCV 2022 | peer-reviewed | Parameter-efficient ViT adaptation. |
| Jie and Deng, [Fact-Tuning for Lightweight Adaptation on Vision Transformer](https://ojs.aaai.org/index.php/AAAI/article/view/25160), AAAI 2023 | peer-reviewed | Additional parameter-efficient adaptation evidence. |

## 7. Long-tail, weak-label, and calibration objectives

| Source | Status | Relevance and boundary |
|---|---|---|
| Ridnik et al., [Asymmetric Loss for Multi-Label Classification](https://arxiv.org/abs/2009.14119), 2021 | peer-reviewed / author manuscript | Strong sparse multi-label ranking objective; probability calibration risk. |
| Wu et al., [Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490154.pdf), ECCV 2020 | peer-reviewed | Handles co-occurrence-aware rebalancing and abundant negatives. |
| Kobayashi, [Two-Way Multi-Label Loss](https://openaccess.thecvf.com/content/CVPR2023/html/Kobayashi_Two-Way_Multi-Label_Loss_CVPR_2023_paper.html), CVPR 2023 | peer-reviewed | Joint hard-sample/hard-class objective. |
| Cui et al., [Class-Balanced Loss Based on Effective Number of Samples](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html), CVPR 2019 | peer-reviewed | Influential imbalance principle; developed primarily for single-label settings. |
| Duarte et al., [PLM: Partial Label Masking for Imbalanced Multi-Label Classification](https://openaccess.thecvf.com/content/CVPR2021W/LLID/html/Duarte_PLM_Partial_Label_Masking_for_Imbalanced_Multi-Label_Classification_CVPRW_2021_paper.html), CVPR Workshops 2021 | peer-reviewed workshop | Stochastic label masking to rebalance positive/negative ratios; not target repair. |
| Cole et al., [Multi-Label Learning from Single Positive Labels](https://arxiv.org/abs/2106.09708), 2021 | peer-reviewed / author manuscript | Missing-label evidence; local targets are not merely single-positive. |
| Yuan et al., [Positive Label Is All You Need for Multi-Label Classification](https://arxiv.org/abs/2306.16016), ICME 2024 | peer-reviewed / author manuscript | PU-MLC formulation; requires noise assumptions not established locally. |
| Zhang et al., [Learning in Imperfect Environment: Multi-Label Classification with Long-Tailed Distribution and Partial Labels](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Learning_in_Imperfect_Environment_Multi-Label_Classification_with_Long-Tailed_Distribution_and_ICCV_2023_paper.html), ICCV 2023 | peer-reviewed | Joint long-tail and partial-label method (COMC). |
| Ghiassi et al., [Trusted Loss Correction for Noisy Multi-Label Learning](https://proceedings.mlr.press/v189/ghiassi23a.html), ACML 2023 | peer-reviewed | Corruption-matrix correction using trusted data; requires a trusted subset/design. |
| Cheng et al., [Towards Calibrated Multi-Label Deep Neural Networks](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Towards_Calibrated_Multi-label_Deep_Neural_Networks_CVPR_2024_paper.html), CVPR 2024 | peer-reviewed | Strictly proper asymmetric objective and multi-label calibration evidence. |

## 8. Ingredient parsing and food knowledge

| Source | Status | Relevance and boundary |
|---|---|---|
| Wróblewska et al., [TASTEset — Recipe Dataset and Food Entities Recognition Benchmark](https://arxiv.org/abs/2204.07775), 2022 | dataset/preprint | Structured recipe entity parsing and domain annotations. |
| Goel et al., [Deep Learning Based Named Entity Recognition Models for Recipes](https://aclanthology.org/2024.lrec-main.406/), LREC-COLING 2024 | peer-reviewed | Domain NER evidence and limits of few-shot LLM parsing in that study. |
| Dooley et al., [FoodOn: A Harmonized Food Ontology to Increase Global Food Traceability, Quality Control and Data Integration](https://pmc.ncbi.nlm.nih.gov/articles/PMC6550238/), 2018 | peer-reviewed | External taxonomy and identifiers; local semantic decisions still required. |
| [FoodOn official site](https://foodon.org/) | official ontology page | Current ontology releases/documentation. |
| EFSA, [Data Standardisation and FoodEx2](https://www.efsa.europa.eu/en/data/data-standardisation) | official authority documentation | Food classification/coding reference, not direct recipe ontology. |
| USDA, [FoodData Central API Guide](https://fdc.nal.usda.gov/api-guide/) | official authority documentation | Food identifiers and metadata for optional crosswalks. |

## 9. Deduplication and split construction

| Source | Status | Relevance and boundary |
|---|---|---|
| Abbas et al., [SemDeDup: Data-Efficient Learning at Web-Scale Through Semantic Deduplication](https://arxiv.org/abs/2303.09540), 2023 | peer-reviewed / author manuscript | Embedding-based duplicate discovery; local auto-link thresholds require review. |
| Meta AI, [official SemDeDup repository](https://github.com/facebookresearch/SemDeDup) | official implementation | Reproducible implementation reference. |
| Ramos et al., [Data Leakage in Visual Datasets](https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Ramos_Data_Leakage_in_Visual_Datasets_ICCVW_2025_paper.html), ICCV Workshops 2025 | peer-reviewed workshop | Current leakage taxonomy and empirical motivation. |
| Barz and Denzler, [Do We Train on Test Data? Purging CIFAR of Near-Duplicates](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321059/), 2020 | peer-reviewed journal paper | Demonstrates evaluation distortion from near-duplicate contamination. |
| Szymański and Kajdanowicz, [A Network Perspective on Stratification of Multi-Label Data](https://proceedings.mlr.press/v74/szyma%C5%84ski17a.html), PMLR 2017 | peer-reviewed | Iterative stratification including label-pair evidence; must be adapted to groups. |

## 10. Image augmentation and robustness

| Source | Status | Relevance and boundary |
|---|---|---|
| Cubuk et al., [RandAugment: Practical Automated Data Augmentation with a Reduced Search Space](https://proceedings.neurips.cc/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html), NeurIPS 2020 | peer-reviewed | General augmentation policy; operations/ranges need food audit. |
| Müller and Hutter, [TrivialAugment](https://openaccess.thecvf.com/content/ICCV2021/html/Muller_TrivialAugment_Tuning-Free_Yet_State-of-the-Art_Data_Augmentation_ICCV_2021_paper.html), ICCV 2021 | peer-reviewed | Low-search augmentation candidate. |
| Hendrycks et al., [AugMix](https://arxiv.org/abs/1912.02781), ICLR 2020 | peer-reviewed / author manuscript | Corruption robustness and uncertainty direction. |
| Zhang et al., [mixup: Beyond Empirical Risk Minimization](https://openreview.net/pdf?id=r1Ddp1-Rb), ICLR 2018 | peer-reviewed | Mixed-sample regularization; recipe-level soft-label semantics are questionable. |
| Yun et al., [CutMix](https://arxiv.org/abs/1905.04899), ICCV 2019 | peer-reviewed / author manuscript | Patch mixing; area-weighted labels do not map cleanly to ingredients. |
| Zhong et al., [Random Erasing Data Augmentation](https://ojs.aaai.org/index.php/AAAI/article/view/7000), AAAI 2020 | peer-reviewed | Occlusion robustness; may erase sole visible evidence. |
| Wang et al., [SpliceMix: A Cross-Scale and Semantic Blending Augmentation Strategy for Multi-Label Image Classification](https://arxiv.org/abs/2311.15200), 2023 | preprint | Multi-label-specific mixing lead; food plausibility unverified. |

## 11. Metrics, calibration, uncertainty, and interpretation

| Source | Status | Relevance and boundary |
|---|---|---|
| Lipton et al., [Thresholding Classifiers to Maximize F1 Score](https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/), 2014 | peer-reviewed | Explains prevalence/score dependence of F1-optimal thresholds. |
| Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017 | peer-reviewed | Temperature-scaling baseline; multi-label reporting needs extra care. |
| Fisch et al., [Conformal Prediction Sets with Limited False Positives](https://proceedings.mlr.press/v162/fisch22a.html), ICML 2022 | peer-reviewed | Future controlled false-positive set prediction. |
| Cauchois et al., [Knowing What You Know: Valid and Validated Confidence Sets in Multiclass and Multilabel Prediction](https://jmlr.org/papers/v22/20-753.html), JMLR 2021 | peer-reviewed | Multi-label confidence sets; requires held-out calibration assumptions. |
| Selvaraju et al., [Grad-CAM](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html), ICCV 2017 | peer-reviewed | Class-specific attribution diagnostic, not localization proof. |
| Sundararajan et al., [Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html), ICML 2017 | peer-reviewed | Integrated Gradients and attribution axioms. |
| Kim et al., [Interpretability Beyond Feature Attribution: TCAV](https://research.google/pubs/interpretability-beyond-feature-attribution-quantitative-testing-with-concept-activation-vectors-tcav/), ICML 2018 | peer-reviewed / official publication page | Concept-level diagnostic with curated concept sets. |
| Geirhos et al., [Shortcut Learning in Deep Neural Networks](https://www.nature.com/articles/s42256-020-00257-z), Nature Machine Intelligence 2020 | peer-reviewed | Framework for context/cuisine/background shortcut analysis. |

## 12. Documentation and reproducibility

| Source | Status | Relevance and boundary |
|---|---|---|
| Gebru et al., [Datasheets for Datasets](https://www.microsoft.com/en-us/research/publication/datasheets-for-datasets/), CACM 2021 | peer-reviewed / official publication page | Dataset provenance, composition, uses, and limitations. |
| Pushkarna et al., [Data Cards: Purposeful and Transparent Dataset Documentation](https://research.google/pubs/data-cards-purposeful-and-transparent-dataset-documentation-for-responsible-ai/), FAccT 2022 | peer-reviewed / official publication page | Structured audience- and use-aware dataset documentation. |
| Mitchell et al., [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993), FAccT 2019 | peer-reviewed / author manuscript | Model scope, subgroup results, limitations, and intended use. |
| Pineau et al., [Improving Reproducibility in Machine Learning Research](https://www.jmlr.org/papers/v22/20-303.html), JMLR 2021 | peer-reviewed | Reproducibility checklist and reporting principles. |

## 13. Evidence gaps after this discovery

Primary literature does not eliminate the need for local studies of:

- normalization precision/recall on Yummly ingredient lines;
- human agreement on the four observability categories;
- duplicate/family edge precision at local thresholds;
- grouped split feasibility for the approved vocabulary;
- DINOv3 and SigLIP 2 throughput and memory in the exact environment;
- food-safe augmentation severity ranges;
- calibration stability for tail labels;
- the marginal value of explicit dependencies over cuisine and prevalence controls.

These gaps are converted to scoped experiments in the [recommended research program](recommendations.md).
