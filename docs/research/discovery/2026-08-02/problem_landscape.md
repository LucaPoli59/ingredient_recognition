# Problem and Literature Landscape

Created: 2026-08-02  
Updated: 2026-08-02

## 1. The benchmark target

For image \(x\), the system predicts a subset of a frozen ingredient vocabulary \(V\):

\[
f(x) = \hat{Y} \subseteq V.
\]

The target \(Y\) is the normalized set of reviewed recipe ingredients, not merely the subset visible in the pixels. This makes the task a mixture of visual recognition and contextual inference. It is weakly supervised because image regions are not aligned to ingredient labels, and it is intrinsically ambiguous because preparation can hide or transform ingredients.

The project’s binding semantics are defined in [problem definition](../../../project_objective/problem_definition.md) and [benchmark decisions](../../../project_objective/benchmark_decisions.md). External papers are useful only to the extent that their supervision and output semantics transfer to those decisions.

## 2. Adjacent tasks are not interchangeable

| Research task | Typical output | What transfers | What does not transfer directly |
|---|---|---|---|
| Food classification | one dish/cuisine class | visual backbones, augmentations | single-label metrics and assumptions |
| Visible ingredient recognition | visible ingredient classes | local evidence, class-specific attention | omission of hidden recipe ingredients |
| Food segmentation | ingredient/food pixels | localization priors, qualitative audits | pixel labels, visible-only ontology |
| Recipe retrieval | image–recipe similarity | multimodal pretraining, hard negatives | ranking a recipe corpus rather than predicting a set |
| Inverse cooking | ingredient set plus instructions | set prediction and dependency models | generation objective, larger paired corpora |
| Dietary assessment | foods, portions, nutrients | fine-grained food understanding | quantities and nutritional ground truth |
| Open-vocabulary recognition | text-conditioned classes | semantic transfer and tail exploration | open output space and prompt dependence |
| This project | normalized recipe ingredient set | — | closed vocabulary, family-disjoint Yummly benchmark |

A literature claim should be downgraded when the source target differs in visibility, granularity, vocabulary construction, or evaluation unit.

## 3. Food-specific research

### 3.1 Direct multi-label ingredient recognition

[Bolaños et al. (2017)](https://arxiv.org/abs/1707.08816) is the closest foundational match. It formulates food ingredients as multi-label classification, uses sigmoid outputs and binary cross-entropy on ImageNet-pretrained CNNs, and evaluates simplified ingredient vocabularies. The paper explicitly recognizes that recipe ingredients may be invisible or visually transformed. Its enduring lessons are:

- multi-label classification is an appropriate baseline formulation;
- ingredient normalization materially affects learnability and reported scores;
- transfer learning is necessary at food-dataset scale;
- visible evidence alone cannot fully determine recipe ingredients.

Its absolute scores should not be imported. Its Ingredients101 and Recipes5k vocabularies, simplification rules, split design, and metric protocol differ from this project.

[Ismail and Yuan (2022)](https://arxiv.org/abs/2210.14147) further studies multi-label ingredient recognition using Nutrition5K. It supports modern multi-label modeling as a relevant direction, but Nutrition5K’s capture process and nutrition-oriented annotations differ from web recipes.

### 3.2 Ingredient set prediction within inverse cooking

[Inverse Cooking](https://arxiv.org/abs/1812.06164) predicts an ingredient set before generating instructions. Its transformer-based ingredient decoder models dependencies without treating the target as a semantically meaningful sequence. The transferable idea is structured set prediction: ingredients co-occur and some combinations are more plausible than independent sigmoids suggest.

The project should not copy the complete generation pipeline. Recipe generation introduces language-model objectives and evaluation questions outside scope. A class-query or set-prediction head can be tested independently against an otherwise identical visual backbone.

### 3.3 Recipe–image representation learning

[Recipe1M](https://openaccess.thecvf.com/content_cvpr_2017/html/Salvador_Learning_Cross-Modal_Embeddings_CVPR_2017_paper.html) and [Recipe1M+](https://arxiv.org/abs/1810.06553) show that large image–recipe corpora can train useful cross-modal embeddings. This supports multimodal pretraining and retrieval-derived features as hypotheses. It does not justify adding those datasets to the active benchmark: licensing, provenance, leakage, vocabulary alignment, and scope must be resolved first.

A recent preprint on [unbiased food image-to-recipe retrieval](https://arxiv.org/abs/2511.15201) makes the visibility gap explicit and uses a Query2Label-like multi-label ingredient module to enrich retrieval features. Its class-query design and analysis of subtle/occluded ingredients are directly relevant leads; its causal formulation and retrieval claims are not yet evidence for the local classification benchmark.

### 3.4 Segmentation and localized ingredient evidence

[FoodSeg103](https://arxiv.org/abs/2105.05409) provides pixel-level food ingredient/food-category annotations and documents the difficulty of overlapping, small, and visually similar food regions. [UECFoodPixComplete](https://mm.cs.uec.ac.jp/uecfoodpix/) is another official food-segmentation resource. [OVFoodSeg](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_OVFoodSeg_Elevating_Open-Vocabulary_Food_Image_Segmentation_via_Image-Informed_Textual_Representation_CVPR_2024_paper.html) extends the problem to open-vocabulary food segmentation using image-informed text representations.

These sources motivate class-specific spatial features and qualitative localization checks. They do not change the primary target to segmentation. Recipe labels may be hidden, while segmentation labels must have visible pixels; forcing every positive recipe label to localize would be semantically wrong.

### 3.5 Dietary and multimodal foundation-model evidence

[Nutrition5K](https://openaccess.thecvf.com/content/CVPR2021/html/Thames_Nutrition5k_Towards_Automatic_Nutritional_Understanding_of_Generic_Food_CVPR_2021_paper.html) shows the value of controlled multi-view capture for mass and nutrition estimation, but the local dataset has only uncontrolled web images and no portions.

A 2025 evaluation asks whether [vision-language models are ready for dietary assessment](https://openaccess.thecvf.com/content/CVPR2025W/MTF/html/Romero-Tapiador_Are_Vision-Language_Models_Ready_for_Dietary_Assessment_Exploring_the_Next_CVPRW_2025_paper.html) and finds persistent difficulty with fine-grained food concepts and cooking styles. That is evidence against treating open-ended VLM answers as ground truth or as the primary benchmark model. VLM encoders, text embeddings, and audited teacher suggestions remain reasonable components.

A 2026 journal comparison of [vision-language models for ingredient recognition and nutrient estimation](https://doi.org/10.1016/j.crfs.2026.101405) tests single- and progressive multi-view prompting on Nutrition5K. It reports a precision/recall tradeoff as additional views expose more components and continued difficulty on composite or visually ambiguous foods. This is the most current direct VLM evidence found, but the local benchmark has one image, a frozen vocabulary, and locally trained predictors; API-model and multi-view results are diagnostic context, not comparable scores.

[Retrieval-Augmented Recipe Generation](https://openaccess.thecvf.com/content/WACV2025/html/Liu_Retrieval_Augmented_Recipe_Generation_WACV_2025_paper.html) is relevant to future recipe generation and retrieval, not to the first closed-vocabulary benchmark.

At the discovery cutoff, a newly submitted preprint on [food segmentation with LLM-derived ingredient labels](https://arxiv.org/abs/2607.25820) injects language-derived semantics into feature- and query-level segmentation modules. It is evidence that semantic injection into dense food representations is an active direction, but it is too recent, uses segmentation supervision, and depends on automatically inferred labels. It is recorded as an emerging lead rather than shortlisted.

## 4. General multi-label research that transfers

### 4.1 Independent classification remains the control

A pooled visual feature followed by one logit per label is intentionally simple. It reveals how much performance comes from the representation and establishes a calibration-friendly baseline. Every more complex head should be compared to this control on the same backbone, input pipeline, loss, and seed schedule.

### 4.2 Class-query and spatial-attention heads

[Query2Label](https://arxiv.org/abs/2107.10834) uses label queries to attend to spatial image features. [ML-Decoder](https://openaccess.thecvf.com/content/WACV2023/html/Ridnik_ML-Decoder_Scalable_and_Versatile_Classification_Head_WACV_2023_paper.html) offers a scalable attention-based classification head with group decoding. Both target a limitation of global average pooling: different labels may depend on different image regions.

This hypothesis is especially relevant for multi-component dishes, but its success should be tested separately from backbone changes. ML-Decoder is the more practical first candidate because it was designed as an efficient drop-in head.

### 4.3 Explicit label dependencies

[ML-GCN](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Multi-Label_Image_Recognition_With_Graph_Convolutional_Networks_CVPR_2019_paper.html) uses label co-occurrence to construct classifier relationships. [C-Tran](https://openaccess.thecvf.com/content/CVPR2021/html/Lanchantin_General_Multi-Label_Image_Classification_With_Transformers_CVPR_2021_paper.html) represents labels as transformer states and studies known, unknown, and partially observed labels. [Inverse Cooking](https://arxiv.org/abs/1812.06164) also models ingredient-set dependencies.

Ingredient relationships are real, but they create two hazards:

1. a model can learn cuisine or recipe-template priors without using the image;
2. co-occurrence estimated from leaked or noisy splits can amplify benchmark defects.

Dependency modeling should therefore be tested only after the family-disjoint split is frozen and always against image-free prevalence/cuisine baselines and observability slices.

### 4.4 Hierarchies and semantic relations

[HMCN](https://proceedings.mlr.press/v80/wehrmann18a.html) jointly models local and global hierarchical losses. A food-specific [multi-relational graph approach](https://ojs.aaai.org/index.php/AAAI/article/view/6626) uses ingredient hierarchy, attributes, and co-occurrence for zero-shot ingredient recognition.

The local ontology will contain aliases, normalization relations, and possibly parent categories. That structure is useful for diagnostics and future tail transfer, but the headline benchmark must retain its frozen leaf-level semantics. Hierarchical credit should not silently replace exact label evaluation.

## 5. The central gaps

The following questions are not answered by existing food papers:

- How much published performance survives a recipe-family-disjoint split?
- Which ingredients are directly visible, contextually inferable, or not inferable from the image?
- How do model rankings change after reproducible boundary-aware ingredient normalization?
- Which methods improve label-macro ranking without degrading probability calibration?
- Does explicit co-occurrence modeling add visual information or mostly reproduce cuisine priors?
- Which pretrained representation gives the best value under an 8 GB training budget?
- How much does aspect-preserving preprocessing matter for the small and non-square Yummly images?

These gaps define the local research program more reliably than leaderboard transfer.

## 6. State-of-the-art conclusion for this project

The relevant state of the art is a **system design**, not one architecture:

1. deterministic and auditable target construction;
2. family-aware leakage control;
3. strong pretrained visual representations;
4. multi-label heads that can access spatial and semantic structure;
5. imbalance-aware but calibration-conscious objectives;
6. validation-only decision rules and uncertainty estimates;
7. analysis that separates direct visual evidence from contextual inference.

The project can make a credible contribution by integrating these pieces under one explicit benchmark contract and showing which gains remain after data integrity and shortcut controls.
