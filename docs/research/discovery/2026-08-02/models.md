# Models, Objectives, and Transfer Strategies

Created: 2026-08-02  
Updated: 2026-08-02

## 1. Experimental decomposition

A model should be treated as four separable choices:

\[
\text{image pipeline} \rightarrow \text{representation} \rightarrow \text{multi-label head} \rightarrow \text{loss and decision rule}.
\]

Changing several choices together prevents causal interpretation. The recommended experiment sequence fixes three components while testing the fourth.

## 2. Visual representations

### 2.1 Supervised convolutional controls

A standard ImageNet-pretrained ResNet remains a necessary control because it is cheap, familiar, and represented in the current codebase. A compact [ConvNeXt V2](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html) variant is the more informative modern convolutional candidate. ConvNeXt V2 spans very small to large models and combines a modern convolutional architecture with masked-autoencoder co-design.

Use one of these as the convolutional baseline, not an exhaustive sweep over ResNet, DenseNet, EfficientNet, and ConvNeXt families. A pretrained ResNet-50 is the lowest-integration-risk baseline; a compact ConvNeXt V2 is the preferred modern comparison if the dependency and checkpoint path are stable.

### 2.2 DINOv2

[DINOv2](https://arxiv.org/abs/2304.07193) is already the project’s principal self-supervised candidate. It offers strong general visual features without food-specific supervision. The current repository implementation aggregates the final four class tokens and mean patch features from a ViT-B/14-reg model, producing a 3,840-dimensional representation.

Before interpreting any result, the local implementation needs a single authoritative preprocessing contract. The current transform composition can normalize twice, and its `pretrained` switch does not provide a genuine random-initialization control. These are implementation defects, not hyperparameters.

The useful DINOv2 adaptation ladder is:

1. frozen encoder plus linear independent-label head;
2. frozen encoder plus the selected class-query head;
3. partial adaptation of late blocks or parameter-efficient modules;
4. full fine-tuning only if memory and overfitting controls permit.

This ladder separates representation quality from adaptation capacity.

### 2.3 DINOv3 as a contemporary lead

[DINOv3](https://arxiv.org/abs/2508.10104) extends self-supervised dense feature learning at larger scale. The official [DINOv3 repository](https://github.com/facebookresearch/dinov3) exposes several ViT and ConvNeXt variants. It is a relevant current comparison because ingredient recognition may benefit from dense local features as well as global semantics.

It should be feasibility-gated rather than automatically replacing DINOv2:

- verify checkpoint access, terms, dependencies, and reproducible loading;
- smoke-test the smallest useful variant at the project input resolution;
- measure frozen feature quality before paying for adaptation;
- retain DINOv2 as the continuity baseline.

### 2.4 Vision-language pretraining

[CLIP](https://arxiv.org/abs/2103.00020) established transferable image–text representations at scale. [SigLIP](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language-Image_Pre-Training_ICCV_2023_paper.html) replaces global contrastive normalization with pairwise sigmoid losses. [SigLIP 2](https://arxiv.org/abs/2502.14786) adds multilingual, localization, and self-supervised objectives and publishes multiple model scales.

Vision-language pretraining is attractive for ingredients because label names carry semantics, but it introduces a prompt and vocabulary pathway absent from purely visual models. Two tests should be distinguished:

- **image-encoder transfer:** freeze or fine-tune the image encoder and train the same independent head used for other backbones;
- **language-informed classification:** initialize or define label queries from ingredient text, then compare with randomly initialized learned queries.

The first tests representation quality; the second tests semantic priors. Prompt ensembles and zero-shot scores can be reported as diagnostics, but a closed-vocabulary supervised benchmark should not be reduced to prompt engineering.

[DualCoOp](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c5169260ef32d1bd3597c14d8c89b034-Abstract-Conference.html) and [MuMIC](https://ojs.aaai.org/index.php/AAAI/article/view/26850) show how vision-language representations can be adapted to multi-label recognition. They are useful focused-research leads if the plain SigLIP/CLIP encoder test is promising.

### 2.5 Native aspect and variable resolution

[NaViT](https://proceedings.neurips.cc/paper_files/paper/2023/hash/06ea400b9b7cfce6428ec27a371632eb-Abstract-Conference.html) trains on native aspect ratios and varying resolutions through sequence packing. [FlexiViT](https://openaccess.thecvf.com/content/CVPR2023/html/Beyer_FlexiViT_One_Model_for_All_Patch_Sizes_CVPR_2023_paper.html) supports multiple patch sizes. These methods reinforce the concern that forced square warping is avoidable, but replacing the complete backbone is not the first test. A controlled aspect-preserving preprocessing ablation on existing backbones is cheaper and directly addresses the local failure mode.

## 3. Multi-label heads

### 3.1 Independent pooled head

Global pooling followed by one affine logit per label is the reference head. Advantages include low memory use, easy calibration, and clean attribution of gains to the backbone. Its limitation is that all labels share one pooled representation.

### 3.2 ML-Decoder

[ML-Decoder](https://openaccess.thecvf.com/content/WACV2023/html/Ridnik_ML-Decoder_Scalable_and_Versatile_Classification_Head_WACV_2023_paper.html) uses learned queries over spatial features and group decoding to control parameter cost. It is the preferred first structured head because it tests class-specific spatial access without requiring an autoregressive ingredient sequence.

Primary hypothesis: on the same backbone and loss, ML-Decoder improves label-macro AP, particularly for directly observable ingredients and multi-component dishes, without relying entirely on label co-occurrence.

### 3.3 Query2Label

[Query2Label](https://arxiv.org/abs/2107.10834) is a stronger transformer-decoder reference for label queries. It is more useful as a focused comparison if ML-Decoder succeeds than as a co-equal first implementation. Running both immediately would spend compute on closely related hypotheses.

### 3.4 Graph and transformer dependency heads

[ML-GCN](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Multi-Label_Image_Recognition_With_Graph_Convolutional_Networks_CVPR_2019_paper.html) and [C-Tran](https://openaccess.thecvf.com/content/CVPR2021/html/Lanchantin_General_Multi-Label_Image_Classification_With_Transformers_CVPR_2021_paper.html) represent two explicit dependency strategies. A local graph could combine training-set co-occurrence with frozen ontology relations.

Safeguards are mandatory:

- build co-occurrence only from the training split;
- freeze graph construction before validation comparison;
- report an image-free graph/prior baseline;
- test whether gains concentrate in contextual or not-inferable labels;
- never infer graph edges from test labels.

Only one dependency head should enter the initial shortlist. Its purpose is to test a scientific hypothesis, not to make the architecture maximally elaborate.

### 3.5 Autoregressive set decoding

Inverse Cooking demonstrates unordered ingredient-set decoding, but an autoregressive implementation adds order conventions, exposure bias, decoding rules, and higher integration cost. It should be considered only if non-autoregressive class-query and graph heads expose a clear limitation.

## 4. Objectives for imbalance and weak labels

### 4.1 Binary cross-entropy

Sigmoid binary cross-entropy is the essential baseline. It is a proper scoring rule under the assumed target distribution, is simple to weight or mask, and supports probability calibration analysis. It should not be discarded merely because ranking-focused losses may raise average precision.

### 4.2 Asymmetric loss

[Asymmetric Loss](https://arxiv.org/abs/2009.14119) downweights easy negatives and treats positive and negative focusing differently. This directly addresses sparse multi-label gradients and is a strong ranking-oriented candidate.

Risk: the loss intentionally changes the probabilistic objective. A 2024 study of [calibrated multi-label neural networks](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Towards_Calibrated_Multi-label_Deep_Neural_Networks_CVPR_2024_paper.html) notes that asymmetric and related losses are not strictly proper. ASL results therefore require post-hoc calibration evaluation and must not be interpreted as probabilities by default.

### 4.3 Distribution-balanced loss

[Distribution-Balanced Loss](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490154.pdf) addresses label co-occurrence and negative-tolerant regularization in long-tailed multi-label data. It is more tailored to the local setting than importing a single-label class-balanced formula unchanged.

### 4.4 Two-way and calibrated multi-label losses

[Two-Way Multi-Label Loss](https://openaccess.thecvf.com/content/CVPR2023/html/Kobayashi_Two-Way_Multi-Label_Loss_CVPR_2023_paper.html) jointly emphasizes hard samples and hard classes. The calibrated-loss work above proposes a strictly proper asymmetric construction and label-pair regularization. These are valuable focused comparisons after BCE, distribution-balanced loss, and ASL establish the local behavior.

### 4.5 Partial, missing, and noisy labels

The literature includes [partial label masking for imbalance](https://openaccess.thecvf.com/content/CVPR2021W/LLID/html/Duarte_PLM_Partial_Label_Masking_for_Imbalanced_Multi-Label_Classification_CVPRW_2021_paper.html), learning with [single observed positive labels](https://arxiv.org/abs/2106.09708), [positive–unlabeled multi-label learning](https://arxiv.org/abs/2306.16016), joint long-tail and partial-label learning in [COMC](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Learning_in_Imperfect_Environment_Multi-Label_Classification_with_Long-Tailed_Distribution_and_ICCV_2023_paper.html), and [trusted multi-label loss correction](https://proceedings.mlr.press/v189/ghiassi23a.html).

These methods make different noise assumptions. The local legacy targets contain systematic false positives from substring collisions as well as omissions and normalization ambiguity. Treating every unobserved class as an unlabeled positive candidate is not a sufficient repair. The correct priority is:

1. regenerate targets from raw ingredient lines;
2. represent genuinely unresolved mappings explicitly;
3. measure residual label quality on a reviewed audit sample;
4. adopt a noise method only if its assumptions match the measured residual errors.

## 5. Parameter-efficient adaptation

[Visual Prompt Tuning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930696.pdf) and other [parameter-efficient ViT adaptation](https://ojs.aaai.org/index.php/AAAI/article/view/25160) reduce trainable parameters. They may fit the 8 GB constraint and reduce overfitting. They are not substitutes for the frozen and full/late-block baselines: parameter count, activation memory, throughput, and accuracy must all be measured.

Recommended use: introduce one PEFT method only when full adaptation is memory-limited or demonstrably overfits, and compare it with frozen and late-block fine-tuning on the same encoder.

## 6. Compute-aware feasibility

The project should record peak allocated VRAM, images per second, effective batch size, input resolution, precision, trainable parameter count, and wall-clock time for every candidate. Gradient accumulation changes optimizer batch size but does not reduce activation memory per image. Mixed precision, activation checkpointing, frozen encoders, smaller variants, and late-block tuning are the primary levers.

Suggested feasibility gate for each new backbone:

1. deterministic checkpoint load without network dependence during normal runs;
2. one-batch forward and backward at the planned resolution;
3. transform outputs verified numerically and visually;
4. peak VRAM below a documented safety margin;
5. short overfit test on a tiny training subset;
6. frozen-feature pilot before an expensive fine-tune.

## 7. What not to prioritize

- A broad sweep of near-equivalent CNN families.
- Very large foundation models that require aggressive resolution or batch compromises.
- Open-ended VLM generation as the headline predictor.
- Autoregressive recipe generation.
- Segmentation as a required auxiliary task without local masks.
- Complex long-tail architectures before target quality is measured.
- Simultaneous changes to backbone, head, loss, preprocessing, and vocabulary.
- A model selected from one seed or one aggregate F1 score.

## 8. Model evidence matrix

| Candidate | Scientific question | First mode | Main risk | Priority |
|---|---|---|---|---|
| ResNet-50 or compact ConvNeXt V2 | What does a strong supervised convolutional representation achieve? | full or late-block fine-tune | underpowered/overfit baseline | mandatory |
| DINOv2 ViT-B/14-reg | Does self-supervised general representation transfer? | frozen, then partial | transform bug; memory | mandatory |
| SigLIP 2 feasible base variant | Do image–text semantics improve ingredient transfer? | frozen image encoder | prompt/semantic shortcut | high |
| DINOv3 feasible small/base variant | Do newer dense SSL features improve transfer? | frozen encoder | access/dependency/compute | conditional high |
| ML-Decoder | Does label-specific spatial access beat global pooling? | strongest fixed backbone | complexity without gain | high |
| One graph or label-state head | Do explicit dependencies add beyond priors? | strongest fixed backbone | cuisine shortcut | medium |
| BCE / DB / ASL | Which imbalance objective improves tail ranking? | fixed model | calibration tradeoff | high |
| Strictly proper asymmetric objective | Can ranking and calibration both improve? | fixed model | implementation maturity | medium-high |
| One PEFT method | Can adaptation fit 8 GB without losing quality? | selected ViT/VLM | extra tuning surface | conditional |
