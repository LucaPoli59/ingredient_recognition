# Broad State-of-the-Art Discovery — 2026-08-02

Created: 2026-08-02  
Updated: 2026-08-02

## Context

This discovery supports the thesis benchmark defined in the [project objective](../../../project_objective/README.md) and the current [execution tracker](../../../next_steps.md). The target is **closed-vocabulary, recipe-level multi-label ingredient inference from one RGB image of a finished dish**. It is not ingredient-instance detection, segmentation, open-ended captioning, or full recipe generation.

The local evidence changes how external research should be interpreted:

- labels describe the reviewed recipe, including ingredients that may be visually hidden;
- the legacy Yummly-66K targets are not reproducible and include systematic substring-collision errors;
- exact and broader recipe-family leakage crosses the historical splits;
- the label distribution is sparse and strongly imbalanced;
- many source images are small or non-square, while the legacy pipeline warps them to a square;
- the available development GPU is an RTX 4060 with 8 GB VRAM.

These constraints make target repair, leakage control, evaluation design, and calibration part of the modeling problem rather than preliminary housekeeping.

## Scope and method

The review searched primary literature and official project or dataset documentation through **2026-08-02**. It deliberately spans adjacent fields because no single literature exactly matches the benchmark target:

1. food ingredient recognition, inverse cooking, recipe retrieval, food segmentation, and dietary assessment;
2. general multi-label classification, label-dependency modeling, and class-query heads;
3. supervised, self-supervised, and vision-language visual representation learning;
4. ingredient normalization, food ontologies, weak supervision, partial labels, and noisy labels;
5. long-tail learning and multi-label losses;
6. image preprocessing, augmentation, native-aspect processing, and sample mixing;
7. duplicate detection, group-aware splitting, metrics, calibration, uncertainty, and interpretability;
8. dataset, model, and reproducibility documentation.

The source cutoff and evidence-selection rules are documented in the [source catalog](source_catalog.md). Published peer-reviewed work is preferred. Preprints and very recent work are explicitly marked and treated as leads rather than settled evidence.

## Documents

- [Problem and literature landscape](problem_landscape.md): task taxonomy, food-domain evidence, and the gaps between published benchmarks and this project.
- [Models, objectives, and transfer strategies](models.md): backbones, multi-label heads, label dependencies, losses, VLMs, and 8 GB feasibility.
- [Data processing and augmentation](data_and_augmentation.md): target construction, ontology design, deduplication, split construction, preprocessing, and augmentation risks.
- [Evaluation, calibration, and analysis](evaluation_and_analysis.md): metrics, thresholding, confidence, observability, shortcuts, and reproducibility.
- [Recommended research program](recommendations.md): prioritized hypotheses, bounded candidate shortlist, experiment sequence, and stop/go criteria.
- [Primary-source catalog](source_catalog.md): categorized bibliography and transfer notes.

## Executive synthesis

### 1. There is no defensible single imported “state of the art”

Published food work variously predicts simplified ingredient taxonomies, visible ingredient pixels, recipes, instructions, nutritional quantities, or retrieval embeddings. Dataset definitions, vocabularies, splits, and metrics are incompatible. A high score on a food-segmentation or recipe-generation dataset is not evidence that the same method solves recipe-level ingredient-set inference on leakage-controlled Yummly-66K.

The project should therefore establish its own benchmark and compare controlled hypotheses on the same repaired targets and family-disjoint split.

### 2. Data and evaluation integrity dominate early model choice

No loss, foundation model, or label-dependency head can repair an incorrect ontology or prevent train/test family leakage. The benchmark builder remains the blocking dependency. Model research should proceed as executable specifications and shortlists until the regenerated labels, exclusions, family graph, split, vocabulary, and metric implementation are frozen.

### 3. The most informative model comparison is representation × head, not a model zoo

The first serious comparison should separate:

- representation quality: supervised convolutional baseline vs self-supervised visual foundation model vs vision-language pretraining;
- independent pooled classification vs class-specific spatial attention;
- independent labels vs explicit label-dependency modeling;
- ranking-oriented losses vs probability-calibrated objectives.

Testing every cross-product would be wasteful on an 8 GB GPU. The [recommendations](recommendations.md) define a staged sequence that promotes only candidates that beat simpler controls on multiple seeds and observability slices.

### 4. The highest-value near-term candidates are conservative

The evidence supports this initial order:

1. prevalence and cuisine-only diagnostics;
2. a reproducible pretrained ResNet or ConvNeXt baseline with sigmoid binary cross-entropy;
3. the existing DINOv2 representation after fixing the transform contract and comparing frozen, partial, and full adaptation;
4. one compact modern convolutional representation and one feasible vision-language representation;
5. ML-Decoder or a comparable class-query head on the strongest affordable backbone;
6. one explicit dependency model only after independent-head performance is stable;
7. a fixed-backbone loss study: BCE, distribution-balanced loss, asymmetric loss, and—if calibrated probabilities are a primary output—a strictly proper calibrated multi-label objective.

DINOv3, SigLIP 2, and open-vocabulary food segmentation are valuable contemporary leads, but they should enter through feasibility gates and controlled comparisons rather than being presumed superior.

### 5. Augmentation must respect the meaning of recipe-level labels

Aspect-preserving resizing, modest crops, flips where appropriate, and light photometric variation are defensible defaults. Aggressive crops can remove the only visible evidence for a label; strong color changes can destroy food cues. MixUp and CutMix create especially ambiguous supervision because recipe-level ingredient labels are not proportional to mixed pixel area. They belong in ablations, not the baseline. Synthetic image generation is not recommended before the benchmark is validated because ingredient fidelity is difficult to guarantee.

### 6. Calibration and shortcut analysis are benchmark outputs

Validation-only threshold selection, label-macro average precision, global-threshold micro-F1, group-level bootstrap intervals, and at least three seeds remain the primary evaluation contract. Calibration should be reported, not assumed. Attribution maps are diagnostic rather than proof of ingredient localization and must be paired with observability slices, cuisine/context controls, duplicate-free evaluation, and controlled image perturbations.

## Immediate consequence for project scope

This discovery completes the broad-discovery work package. It does **not** approve a final model stack and does not unblock training against the legacy labels. The next research step is to turn the highest-value uncertainties into focused topic investigations, then approve a bounded shortlist after the benchmark and ingredient protocol are stable.
