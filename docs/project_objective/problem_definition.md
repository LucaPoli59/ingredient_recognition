# Ingredient prediction from a food image: project objective

**Created:** 2026-08-02  
**Last updated:** 2026-08-02

## Executive definition

This project investigates **to what extent a machine-learning model can infer the normalized set of ingredients declared by a recipe from one RGB image of the finished dish**.

The task is not literal ingredient object detection. The target is recipe-level supervision: it includes ingredients that may be dissolved, transformed, occluded, used in small quantities, or entirely invisible in the final image. The correct scientific framing is therefore **weakly supervised, multi-label recipe-ingredient prediction under visual ambiguity and noisy labels**.

The project currently uses only the locally processed Yummly-66K data. Other datasets in the repository are outside the active research scope unless a later, explicit decision changes this objective.

## Research problem

Given a single photograph of a prepared dish, predict which entries from a fixed ingredient vocabulary were used in the associated recipe.

Let:

- $x_i$ be the image associated with recipe $i$;
- $V = \{v_1, \ldots, v_L\}$ be the normalized ingredient vocabulary;
- $y_i \in \{0,1\}^{L}$ be the recipe's multi-hot ingredient vector;
- $f_\theta(x_i) \in \mathbb{R}^{L}$ be a model's output logits.

The model learns an approximation to:

$$
P(y_i \mid x_i)
$$

and produces a score or calibrated probability for each ingredient. The final value of $L$ will be determined by the approved deterministic standardization and support rules. The legacy local metadata has 182 observed labels. The existing robust encoder adds an extra `<UNK>` output, yielding 183 model outputs, although no validation or test label is outside the training vocabulary. Its role in multi-label outputs, ingestion, filtered vocabularies, and sequence encoders must be investigated before the new encoder contract is fixed.

This formulation does **not** imply that $y_i$ is uniquely recoverable from $x_i$. Multiple recipes can produce visually indistinguishable dishes, and the same dish can be prepared with different ingredients. The project measures useful probabilistic inference, not deterministic reconstruction of the recipe.

## Primary research objective

Develop and evaluate a reproducible method that predicts useful ingredient profiles from food images while explicitly accounting for:

- incomplete visual observability of recipe ingredients;
- class imbalance and ingredient co-occurrence;
- noisy and semantically inconsistent target normalization;
- duplicated recipes and images;
- leakage between training, validation, and test data;
- domain and presentation biases in web-sourced food photography;
- uncertainty and calibration of multi-label predictions.

A model comparison is scientifically meaningful only after the target vocabulary and evaluation split satisfy the data-readiness gates defined below.

## Why the problem is intrinsically difficult

### Recipe labels are not visible-object labels

The most frequent targets include `salt`, `oil`, `pepper`, `sugar`, `water`, and `flour`. These often affect taste or preparation without having a unique visual signature. Other ingredients change state during cooking: butter melts, flour becomes dough or sauce, onions dissolve into a base, and spices alter color without remaining individually identifiable.

Consequently, the model can learn three different kinds of evidence:

1. **direct visual evidence**, such as visible tomato, shrimp, or broccoli;
2. **dish-level evidence**, such as a cake suggesting flour, egg, sugar, and butter;
3. **dataset priors and shortcuts**, such as cuisine, plating style, color, or common ingredient combinations.

The evaluation must distinguish useful inference from shortcut exploitation as far as practicable.

### The mapping from image to recipe is one-to-many

An image of soup, bread, curry, salad, or pasta does not identify a unique ingredient list. The local data contains identical image files associated with different label sets, which provides direct evidence of contradictory supervision. Even a perfect visual representation cannot remove this ambiguity.

### The target is a correlated set

Ingredients are not independent. Garlic, oil, onion, salt, and pepper co-occur frequently; cuisine and course further change their priors. Independent sigmoid outputs are a valid baseline, but structured or conditional methods may exploit these relationships. Such gains must be compared with simple global and cuisine-prior baselines to establish whether the image contributes information.

### The annotations are generated, not native ground truth

The current `ingredients_ok` field is a locally normalized derivative of ingredient lines. Historical preprocessing relies on an incorrectly defined support threshold, approximate merging, and unbounded substring rules. Some targets therefore represent preprocessing artifacts instead of ingredients. The checked-in preprocessor recreates the exact stored label set for only 80.42% of processed recipes. Model performance against these labels measures agreement with a non-reproducible legacy pipeline, so final targets will be regenerated from original ingredient lines rather than patched in place.

## Research questions

### Main question

How accurately and reliably can the ingredient set associated with a Yummly recipe be inferred from a single image after constructing a leakage-controlled, semantically defensible benchmark?

### Supporting questions

1. Which ingredient categories are visually predictable, contextually predictable, or effectively unobservable from a finished-dish image?
2. How much do pretrained visual representations improve over global-frequency, cuisine-prior, and simple supervised baselines?
3. How do class imbalance strategies affect common and rare ingredients separately?
4. Does explicitly modeling ingredient co-occurrence improve ranking, recall, calibration, or rare-label performance?
5. How much of the apparent performance on the current split is attributable to exact or near-duplicate leakage?
6. How much does repairing the ingredient taxonomy and known substring collisions change results?
7. Which image preprocessing strategy best preserves useful food structure without unnecessary distortion or upscaling artifacts?
8. How should uncertainty be represented when multiple ingredient sets are plausible for the same visual evidence?

## Scope

### Included

- Yummly data under `data/input/yummly/` and its traceable raw sources under `data/raw_input/yummly/`;
- one recipe-associated RGB image as the required model input;
- multi-label prediction of a deterministic, fixed training vocabulary;
- train/validation/test evaluation with exact-image leakage control;
- pretrained and task-specific visual encoders;
- class imbalance, label dependence, thresholding, ranking, and calibration;
- overall, per-label, and relevant metadata-slice error analysis;
- reproducible data transformations and experiment configurations.

### Excluded from the current objective

- Recipes1M or other datasets present in the repository;
- ingredient quantities, cooking instructions, preparation steps, nutrition, or portion estimation;
- cuisine or flavor prediction as the primary task;
- open-vocabulary ingredient generation;
- object localization, segmentation, or bounding-box annotation;
- guarantees about allergens, dietary compliance, health, or food safety;
- real-time deployment or product integration;
- claims of generalization beyond Yummly-like web food photography without an external evaluation dataset.

Cuisine, course, flavor, recipe name, and raw ingredient text may be used for auditing, grouping, error analysis, or auxiliary experiments. They must not silently become inference inputs in an image-only experiment.

## Intended output and interpretation

For an image, the system should return:

- one score or probability per ingredient;
- a ranked ingredient list;
- a thresholded set when a discrete prediction is required;
- sufficient metadata to identify the vocabulary and decision thresholds used.

The output means “ingredients plausible for the recipe represented by this image under the training distribution.” It does not mean “ingredients visibly detected in the pixels,” “the only possible recipe,” or “a safety-certified ingredient list.”

## Data-readiness gates

No result should be treated as a final project result until all of the following are satisfied.

### Gate 1: reproducibility

- Preserve a deterministic raw-ingredient-to-`ingredients_target` transformation.
- Keep the standardization rules, configuration, and random seed under version control.
- Represent split membership through the same selected metadata filename under train, validation, and test.
- Verify that identical input and configuration produce identical metadata content.

### Gate 2: target validity

- Remove known substring collisions and cover the adopted rules with regression tests.
- Regenerate targets from original ingredient lines using token-aware, deterministic rules; do not patch legacy `ingredients_ok` lists in place.
- Merge accidental singular/plural and phrase variants where semantically appropriate.
- Define broad categories, fine-grained categories, aliases, retention criteria, and intentionally excluded ingredients in code and documentation.
- Preserve the source `ingredients` lines and write deterministic, duplicate-free `ingredients_target` lists.

### Gate 3: image validity

- Verify automatically that every referenced image exists and decodes.
- Accept residual quality and semantic noise instead of introducing manual image review.
- Keep image preprocessing consistent across compared models or document model-specific differences.

### Gate 4: split integrity

- Ensure no byte-identical SHA-256 image group crosses split boundaries.
- Do not group by perceptual similarity, recipe name, ingredient similarity, or manual family assignment.
- Preserve meaningful cuisine and target distributions without breaking exact-image groups.
- Freeze the final test set before model selection.

### Gate 5: evaluation validity

- Include all-negative, global-prior, cuisine-prior, and simple visual baselines.
- Report common and rare labels separately.
- Select calibration and thresholds using validation data only after threshold-free model selection.
- Report uncertainty across seeds or bootstrap samples.
- Investigate and test `<UNK>` separately for multi-label outputs, ingestion fallback, filtered vocabularies, and sequence inputs before changing it.

## Evaluation principles

### Metrics

No single metric is sufficient. Two metrics are primary and must be reported together:

- **mean average precision (mAP), macro-averaged over labels** for threshold-free model selection and rare-label visibility;
- **micro F1** for the discrete ingredient set, using the frozen validation-selected global threshold.

The required secondary evaluation set is:

- **micro average precision** and **macro F1**;
- **per-label average precision, precision, recall, and support**;
- **precision@5**, **recall@5**, **precision@10**, and **recall@10** for ranked ingredient suggestions;
- **sample-wise F1** for recipe-level usefulness;
- **Brier score and expected calibration error** where scores are interpreted as probabilities;
- **Hamming loss** as a secondary error-rate description;
- **exact match ratio** only as a strict secondary measure.

Raw label accuracy must not be a primary metric. Because the current target density is about 5%, predicting every label as absent already produces approximately 94.98% label accuracy while obtaining zero precision, recall, and F1.

### Required baselines

The current test data gives the following reproducible prior baselines:

| Baseline | Micro F1 | Mean sample F1 | Interpretation |
|---|---:|---:|---|
| All labels absent | 0.000 | 0.000 | Demonstrates misleading label accuracy |
| Global top 5 train labels | 0.357 | 0.341 | Frequency-only baseline |
| Global top 9 train labels | 0.393 | 0.379 | Frequency baseline near mean cardinality |
| Cuisine top 5 train labels | 0.387 | 0.378 | Uses ground-truth cuisine prior |
| Cuisine top 9 train labels | 0.459 | 0.443 | Measures strength of cuisine-label correlation |

The cuisine-prior baseline is diagnostic and not an image-only competitor because it uses known cuisine metadata. An image model should nevertheless be analyzed for whether it merely reconstructs this prior.

### Evaluation slices

Results should be broken down by:

- ingredient frequency;
- cuisine and course where metadata is valid;
- direct versus contextual versus low-observability ingredients;
- image resolution and source-quality flags;
- records belonging to singleton or repeated exact-image groups;
- common dish types and high-cardinality recipes;
- labels affected by normalization changes.

Observability remains an important interpretation issue, but it is not a benchmark-construction review requirement. If a later ingredient-selection study introduces observability annotations, it must be reported as a separate analysis and must not silently redefine the recipe-level target.

## Success criteria

The project is successful when it delivers all of the following:

1. a documented, reproducible, leakage-controlled Yummly benchmark;
2. a deterministic target vocabulary with known residual noise documented;
3. prior and visual baselines evaluated with appropriate multi-label metrics;
4. a justified model choice supported by controlled comparisons;
5. per-label and observability-aware analysis rather than only aggregate scores;
6. calibrated limitations about what can and cannot be inferred from a food image;
7. enough code, metadata, and experiment configuration to reproduce the data, thresholds, and reported results.

Numeric model-performance thresholds should be set after discovery research identifies comparable work and after the corrected benchmark is frozen. Setting them on the current contaminated split would create a target tied to leakage and label noise.

## Assumptions and constraints

- Each retained record is intended to pair one recipe with one image, but current data includes duplicates and mismatches.
- Ingredient absence from the selected feature field is treated as negative supervision even though normalization may omit valid raw ingredients.
- Images are web-sourced and biased toward attractive, finished dishes rather than controlled observations.
- The local dataset covers ten cuisine labels and is not representative of global food in general.
- Compute, thesis schedule, and available annotations constrain manual relabeling.
- The current model code assumes a fixed closed vocabulary and uses `BCEWithLogitsLoss` by default.
- The default split, vocabulary, and metrics are provisional until the data-readiness gates are met.

## Decisions fixed before model discovery

The benchmark decisions are now fixed in [`benchmark_decisions.md`](benchmark_decisions.md):

1. keep `feature_label` configurable and default new configurations to `ingredients_target`;
2. regenerate `ingredients_target` deterministically from all original Yummly ingredient lines while leaving legacy `ingredients_ok` metadata unchanged;
3. perform automatic image existence and decoding checks without manual review;
4. construct a deterministic 80/10/10 split that groups only byte-identical SHA-256 images and balances cuisine and targets;
5. derive the ordered vocabulary from training metadata and preserve it with each experiment;
6. preserve historical experiments through in-memory compatibility instead of rewriting saved files;
7. investigate `<UNK>` before deciding its new role and preserve its historical behavior;
8. use label-macro mAP and micro F1 as the paired primary metrics;
9. fit calibration and thresholds on validation only;

These decisions close the design questions but do not satisfy the data-readiness gates. Model discovery can survey candidate methods now; final comparative experiments must wait for the regenerated benchmark.

## Evidence and related documents

- [`yummly_data_audit.md`](yummly_data_audit.md) contains the complete data evidence supporting this formulation.
- [`benchmark_decisions.md`](benchmark_decisions.md) records the binding target, split, quality, threshold, metric, observability, and reproducibility decisions.
- [`../implementation_details/models.md`](../implementation_details/models.md) describes the model implementations currently available.
- [`../research/README.md`](../research/README.md) defines how subsequent discovery and topic research is recorded.
- The reproducible audit is implemented in [`../../src_scratches/data_anlysis/yummly_audit.py`](../../src_scratches/data_anlysis/yummly_audit.py).
- The decision-oriented second pass is implemented in [`../../src_scratches/data_anlysis/yummly_deep_audit.py`](../../src_scratches/data_anlysis/yummly_deep_audit.py).

## External source

The source dataset is described by Min et al. in [*You Are What You Eat: Exploring Rich Recipe Information for Cross-Region Food Analysis*](https://openreview.net/pdf?id=F9oSOeGGkwP), IEEE Transactions on Multimedia, 2018.
