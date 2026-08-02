# Evaluation, Calibration, and Analysis

Created: 2026-08-02  
Updated: 2026-08-02

## 1. Evaluation contract

The primary evaluation remains the paired view approved in the [benchmark decisions](../../../project_objective/benchmark_decisions.md):

- **label-macro average precision** for ranking quality across the frozen label set;
- **global-threshold micro-F1** for one operational set-prediction view.

Neither number is sufficient alone. Macro AP exposes performance on less frequent labels but does not define a predicted set. Micro-F1 weights common label decisions heavily and depends on threshold selection.

## 2. Metric implementation details that must be frozen

The evaluation specification should state:

- the exact vocabulary and label order;
- whether AP follows non-interpolated precision–recall integration and which library/version implements it;
- behavior for tied scores;
- policy for labels with zero positive support;
- whether every sample and every prediction is included after exclusions;
- the grid or algorithm used for global-threshold selection;
- deterministic tie-breaking between equal validation scores;
- the exact bootstrap unit and random seeds;
- aggregation across training seeds.

Labels should not be dropped from a result because a model predicts them poorly. Zero-support prevention belongs in vocabulary/split construction; any unavoidable zero-support class must be disclosed and handled by a predeclared rule.

## 3. Threshold selection

F1-optimal thresholds depend on score distributions and class prevalence, as analyzed in [Thresholding Classifiers to Maximize F1 Score](https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/). Thresholds are therefore learned decisions, not harmless display settings.

Binding policy:

1. select one global threshold using validation predictions only;
2. freeze it before test evaluation;
3. report test micro-F1 at that threshold;
4. keep label-specific thresholds secondary because they add many fitted degrees of freedom;
5. never choose thresholds on the full test set or per reported test slice.

Top-\(k\) prediction is a useful diagnostic but is not the primary decision rule because recipe ingredient-set sizes vary.

## 4. Calibration

### 4.1 Why calibration matters

Ingredient probabilities may support thresholding, error review, or future uncertainty-aware interfaces. Ranking improvements do not guarantee calibrated confidence. Asymmetric and focal-style objectives can be particularly strong rankers while distorting probabilities.

Report at minimum:

- multi-label negative log-likelihood or binary cross-entropy;
- Brier score, micro and label-macro where support permits;
- reliability diagrams for pooled predictions and selected support bands;
- an expected calibration error with binning fully specified;
- calibration slope/intercept or a comparable continuous diagnostic;
- performance before and after validation-fitted calibration.

Multi-label calibration is not fully described by pooling every label decision because common negatives dominate. Include positive-focused, per-label, and support-stratified views.

### 4.2 Post-hoc calibration

[Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html) is a simple validation-fitted baseline. One shared temperature is low variance but may not correct label-specific differences. Per-label calibrators need enough positives and can overfit tail labels.

Recommended sequence:

1. uncalibrated sigmoid scores;
2. one shared validation-fitted temperature or logistic recalibration;
3. support-pooled or regularized per-label calibration only if the shared method is insufficient;
4. refit nothing on test data.

The 2024 work on [calibrated multi-label neural networks](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Towards_Calibrated_Multi-label_Deep_Neural_Networks_CVPR_2024_paper.html) motivates comparing ranking-oriented objectives with a strictly proper asymmetric objective rather than assuming post-hoc correction will remove every loss-induced bias.

## 5. Statistical uncertainty

### 5.1 Training-seed variation

Every promoted model should run at least three predetermined seeds. Report mean, standard deviation, and individual runs for the paired primary metrics. A model is not promoted on its best seed.

### 5.2 Group-level bootstrap

Because images/recipes are related within recipe families, bootstrap recipe-family IDs rather than independent images. For each replicate:

1. sample test families with replacement;
2. include all records belonging to each sampled family;
3. compute the frozen metrics and pairwise model difference;
4. report percentile or bias-corrected intervals according to a predeclared implementation.

For model comparison, paired bootstrap replicates should resample the same families for both models. Seed variance and test-sample uncertainty answer different questions and should be shown separately.

### 5.3 Multiple comparisons

A large model/hyperparameter search makes chance improvements more likely. Use validation to reduce the field, keep the final test frozen, and report the number of candidates considered. The experiment registry should distinguish exploratory results from confirmatory final comparisons.

## 6. Observability and ambiguity

The project’s reviewed observability tags—`direct`, `contextual`, `not_inferable`, and `uncertain`—are central analysis dimensions.

Recommended reporting:

- primary metrics on the full frozen target;
- label/sample performance by observability tag;
- confidence and calibration by tag;
- examples where a contextual prediction is correct but lacks local evidence;
- examples where a not-inferable ingredient is confidently predicted from cuisine or plating context;
- inter-reviewer agreement and adjudication procedure for the audit subset.

Do not remove not-inferable labels from the primary target after viewing model results. They express the approved recipe-level task and define an empirical ceiling/ambiguity question. A visible-only secondary analysis may be reported if its subset was frozen from independent review.

## 7. Baselines that diagnose shortcuts

The benchmark should include:

- all-negative predictions;
- global prevalence scores;
- fixed global top-\(k\) or prevalence-threshold prediction;
- cuisine-only or metadata-only prediction where metadata is available;
- a simple visual pretrained model;
- an image-shuffled control for any multimodal or dependency component.

If a structured image model barely beats cuisine-only prediction on contextual labels, the result should be described as prior exploitation rather than fine-grained visual recognition.

## 8. Interpretability

[Grad-CAM](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html), [Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html), and [TCAV](https://research.google/pubs/interpretability-beyond-feature-attribution-quantitative-testing-with-concept-activation-vectors-tcav/) offer complementary attribution and concept-analysis approaches.

Their correct role is diagnosis:

- visualize evidence for selected ingredient logits;
- compare class-specific attention with pooled-head attributions;
- test attribution stability across seeds and mild transformations;
- measure deletion/insertion or masking sensitivity;
- inspect whether text, plates, backgrounds, and garnishes dominate;
- use TCAV-style concepts only with independently curated concept examples.

A heatmap is not proof that the model localized an ingredient. For a contextual or hidden ingredient, broad attention may be semantically consistent. Explanations should be compared with the observability annotations and controlled interventions.

## 9. Shortcut and robustness studies

[Shortcut Learning in Deep Neural Networks](https://www.nature.com/articles/s42256-020-00257-z) provides the general framework: models exploit predictive but unintended cues when they are easier than the desired signal.

High-value local probes include:

- central-dish crop vs full image;
- background/plate attenuation or masking;
- low-resolution and JPEG-compression stress tests;
- color-desaturation or bounded hue perturbations;
- source/cuisine stratification;
- watermark/text-bearing vs clean images;
- nearest-neighbor review in the training set for high-confidence test predictions;
- performance as a function of family size and duplicate-evidence type.

These are diagnostic evaluations, not opportunities to tune on the test set. Define perturbations and select any parameters on validation data.

## 10. Confidence sets and abstention

[Conformal prediction with limited false positives](https://proceedings.mlr.press/v162/fisch22a.html) and [confidence sets for multi-label classification](https://jmlr.org/papers/v22/20-753.html) offer principled directions for set-valued uncertainty. They are secondary because their guarantees depend on exchangeability and a separate calibration sample, assumptions complicated by recipe families and domain shift.

A future conformal study should:

- split calibration families from model-training and final-test families;
- define whether the guarantee concerns false positives, coverage of the full true set, or per-label error;
- report set size and utility, not coverage alone;
- avoid selecting the conformity rule on test outcomes.

For the first benchmark, calibrated scores, a frozen threshold, and confidence-stratified error analysis are sufficient.

## 11. Error taxonomy

Every serious candidate should be reviewed against a fixed sample of errors with categories such as:

- ontology/mapping defect;
- annotation omission or questionable recipe label;
- duplicate/family assignment defect;
- directly visible ingredient missed;
- visually confusable ingredient;
- contextual prior correct;
- contextual prior false positive;
- ingredient not inferable from the image;
- image quality/resolution failure;
- preprocessing/crop removed evidence;
- calibration/threshold-only error.

The taxonomy connects modeling results back to benchmark maintenance. Changes justified by test error review must create a new benchmark version and restart confirmatory evaluation; they must not patch the current test labels in place.

## 12. Reporting artifacts

Each experiment family should retain:

- benchmark, ontology, vocabulary, split, exclusion, and transform identifiers;
- code commit and environment lock;
- checkpoint source and checksum;
- complete configuration and seed;
- validation-selected epoch, threshold, and calibration parameters;
- per-example logits/probabilities and targets;
- per-label metric table with support;
- per-family bootstrap input IDs;
- resource measurements;
- aggregate metrics, intervals, and failure notes.

[Datasheets for Datasets](https://www.microsoft.com/en-us/research/publication/datasheets-for-datasets/), [Data Cards](https://research.google/pubs/data-cards-purposeful-and-transparent-dataset-documentation-for-responsible-ai/), and [Model Cards](https://arxiv.org/abs/1810.03993) provide useful documentation patterns. The principles in [Reproducibility in Machine Learning](https://www.jmlr.org/papers/v22/20-303.html) support reporting all material sources of variation rather than relying on a single checkpoint and score.

## 13. Intended-use boundary

The benchmark measures agreement with normalized Yummly recipe ingredients under the approved ontology. It does not verify the chemical composition of the photographed dish, ingredient quantities, cross-contact, substitutions made by a cook, or allergens. Predictions must not be presented as allergy-safety or clinical nutrition advice.

External validity is also bounded: Yummly-like web food photography, the retained cuisines, and the benchmark vocabulary do not establish performance for household camera feeds, restaurants, other regions, or unseen ingredients. A model card should state these boundaries beside headline scores.
