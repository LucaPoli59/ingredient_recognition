# Ingredient label autoencoder: exploratory feasibility

**Created:** 2026-10-21

**Last updated:** 2026-10-21

**Status:** Research and train-metadata inspection only; no model training performed.

## Question and provisional assessment

Can a compact representation learned from ingredient sets, optionally informed
by cuisine and flavor metadata, improve prediction of all 165 ingredients from
an image alone? This is a label-space representation experiment, distinct from
selecting the existing 59-label subset. A latent code with 32 coordinates still
decodes to 165 ingredient scores; its coordinates are not 32 named ingredients.

The hypothesis is plausible because ingredient combinations are structured.
However, a code optimized to reconstruct recipes may encode information that
cannot be inferred from their photographs. The useful objective is a code that
is both recoverable into ingredients and predictable from images. No reviewed
source establishes that this method will improve this project's benchmark.
For 165 labels, potential statistical regularization matters more than a claimed
extreme-classification speedup; actual compute savings would need measurement.

The reconstruction stage is self-supervised on existing recipe annotations.
The complete image-to-ingredient system remains supervised. It requires no new
manual annotations, but it does not eliminate the need for ingredient targets.

## Local evidence and reproducible audit

- [audit_train.py](audit_train.py) uses the Python standard library and reads
  only `data/input/yummly/train/ingredients_target_v5_metadata.json`.
- [train_audit.json](train_audit.json) contains input/script SHA-256 hashes and
  aggregate counts. No images, validation/test data, checkpoints or GPU were
  accessed. Run from the repository root with
  `python3 src_scratches/label_autoencoder/audit_train.py` to reproduce stdout.
- The train contains 47,965 recipes, 165 targets and 369,837 positive assignments:
  7.71056 labels per recipe, or 4.67307% density. Predicting every label absent
  achieves 95.32693% per-entry accuracy, so that accuracy cannot validate an AE.
- All six numeric flavor dimensions are available for 35,339 recipes (73.67664%).
  Another 3,426 have absent/empty flavor fields, and 9,200 have nonempty incomplete
  or nonnumeric fields. Numeric values are in [0, 1]. Lower/title-case schemas
  coexist; use an explicit ordered list of normalized keys, not dictionary order.
- Flavor completeness varies by cuisine: American 96.20%, Indian 54.40%, Thai
  59.91%. Dropping incomplete recipes would change the population and comparisons.
- The existing [scratch AE](../recipes_encoding/autoencoder.py) is a historical
  starting point, not this experiment: its launcher uses `ingredients_ok`, a
  `topk_` metadata name, input dimension 100, bottleneck 15, GPU training and an
  accuracy metric. Its defaults and epsilon-adjusted binary targets should not
  be carried over. The class structure may be reusable after an explicit revision.
- The existing [flavor helper](../../src/data_processing/images_recipes.py)
  (`_preprocess_flavor`) filters incomplete records and relies on dictionary value
  ordering. A new scratch experiment needs explicit key order and masks instead.

The numeric audit establishes availability, not annotation quality. The exact
generation of the local flavor scores remains unverified; do not describe them
as independent sensory measurements or independent evidence of ingredient truth.

## Targeted literature and evidence matrix

Search date: 2026-10-21. This is a targeted methodological search, not an exhaustive
or systematic review. Existing structured-head and custom-attention research was
checked first. Discovery used web search, then original AAAI, NeurIPS, ACL and
author-hosted papers. No new bibliography or citation registry was created.

Representative queries included `multi label learning deep canonical correlation
autoencoder label embedding`, `Learning Deep Latent Space for Multi-Label
Classification`, `Ranking-Based Autoencoder for Extreme Multi-label
Classification`, `Conditional Principal Label Space Transformation`, and
`food autoencoder ingredients cuisine`. General recipe-generation and food-pairing
results were not treated as evidence for image-only multi-label prediction.

| Source | Inspected evidence and reason for inclusion | Transfer limit |
| --- | --- | --- |
| [Chen and Lin, NeurIPS 2012: Feature-aware Label Space Dimension Reduction](https://proceedings.neurips.cc/paper/2012/hash/d4c2e4a3297fe25a71d030b67eb83bfc-Abstract.html); [author PDF](https://www.csie.ntu.edu.tw/~htlin/paper/doc/nips12cplst.pdf) | Introduction and Sections 2–3 motivate CPLST: account for input features when compressing label space, rather than only label reconstruction. Foundational reason to evaluate predictability. | Its Hamming-loss formulation and generic benchmarks do not establish macro-AP or rare-ingredient gains here. A PCA control is useful but is not itself CPLST. |
| [Yeh et al., AAAI 2017: C2AE](https://ojs.aaai.org/index.php/AAAI/article/download/10769/10628) | Figure 1, method pp. 2839–2841, equations 1–2: feature and label encoders share a latent space; a label decoder maps image/input features back to labels at inference. Table 1 includes image annotation datasets. Closest architectural precedent. | DCCA constraints and a pairwise ranking objective differ from the simpler proposed BCE experiment. Old features, datasets and splits differ from Yummly. We would not call our simplified model a C2AE reproduction. |
| [Wang et al., NAACL 2019: Rank-AE](https://aclanthology.org/N19-1289/); [paper](https://aclanthology.org/N19-1289.pdf) | Sections 2.2–2.3, Figure 1: input/label embedding alignment and output reconstruction; the label encoder is removed at inference. A closer precedent for using MSE alignment than copying CCA. | Section 3 uses different extreme multi-label datasets and ranking measures. Its headline comparisons do not predict a benefit on 165 Yummly labels. Do not infer that BCE prevents shared representations from exploiting correlations. |
| [Ngiam et al., ICML 2011: Multimodal Deep Learning](https://people.csail.mit.edu/khosla/papers/icml2011_ngiam.pdf) | Figure 3 and Sections 3.2/4.3 examine learning with multiple modalities and inference with one. Their discussion also reports cases where joint features or an added modality hurt the single-modality task. Supports evaluating modality availability explicitly. | Audio/video speech experiments, not recipe metadata or ingredients; no transfer of numerical gains. |

The project's existing [structured-head dossier](../../docs/research/topics/experimental_model_candidates/structured_multilabel_head.md)
already covers Inverse Cooking as a food-domain precedent for set dependencies.
Its CVF PDF could not be freshly fetched (HTTP 403), so no new detailed claim is
based on that failed fetch. The Yummly paper's OpenReview copies presented a
browser challenge and the author-hosted PDF timed out; flavor-score provenance
therefore remains open rather than being inferred from search snippets.

These papers support a method family, not a claim of current state of the art or
novelty. No published score is compared numerically with local results.

## Proposed model and use of metadata

Let `y` be the 165-dimensional binary target, `x` the image, `E` the label encoder,
`D` the ingredient decoder and `g` the image encoder plus latent projection.

```text
Recipe branch, training only:  y -> E -> z -> D -> reconstructed y
Image branch, deployment:      x -> g -> z_image -> D -> 165 ingredient logits
```

An illustrative compact network is `165 -> 128 -> k -> 128 -> 165`, with
`k in {16, 32, 64}` as a small proposed scan, not a chosen optimum. Use sigmoid
only for scoring; the decoder emits logits for BCEWithLogitsLoss. Do not use
softmax across ingredients. The latent axes need not have a semantic name.

First learn `E,D` on train ingredient sets. Compare a plain AE with a later mild
denoising extension that masks some input ingredients but reconstructs the
original target; keep the clean-versus-denoising distinction explicit. Match the
ingredient-loss policy across the AE and image-head controls; all class weights
are train-derived. A denoising task tests co-occurrence, not visual recovery.

Then freeze the teacher encoder and initially freeze the decoder while learning
the image projection. The primary image loss should remain

`L_image = BCEWithLogits(D(g(x)), y)`.

An optional, separately tested term is
`lambda * MSE(g(x), stop_gradient(E(y)))`, with compatible latent scale and
train-fitted normalization. Latent MSE alone is not a sufficient prediction
objective: small latent errors can become large ingredient errors after decoding,
and averaging ambiguous latent codes through a nonlinear decoder need not yield
the correct ingredient probabilities. Later decoder fine-tuning is a distinct
ablation, since it changes how much the original label manifold is preserved.

Two metadata designs answer different questions:

1. **Recommended first extension: metadata as auxiliary targets.** Keep `E(y)`
   as the encoder input, and add cuisine classification and flavor regression
   heads from `z`. Optimize ingredient BCE plus separately weighted cuisine CE
   and flavor regression on observed dimensions. This tests whether metadata
   shapes a useful code without giving the encoder another input modality.
2. **Direct version of the user's proposal: metadata as encoder inputs.** Use
   `E(y, cuisine, flavors, masks)`. The image branch must still predict this code
   without those inputs. Evaluate the risk that the teacher encodes details not
   recoverable from pixels; consider modality dropout and image alignment as
   explicit later variants. The decoder must consume only `z`, with no ground-
   truth metadata skip connection at deployment. This is worth testing after
   the ingredient-only baseline, not assuming equivalent to design 1.

For flavors, normalize key casing and use six fixed columns. Mask absent values
in the auxiliary loss, normalize by observed entries, and retain every recipe.
In an input-based variant, use train-derived imputation plus explicit missingness
masks. Zero is a real observed value and cannot alone mean missing. Treat cuisine
as a categorical variable rather than a numeric code. Scale each loss per task
before choosing its coefficient; 165 binary outputs, one cuisine and six flavor
values should not receive accidental relative weighting from tensor dimensions.

Metadata-derived targets are training information, not leakage by themselves.
Using validation/test labels to fit the AE, vocabularies, imputations, weights
or correlations would contaminate the held-out evaluation, even if called
unsupervised. Metadata unavailable at inference must never enter validation/test
image predictions. A reconstruction diagnostic with true labels is explicitly
a different task and must not be reported as image prediction.

## Small, discriminating experiment

**Stage A — label feasibility.** Fit a simple linear/PCA label compression control
and the small AE on train only. Use existing validation solely for selection and
held-out reconstruction diagnostics. Measure ingredient macro-AP, micro-F1 with
validation-selected thresholds, per-label AP and frequency-sliced recall.
PCA explained variance and a visually tidy latent plot are secondary diagnostics.
Reconstruction is not a mathematical upper bound on subsequent image performance.

**Stage B — image usefulness.** Once feature extraction is appropriate for the
available compute, use one fixed pretrained visual encoder and the same cached
features, examples, preprocessing, loss policy and tuning budget for all arms.
A frozen-feature experiment measures head quality under that representation;
it cannot be compared as an equal-budget end-to-end benchmark with current runs.

| Arm | Purpose |
| --- | --- |
| Image features -> direct 165-output linear head | Standard image baseline. |
| Image features -> bottleneck -> decoder, random initialization | Same inference architecture as the AE arm; isolates bottleneck/capacity from AE pretraining. |
| Image features -> bottleneck -> pretrained ingredient AE decoder | Tests the usefulness of ingredient reconstruction pretraining; record decoder frozen/fine-tuned policy. |
| Previous arm with metadata-regularized AE | Tests the additional training information from cuisine/flavors. |

If the metadata arm wins, add a direct image-head model with the same auxiliary
metadata losses to distinguish an autoencoder benefit from extra supervision.
Record trainable parameters, training steps and head compute; do not hide extra
AE pretraining cost. Keep latent matching, denoising and decoder fine-tuning as
separate factors rather than changing all three in the first comparison.

Report full-vocabulary metrics first. A secondary analysis can use the already
fixed 59-label projection, keeping exactly the same rows/columns across models;
never compare full-165 macro-AP directly with selected-59 macro-AP as evidence of
improvement. Flavor completeness, cuisine and train-defined frequency slices
help detect who benefits or loses. Compare against a train global-prevalence
baseline. Ground-truth cuisine priors are diagnostic oracles, not image-only
competitors. Shuffling image features within the validation evaluation is a
bounded diagnostic of image dependence, not proof of causal visual grounding.

Positive evidence would be better decoded image macro-AP without a material
micro-F1 or rare-label regression, under matched conditions. Better recipe
reconstruction alone is insufficient. Improvements limited to common cuisine
combinations merit a narrower interpretation. If the randomly initialized
bottleneck matches the AE, pretraining has not demonstrated added value. A single
pilot seed permits an exploratory observation, not a training-stability claim.
Any uncertainty estimate should use the existing exact-image grouping policy.

## Boundaries and next decision

The useful next experiment is ingredient-only compression with a small latent
scan, followed by the matched frozen-feature comparison. A VAE introduces a KL
trade-off and possible additional reconstruction loss; generative sampling is
not required to answer this first question. It remains a later alternative.

This scratch investigation does not alter the
[benchmark methodology](../../docs/project_objective/model_comparison_methodology.md),
selected vocabulary, active HPO studies or training sources. No AE, feature
extraction, CNN training, dependency installation or test evaluation was run.
The question remains open for discussion; this is not an adopted thesis-method
decision or a completed experiment. Reviewed findings can later be promoted
under the project's [documentation policy](../../docs/README_DOCS_ORGN.md).
