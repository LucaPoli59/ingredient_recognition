# Data Processing and Augmentation

Created: 2026-08-02  
Updated: 2026-08-02

## 1. Data processing is part of the model

The benchmark does not inherit a trustworthy target vector from the legacy dataset. It constructs a versioned semantic object from raw ingredient lines. That construction determines which patterns a visual model is rewarded for learning.

The required order is:

1. preserve raw recipe and image evidence;
2. normalize ingredient lines with traceable rules;
3. adjudicate ambiguous mappings and exclusions;
4. construct recipe-family groups;
5. split groups deterministically;
6. derive the vocabulary from training support and semantic gates;
7. fit model preprocessing and augmentation on the training path only.

Changing the order can leak validation/test information into ontology thresholds, vocabulary selection, or augmentation statistics.

## 2. Ingredient normalization and ontology design

### 2.1 Local objective

Each raw ingredient line should produce a trace, not just a label:

```text
raw line
  -> normalized text
  -> parsed quantity/unit/preparation spans
  -> candidate food spans
  -> alias or ontology rule
  -> canonical local ingredient ID(s)
  -> decision status and provenance
```

Canonical IDs must remain stable when display names change. Rules and adjudications need versions so a target vector can be reproduced from raw inputs.

### 2.2 Rule-first, model-assisted processing

[TASTEset](https://arxiv.org/abs/2204.07775) provides a corpus and model direction for structured ingredient parsing. A 2024 [recipe named-entity-recognition study](https://aclanthology.org/2024.lrec-main.406/) compares fine-tuned transformer/spaCy systems with few-shot large language models and supports domain-trained parsing rather than assuming generic prompting is sufficient.

The recommended pipeline is deterministic at the acceptance boundary:

- Unicode and whitespace normalization;
- explicit quantity, unit, punctuation, and preparation handling;
- token/phrase-boundary alias matching;
- curated multi-ingredient phrase rules where decomposition is semantically valid;
- parser or language-model suggestions only as review candidates;
- human adjudication for unresolved high-impact lines;
- regression fixtures for every accepted rule.

An opaque model should not silently assign benchmark labels. If a learned parser is used, record its model/version, output, confidence, and the deterministic rule or review decision that accepted the mapping.

### 2.3 External ontologies as references

[FoodOn](https://pmc.ncbi.nlm.nih.gov/articles/PMC6550238/) supplies a broad food ontology; [FoodEx2](https://www.efsa.europa.eu/en/data/data-standardisation) provides an official European food classification and description system; [USDA FoodData Central](https://fdc.nal.usda.gov/api-guide/) exposes food records and identifiers through an official API.

These resources can help with synonyms, taxonomic relations, and identifier crosswalks. None should be imported wholesale as the benchmark vocabulary:

- their granularity may not match image inferability;
- regulatory, nutritional, and recipe uses distinguish different concepts;
- coverage and naming conventions vary by geography and purpose;
- a hierarchy may collapse distinctions the thesis intends to measure.

Maintain local canonical IDs and optional external cross-references. Record mapping type (`exact`, `narrower`, `broader`, `related`) rather than claiming every crosswalk is equivalent.

### 2.4 Multi-component and preparation-sensitive lines

Lines such as sauces, mixes, garnishes, and “for serving” ingredients require an explicit policy. A parser must not invent constituent ingredients that are not stated, while a benchmark ontology should avoid treating arbitrary brand/product strings as stable visual classes.

Recommended statuses:

- `mapped`: supported canonical ingredient assignment;
- `excluded_non_food`: equipment, instruction fragments, or invalid text;
- `excluded_out_of_scope`: concept deliberately outside target semantics;
- `composite_kept`: a semantically meaningful composite class;
- `ambiguous_review`: multiple defensible interpretations;
- `unmapped`: no approved mapping.

No generic `<UNK>` output should enter the label vector. Coverage is instead reported from these statuses.

## 3. Vocabulary construction

Vocabulary selection is a benchmark decision, not a frequency-only filter. The binding policy combines semantic quality with training and evaluation support. For every candidate class, record:

- canonical ID and display name;
- alias/rule coverage;
- train/validation/test positive counts after the group split;
- number of recipe families, not only images;
- observability review evidence;
- exclusion or merge rationale;
- ontology version.

The headline vocabulary should be frozen before full-test evaluation. Exploratory tail results may use a separately named vocabulary but must never silently change the denominator of macro metrics.

## 4. Image integrity and exclusions

Known invalid image groups must be excluded through a manifest with reason codes. Additional checks should be deterministic and cacheable:

- decode success and channel count;
- dimensions and extreme aspect ratios;
- file hash and perceptual hash;
- near-uniform or placeholder-like content;
- image–recipe linkage validity;
- optional embedding-neighbor candidates for review.

Automated anomaly scores should propose candidates, not delete data without a recorded rule or review decision.

## 5. Duplicate and family discovery

### 5.1 Why exact hashes are insufficient

The local audit already shows exact duplicates crossing historical splits, while a broader graph finds additional train/evaluation relations. Crops, recompression, resized copies, near-identical plating photographs, and duplicated recipe records can evade file hashes.

[SemDeDup](https://arxiv.org/abs/2303.09540) shows how semantic duplicate discovery can scale with learned embeddings, and the [official implementation](https://github.com/facebookresearch/SemDeDup) documents the method. Work on [data leakage in visual datasets](https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Ramos_Data_Leakage_in_Visual_Datasets_ICCVW_2025_paper.html) and the duplicate-cleaned [ciFAIR evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321059/) reinforce that duplicate contamination can distort model comparisons.

### 5.2 Evidence graph

Construct a graph whose nodes are recipe records and whose edges have typed evidence:

- exact file/content hash;
- perceptual-hash distance within a conservative threshold;
- normalized source URL or source identifier;
- normalized recipe-title and ingredient-set similarity;
- high image-embedding similarity;
- manual relation decision.

Only high-precision rules should auto-link. Generic embedding similarity is a candidate generator because different legitimate dishes can be visually similar. Thresholds must be selected from reviewed positive and negative pairs, not convenience.

Connected components or an explicitly documented clustering policy define recipe families. Store both the raw evidence edges and the final family assignment so split decisions remain auditable.

## 6. Group-aware multi-label splitting

Plain random splitting is invalid once related records form families. A group allocator should optimize several constraints together:

- deterministic 80/10/10 family assignment;
- zero family overlap;
- overall record-count ratio;
- per-label positive support, especially evaluation minimums;
- cuisine/source balance where it prevents obvious shift;
- invalid/excluded records removed before allocation.

[Iterative stratification](https://proceedings.mlr.press/v74/szyma%C5%84ski17a.html) motivates preserving multi-label evidence, including label-pair structure. It does not directly solve grouped splitting. The project needs a deterministic group-level optimizer inspired by those objectives, with a fixed seed, tie-breaking rules, and a report of infeasible constraints.

Do not repeatedly tune the split to make a favored model look stable. Freeze it based on data criteria before serious model comparison.

## 7. Image preprocessing

### 7.1 Geometry

The legacy direct resize to `224 × 224` changes aspect ratio for the common `3:2` images. Because 28.4% of audited images have at least one side below 224 pixels, aggressive crop and upsample policies also deserve scrutiny.

Primary comparison:

1. aspect-preserving resize of the long or short side plus padding to the model canvas;
2. aspect-preserving resize plus bounded random/resized crop during training and deterministic center crop during evaluation;
3. legacy square warp only as an ablation.

Padding color/mode and valid-region masks should be recorded. If a transformer accepts variable resolution, native-aspect inference can be a later comparison; it should not make evaluation preprocessing image-dependent in undocumented ways.

### 7.2 Normalization contract

Each backbone must expose exactly one authoritative preprocessing specification: input range, channel order, resize/crop, interpolation, mean, and standard deviation. Dataset code should not infer normalization from whether a transform object happens to be a list.

Tests should verify:

- output shape, dtype, and finite range;
- normalization applied exactly once;
- train/eval geometric differences are intentional;
- a saved visualization can approximately invert normalization;
- checkpoint-specific requirements are respected.

### 7.3 Resolution

Ingredient evidence can be small, but high resolution increases ViT token memory quadratically. Select resolution by a controlled curve on one frozen model (for example 224 vs one moderately larger resolution), reporting both accuracy and resource cost. Avoid tuning a separate resolution for every architecture unless the checkpoint requires it.

## 8. Augmentation evidence and risks

[RandAugment](https://proceedings.neurips.cc/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html) reduces augmentation search to a small set of global parameters. [TrivialAugment](https://openaccess.thecvf.com/content/ICCV2021/html/Muller_TrivialAugment_Tuning-Free_Yet_State-of-the-Art_Data_Augmentation_ICCV_2021_paper.html) simplifies further to a tuning-free random operation. [AugMix](https://arxiv.org/abs/1912.02781) targets robustness and uncertainty under corruptions. These methods are general image-classification evidence, not automatic food-domain prescriptions.

### 8.1 Recommended baseline policy

- aspect-preserving geometric transform;
- modest random crop with a high retained-area floor;
- horizontal flip unless a reviewed slice reveals directional artifacts;
- light brightness/contrast/saturation variation;
- optional mild blur or compression to reflect web-image variation;
- no validation/test stochasticity.

The exact ranges should be fixed from a contact sheet review and a small ablation, not inherited blindly from ImageNet recipes.

### 8.2 Risk matrix

| Augmentation | Potential benefit | Ingredient-specific failure mode | Initial status |
|---|---|---|---|
| Horizontal flip | cheap invariance | usually low risk; may flip text/watermarks | baseline candidate |
| Random resized crop | scale/local robustness | removes only visible ingredient or dish context | conservative only |
| Color jitter | lighting robustness | changes doneness, sauce, produce-color cues | light only |
| Blur/compression | web-image robustness | erases small garnish/texture evidence | mild ablation |
| Random erasing | occlusion robustness | hides small evidence while label remains positive | ablation |
| Rand/TrivialAugment | diverse regularization | includes semantically harsh operations | filtered ablation |
| AugMix | corruption robustness/calibration | mixtures may alter food appearance | robustness ablation |
| MixUp | regularization | area-weighted soft labels do not equal recipe semantics | not baseline |
| CutMix | local compositing | ingredient label is not proportional to pasted area | not baseline |
| SpliceMix | multi-label-tail composition | synthetic co-occurrence/context can be implausible | later ablation |
| Generative synthesis | class/tail expansion | ingredient fidelity and domain artifacts unverified | defer |

[Random Erasing](https://ojs.aaai.org/index.php/AAAI/article/view/7000), [MixUp](https://openreview.net/pdf?id=r1Ddp1-Rb), and [CutMix](https://arxiv.org/abs/1905.04899) are influential regularizers. Their label assumptions matter here. If two recipe images are combined, the union or area-weighted mixture of recipe-level label vectors may contain hidden ingredients with no visual support and visible ingredients removed by the composite.

[SpliceMix](https://arxiv.org/abs/2311.15200) is designed for long-tailed multi-label recognition and is a more relevant sample-mixing lead. It still needs a food-specific plausibility test and should follow, not precede, the clean baseline.

### 8.3 How to decide whether an augmentation survives

Evaluate each policy on a fixed model and at least three seeds. Retain it only if it provides a worthwhile gain without unacceptable damage to:

- label-macro AP and global-threshold micro-F1;
- direct-observability ingredients;
- rare-label and small-object slices;
- calibration error and confidence distributions;
- robustness to resolution/compression shifts;
- training stability and throughput.

A small aggregate gain that comes from amplifying contextual priors while degrading directly visible labels is not an unqualified improvement.

## 9. Synthetic and external data

External food datasets can support representation pretraining or diagnostic transfer, but they must not be casually merged with Yummly labels. Required checks include license, provenance, overlap with Yummly images/recipes, target-semantics alignment, ontology crosswalk quality, and source-domain imbalance.

Synthetic images have an additional verification problem: a prompt or generated caption is not proof that each ingredient is visually or compositionally present. Any future synthetic-data study needs human ingredient-fidelity review, provenance fields, a real-only evaluation set, and an ablation showing that improvements are not generator-style recognition.

## 10. Recommended data ablations

After the benchmark is frozen, the minimum data-side experiment set is:

1. aspect-preserving pad vs controlled crop vs legacy warp;
2. no augmentation vs conservative food-safe augmentation;
3. conservative policy vs one filtered TrivialAugment/RandAugment policy;
4. optional AugMix robustness study;
5. no sample mixing vs one food-audited mixing method only if tail performance remains a bottleneck.

Run these on one fixed, affordable backbone before applying the winning policy to the wider model shortlist.

