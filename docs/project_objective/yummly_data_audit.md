# Yummly data audit

**Created:** 2026-08-02  
**Last updated:** 2026-08-02

## Purpose and audit boundary

This document establishes what data the Ingredient Recognition project actually uses, how it differs from the published Yummly-66K dataset, how the repository transforms it, and which limitations affect the research objective.

The audit covers:

- all raw Yummly metadata, recipe, and image files present locally;
- all processed train, validation, and test metadata;
- all 65,146 processed images, decoded and hashed individually;
- historical and active preprocessing code;
- target distributions, schema quality, duplicate leakage, prior baselines, and visual samples.

Recipes1M and the other dataset directories are intentionally excluded because they are not used by the active project objective.

## Reproducibility

Run the complete audit from the repository root:

```powershell
python src_scratches/data_anlysis/yummly_audit.py
```

Run the decision-oriented second pass, including historical target reproduction, dual perceptual hashes, and grouping policies, with:

```powershell
python src_scratches/data_anlysis/yummly_deep_audit.py
```

For a metadata-only refresh that preserves a previously generated image section:

```powershell
python src_scratches/data_anlysis/yummly_audit.py --skip-images
```

The script produces:

- `src_scratches/data_anlysis/outputs/yummly_audit.json` — the complete machine-readable audit;
- `src_scratches/data_anlysis/outputs/ingredient_frequency.csv` — label support and split frequencies;
- `src_scratches/data_anlysis/outputs/yummly_sample_contact_sheet.jpg` — a deterministic cuisine-stratified visual sample.

The second pass produces:

- `src_scratches/data_anlysis/outputs/yummly_deep_audit.json` — target reproduction, collision, taxonomy, shortcut, support, and grouping evidence;
- `src_scratches/data_anlysis/outputs/target_review.csv` — one row per current or historical target candidate;
- `src_scratches/data_anlysis/outputs/duplicate_group_review.csv` — prioritized duplicate-family review data;
- `src_scratches/data_anlysis/outputs/duplicate_group_review.jpg` — a visual sheet of the largest exact and perceptual groups.

All quantitative values below are derived from these outputs unless an external source is explicitly cited.

## Dataset identity and provenance

The local raw tree contains exactly 66,615 image files and 66,615 cuisine-specific metadata records with these counts:

| Cuisine | Raw records/images | Published Yummly-66K count |
|---|---:|---:|
| American | 13,262 | 13,262 |
| Italian | 9,401 | 9,401 |
| Mexican | 7,960 | 7,960 |
| French | 6,173 | 6,173 |
| Indian | 5,470 | 5,470 |
| Thai | 5,282 | 5,282 |
| Chinese | 5,251 | 5,251 |
| Greek | 4,998 | 4,998 |
| Japanese | 4,804 | 4,804 |
| Spanish | 4,014 | 4,014 |
| **Total** | **66,615** | **66,615** |

The exact agreement identifies the source as Yummly-66K with high confidence. The source paper describes 66,615 recipes, ten cuisines, fourteen course attributes, food images, and a preprocessed vocabulary of 2,416 ingredients. It also describes the dataset as the combination of an earlier 27,638-item collection and a second crawled component.

The local repository does not contain a source checksum manifest or complete generation record. These must be added to make future benchmark builds independently identifiable and reproducible.

## Local processing lineage

```text
Yummly-66K raw metadata and images
66,615 metadata + 66,615 images
        |
        | historical ingredient normalization, frequency filtering,
        | approximate merging, minimum 3 normalized labels
        v
data/raw_input/yummly/recipes/*.json
65,159 recipes in an 80/10/10 historical split
        |
        | recipes_merge.py: concatenate historical files
        v
all_recipes.json
65,159 recipes
        |
        | sort_recipes_as_img.py: reorder by cuisine and source id
        v
recipes_sorted.json
65,159 recipes
        |
        | creation.py: random split again, seed 42,
        | configured 84/8/8, copy available images
        v
data/input/yummly/{train,val,test}
65,146 final records/images
```

### Retention

| Stage | Records | Change |
|---|---:|---:|
| Original metadata | 66,615 | — |
| Historical normalized recipes | 65,159 | −1,456 |
| Final processed dataset | 65,146 | −13 |

The historical script in `prev_attempts/attempt1/preprocessing_v2.py` uses a minimum of three retained normalized ingredients and an alleged minimum ingredient frequency. Its 80/10/10 split sizes exactly match the three raw recipe JSON files, which establishes close lineage.

Direct execution of its pure normalization functions on all 144,010 unique original ingredient strings proves that it is not the exact current generator:

- 174 labels survive the pre-merge threshold, compared with 182 stored labels;
- only 165 labels are shared;
- 52,388 of 65,146 processed recipes, 80.42%, have an exact pre-merge label-set match;
- the regenerated-versus-stored micro F1 is 0.978, showing similarity without identity;
- the script considers 65,170 source records eligible before similarity merging;
- four checked-in recipe IDs are already ineligible before the merge, which only collapses labels and cannot restore them.

The threshold of 50 counts distinct raw strings mapping to a normalized value, not recipe occurrences. The similarity stage consumes unordered multiprocessing results, and no saved mapping, dependency lock, checksum, or execution manifest exists. The present `ingredients_ok` field is therefore a non-reproducible legacy artifact.

Of the final thirteen omitted records, eleven source recipe IDs currently lack a corresponding raw image. Two source images exist but were not copied, plausibly because of historical path-length or copy-time behavior; the repository contains no run log that proves the cause.

### Split construction

The final split is:

| Split | Records | Fraction |
|---|---:|---:|
| Train | 54,724 | 84.00% |
| Validation | 5,210 | 8.00% |
| Test | 5,212 | 8.00% |
| **Total** | **65,146** | **100%** |

The split uses `sklearn.model_selection.train_test_split` with seed 42. It is not grouped by recipe family, name, ingredient list, image hash, or image similarity, and it is not explicitly stratified. Observed cuisine drift is small in validation and modest in test, but balanced marginals do not prevent duplicate leakage.

The historical 80/10/10 split has no operational meaning after the files are merged and split a second time.

## Processed schema

Every final record contains the following fields:

| Field | Main type | Meaning | Audit note |
|---|---|---|---|
| `id` | integer | New sequential project ID | Assigned during image copying |
| `old_id` | string | Source recipe identifier | Unique in the final data |
| `image` | string | Split-local JPEG filename | Includes encoded cuisine text |
| `name` | string | Recipe title | Many canonical duplicates |
| `cuisine` | string | One of ten cuisine labels | Complete but not always visually credible |
| `ingredients` | list of strings | Raw/source ingredient lines | Quantities and preparation text may be present |
| `ingredients_ok` | list of strings | Locally normalized targets | Active multi-label target |
| `course` | list or string | Course metadata | 6,152 blank strings violate the dominant schema |
| `flavors` | dict, empty dict, or null | Six Yummly flavor dimensions | Incomplete and casing-inconsistent |

No final record is missing a required image-task field, and no `ingredients_ok` list is empty. This apparent completeness partly results from filtering out recipes with fewer than three normalized labels.

## Cuisine distribution

| Cuisine | Final records | Share |
|---|---:|---:|
| American | 12,881 | 19.77% |
| Italian | 9,107 | 13.98% |
| Mexican | 7,883 | 12.10% |
| French | 5,975 | 9.17% |
| Indian | 5,428 | 8.33% |
| Thai | 5,209 | 8.00% |
| Chinese | 5,187 | 7.96% |
| Greek | 4,852 | 7.45% |
| Japanese | 4,655 | 7.15% |
| Spanish | 3,969 | 6.09% |

The largest cuisine has 3.25 times as many records as the smallest. Cuisine is strongly associated with ingredient priors, so it can become a shortcut even when not explicitly passed to the model.

## Ingredient representation

### Raw versus normalized data

The final data contains:

- 740,847 raw ingredient-line occurrences;
- 141,272 unique raw ingredient-line strings;
- 594,896 normalized target occurrences;
- 182 unique normalized target labels.

Raw recipes contain a mean of 11.37 ingredient lines. The normalized target contains a mean of 9.13 labels, with a median of 9, a 95th percentile of 16, a minimum of 3, and a maximum of 42.

There are 9,746 repeated raw ingredient lines within recipes, often arising from source duplication such as “salt and pepper to taste” appearing twice. The normalized lists contain no repeated label within a record because the historical process deduplicated them.

### Most frequent labels

| Rank | Label | Recipes | Prevalence |
|---:|---|---:|---:|
| 1 | `salt` | 40,440 | 62.08% |
| 2 | `oil` | 35,707 | 54.81% |
| 3 | `pepper` | 31,692 | 48.65% |
| 4 | `garlic` | 29,125 | 44.71% |
| 5 | `onion` | 27,851 | 42.75% |
| 6 | `cheese` | 19,468 | 29.88% |
| 7 | `sugar` | 17,202 | 26.41% |
| 8 | `tomato` | 15,842 | 24.32% |
| 9 | `chicken` | 15,022 | 23.06% |
| 10 | `egg` | 14,970 | 22.98% |
| 11 | `butter` | 13,872 | 21.29% |
| 12 | `water` | 12,492 | 19.18% |
| 13 | `liquor` | 12,311 | 18.90% |
| 14 | `flour` | 11,591 | 17.79% |
| 15 | `lemon` | 9,680 | 14.86% |

The least frequent label, `serrano chiles`, occurs 63 times. The most-to-least support ratio is approximately 642:1. There are no train/validation/test OOV labels: every one of the 182 labels occurs in every split.

### Vocabulary truncation

Using labels ranked by training frequency:

| Vocabulary size | Label-occurrence coverage | Recipes retaining ≥1 label | Recipes retaining ≥3 labels |
|---:|---:|---:|---:|
| 20 | 58.77% | 64,625 | 57,970 |
| 50 | 82.26% | 65,101 | 62,786 |
| 100 | 95.51% | 65,144 | 64,584 |
| 182 | 100% | 65,146 | 65,146 |

A top-100 vocabulary would preserve most occurrences and almost all recipes, but frequency alone is not a sufficient taxonomy rule. It would retain corrupted frequent labels and might discard visually meaningful rare ingredients.

### Support reliability

The legacy split gives:

- 171 labels with at least 100 training positives;
- 150 labels with at least 200 training positives;
- 147 labels with at least 20 positives in both validation and test;
- 104 labels with at least 50 positives in both validation and test;
- 11 labels with fewer than 100 training positives.

The tail is too small for stable per-label threshold selection or precise recall estimates. The regenerated benchmark will use at least 200 training recipes for headline labels and target at least 20 positives in each evaluation split. Rarer reviewed concepts may remain as an explicitly exploratory tail.

## Target-generation defects

### Unbounded substring matching

The historical preprocessing uses conditions equivalent to `substring in ingredient` without enforcing word boundaries. Some later rules can overwrite earlier simplifications. This creates deterministic lexical collisions.

The strongest example is `liquor`:

1. an ingredient containing `ginger` is simplified to `ginger`;
2. the liquor alias list contains `gin`;
3. `gin in ginger` evaluates true;
4. the output becomes `liquor`.

Observed evidence:

- `liquor` occurs in 12,311 recipes;
- only 2,781 of those contain a recognized alcohol term;
- 9,422 contain ginger;
- 8,570 contain ginger and no recognized alcohol term;
- literal token-level support for the word `liquor` is approximately 0.1%.

Systematic host-token inspection found a wider set of conservative signatures. A record is counted only when the suspicious source token occurs and no standalone legitimate target token or reviewed synonym appears elsewhere in that recipe.

| Target removed in sensitivity test | Conservative candidate positives | Principal false host tokens |
|---|---:|---|
| `liquor` | 9,195 | ginger, portobello, crumb variants |
| `sage` | 1,864 | sausage |
| `tea` | 1,258 | steak, teaspoon variants |
| `egg` | 832 | eggplant, veggies |
| `butter` | 812 | butternut, buttermilk |
| `apple` | 684 | pineapple |
| `lemon` | 642 | lemongrass |
| `beans` | 627 | bean sprouts |
| `radish` | 292 | horseradish |
| `water` | 211 | watercress |
| `pear` | 183 | pearl, spears |
| `pepper` | 179 | pepperoni, peppermint |
| `oil` | 167 | boil variants |
| `grape` | 139 | grapefruit |
| Other audited targets | 255 | creamy→cream, acorn→corn, broccolini→broccoli, cantaloupe/jackfruit→cheese, coating→oat, licorice→rice |

In total, 15,684 recipes, 24.08% of the processed dataset, carry at least one audited conservative collision signature. A sensitivity transformation that removes these candidate positives and applies only the obvious alias merges changes 20,154 recipes, reduces positive labels by 17,530 or 2.95%, lowers mean cardinality from 9.13 to 8.86, and leaves 189 records below the historical minimum of three labels.

This is not a corrected ground truth or an exact noise-rate estimate. Some concepts, such as bean sprouts versus beans or broccolini versus broccoli, depend on the final ontology. The analysis demonstrates that repair-in-place would require extensive adjudication and supports regeneration from the raw lines.

### Taxonomy fragmentation

The 182-label vocabulary mixes broad categories, specific ingredients, aliases, morphological variants, and preparation forms. Examples include:

- `celery`, `celery ribs`, `celery stalks`, and `stalks celery`;
- `arugula` and `baby arugula`;
- `leek` and `leeks`;
- `lime` and `limes`;
- `raspberry` and `raspberries`;
- `strawberry` and `strawberries`;
- `cherry` and `cherries`;
- `garam masala` and `garam masala powder`;
- `tomato` and `tomate`;
- broad `sauce` alongside named sauces.

At the same time, many cheese types collapse into `cheese`, many oils collapse into `oil`, and different preparations of meat collapse into broad animal categories. The taxonomy therefore has inconsistent granularity.

The eleven obvious alias groups contain fourteen redundant legacy labels. Applying only those noncontroversial merges would reduce 182 labels to 168. This numerical coincidence is not a target-size recommendation: the final vocabulary must be rebuilt from semantic rules and training support.

### Filtering bias

The historical process removes recipes with fewer than three recognized normalized ingredients. The resulting benchmark excludes recipes that are simple, poorly parsed, or composed mostly of rare ingredients. The minimum target cardinality of three is a preprocessing guarantee, not a natural property of food recipes.

### Closed-vocabulary encoder artifact

`MultiLabelBinarizerRobust` learns all 182 train labels, appends `<UNK>`, and then transforms validation and test with the same mapping. Because both evaluation splits contain no OOV label, `<UNK>` is always negative. The model nevertheless exposes 183 outputs, which changes loss and metric aggregation and has no semantic value in this benchmark. Its positive weight becomes zero because the class has no positives. The decision is to remove `<UNK>` from model outputs before new experiments.

## Image audit

### Integrity and format

All 65,146 referenced image files:

- exist at their expected split-local path;
- decode successfully;
- are JPEG;
- are RGB;
- have no extra unreferenced JPEG peer in their split directory.

There are no corrupt or missing images in the final processed tree.

### Resolution and geometry

| Property | Value |
|---|---:|
| Median width | 360 px |
| Median height | 240 px |
| Median aspect ratio | 1.5 |
| Landscape images | 65,130 |
| Square images | 14 |
| Portrait images | 2 |
| At least one side below 224 px | 18,504 (28.40%) |

The three dominant dimensions are:

- 360×240: 44,616 images (68.49%);
- 300×200: 9,977 images (15.31%);
- 250×167: 4,239 images (6.51%).

Most images are low-resolution 3:2 landscapes. The default base transformation applies `Resize((224, 224))`, which changes the aspect ratio directly. ImageNet-weight and DINO-specific transformations instead resize and crop. Model comparisons can therefore conflate backbone quality with preprocessing differences unless transformations are controlled.

### Visual inspection

The stratified contact sheet confirms:

- large variation in plating, crop, background, lighting, and camera viewpoint;
- strong dish-level cues but weak evidence for many individual ingredients;
- cooked mixtures and sauces that occlude components;
- occasional cuisine assignments that are not credible from the dish name or appearance;
- repeated branded, blank, or generic placeholder images;
- source-specific presentation styles that a model may exploit.

The contact sheet is an inspection aid, not a statistically random quality estimate. Formal removal decisions require a flagged-sample review protocol.

Conservative review of the largest exact-image groups has already confirmed eight exclusion groups covering 84 processed records. They contain a Cooking Light advertisement, publisher marks, a BBC logo, a featureless silhouette, and generic place-setting images shared by unrelated recipes. This is a verified minimum, not an exhaustive estimate; the remaining groups and singleton images still require a review manifest.

## Duplicate and leakage audit

### Metadata duplication

| Duplicate key | Duplicate groups | Groups crossing split boundaries |
|---|---:|---:|
| Canonical recipe name | 5,828 | 2,148 |
| Normalized label set | 4,365 | 1,450 |
| Normalized raw ingredient list | 2,059 | 613 |
| Canonical name + raw ingredient list | 1,795 | 546 |
| `old_id` | 0 | 0 |

Unique source IDs do not imply independent examples. Yummly contains multiple IDs for repeated or syndicated recipes.

### Exact image duplication

SHA-256 hashing of every processed image found:

- 1,113 duplicate hash groups;
- 2,406 records in those groups (3.69% of the dataset);
- 355 duplicate groups crossing split boundaries;
- a largest exact duplicate group of 25 records;
- 78 duplicate groups spanning multiple cuisine labels;
- 205 duplicate groups with multiple target label sets;
- 512 records in target-conflicting image groups.

Evaluation leakage against train is:

| Evaluation split | Records whose exact image occurs in train | Split share | Same label set as a train duplicate | Same canonical name as a train duplicate |
|---|---:|---:|---:|---:|
| Validation | 185 | 3.55% | 144 | 145 |
| Test | 171 | 3.28% | 140 | 146 |

The target-conflicting image groups contain at least 2,395 per-label disagreements when each group is assigned its within-group majority label vector. This is direct evidence that the present image-only supervision is not deterministic.

Some large duplicate groups represent advertisements or source placeholders rather than duplicate food photographs. Such records provide no dish evidence and can teach source shortcuts. In contrast, legitimate repeated photographs of the same recipe should be grouped into one split rather than automatically removed.

### Perceptual duplicate candidates

Simple difference hashing finds 1,890 repeated-hash candidate groups covering 4,056 records, with 592 groups crossing splits. Requiring exact agreement of an independently computed pHash and dHash yields 1,809 higher-precision groups; 775 have membership that is not identical to one exact-byte group and therefore add resized, recompressed, or otherwise non-byte-identical candidates.

These remain candidates. Hashes can agree for visually simple placeholders or compositionally similar images, so ambiguous groups require manual confirmation before becoming a split constraint or exclusion.

### Recipe-family grouping candidate

The selected high-precision graph unions:

- exact SHA-256 images;
- matching dHash and pHash candidates;
- identical normalized raw ingredient-line multisets;
- identical canonical names with raw-line Jaccard similarity of at least 0.8.

It intentionally excludes label-set equality and name equality alone. On the current 65,146 records it creates 61,851 components, places 6,088 records in multi-record components, and has a maximum component size of 27.

The current split cuts 842 of these components. In consequence, 438 validation records (8.41%) and 413 test records (7.92%) are grouped with at least one training record. These figures are broader candidate family contamination, not confirmed image duplication alone. Their small component sizes show that a grouped split is feasible without discarding large portions of the dataset.

## Course and flavor metadata

### Course

`course` is a list for 58,994 records and a blank string for 6,152 records (9.44%). Treating the field uniformly as an iterable can silently process a blank string rather than a list.

The most frequent valid course annotations are Main Dishes, Desserts, Lunch, Salads, Appetizers, Side Dishes, and Soups. Recipes can have multiple course labels, with a maximum of five.

### Flavors

Only 47,447 records (72.83%) contain all six flavor dimensions as numeric values. Another 5,140 records (7.89%) are null or empty, and 12,559 (19.28%) have a nonempty but incomplete/nonnumeric flavor representation. Key casing differs between source subsets (`Sweet` versus `sweet`, for example), and one record lacks one dimension.

Flavors are not part of the active objective. Any future auxiliary use requires schema normalization and missingness handling.

## Priors and metric implications

The dataset has 182 labels and a mean of 9.13 positives per recipe, for a label density of approximately 5.02%.

On the current test split:

| Baseline | Micro precision | Micro recall | Micro F1 | Label accuracy | Hamming loss |
|---|---:|---:|---:|---:|---:|
| All negative | 0.000 | 0.000 | 0.000 | 0.9498 | 0.0502 |
| Global top 5 | 0.504 | 0.276 | 0.357 | 0.9501 | 0.0499 |
| Global top 9 | 0.396 | 0.390 | 0.393 | 0.9395 | 0.0605 |
| Cuisine top 5 | 0.546 | 0.299 | 0.387 | 0.9524 | 0.0476 |
| Cuisine top 9 | 0.462 | 0.455 | 0.459 | 0.9461 | 0.0539 |

The all-negative result proves that label accuracy is unsuitable as a headline metric. The cuisine-prior results show that metadata correlations alone are strong. An image model must beat frequency baselines and be analyzed for cuisine shortcuts.

Threshold-free per-label analysis reinforces this conclusion. A classifier that assigns each label only its training prevalence within the known cuisine obtains macro average precision 0.114 on the current test set. Examples include:

- `soy`: AP 0.608 versus test prevalence 0.122;
- `turmeric`: AP 0.504 versus prevalence 0.057;
- `yogurt`: AP 0.476 versus prevalence 0.072;
- `garam masala`: AP 0.388 versus prevalence 0.036;
- `fish sauce`: AP 0.353 versus prevalence 0.034.

Cuisine explains 58.98% of the binary entropy of `garam masala`, 55.13% for `turmeric`, and 50.07% for `fish sauce` under normalized mutual information. A visual model may infer cuisine and reconstruct these priors without identifying ingredients directly. Cuisine-stratified and observability-aware error analysis is therefore mandatory.

## How the active data loader changes the data

The main `ImagesRecipesBaseDataModule`:

1. loads `metadata.json` for train, validation, and test;
2. optionally filters records by the `cuisine` field;
3. fits the label encoder on train during the first processing pass;
4. transforms each `ingredients_ok` list into a multi-hot vector;
5. applies the fitted encoder to validation and test;
6. computes inverse-frequency positive weights from train;
7. loads images lazily and applies model-dependent transforms;
8. uses test as predict data when no dedicated predict split exists.

The default training configuration uses weighted `BCEWithLogitsLoss` and weighted multi-label accuracy, precision, recall, and Hamming distance. F1 is not enabled by default, no average-precision or calibration metric is configured, and no threshold-selection procedure is present; TorchMetrics therefore uses its default decision threshold. Weighted aggregates emphasize common labels and hide tail behavior.

The positive weights are standardized to `maximum_train_label_count / label_count`, not the conventional negative-to-positive ratio. This is a valid heuristic only if documented as such; it must not be described as a standard balanced BCE without comparison. The new protocol replaces the current headline metrics with label-macro mAP, micro F1, per-label results, and calibration diagnostics.

## Consequences for the project objective

The current data can support exploratory engineering, but it is not yet a defensible final benchmark. Four issues are decisive:

1. **target corruption:** some common labels are systematically wrong;
2. **semantic ambiguity:** recipe ingredients are not uniquely observable in the image;
3. **split leakage:** exact and semantic duplicates cross train/evaluation boundaries;
4. **metric inflation:** class sparsity makes label accuracy appear high without useful positive predictions.

Therefore, final comparative model experiments must follow rather than precede benchmark repair. Literature discovery and implementation prototyping may proceed under the documented constraints. Results obtained before repair must be labeled provisional and must not be compared directly with claims using the original 2,416-ingredient Yummly-66K representation.

## Required remediation order

The design choices are resolved in [`benchmark_decisions.md`](benchmark_decisions.md). Implementation must proceed in this order:

1. Freeze and checksum all 66,615 raw metadata/image pairs.
2. Regenerate targets with token-aware rules and a reviewed ontology, preserving a mapping trace.
3. Audit mapping precision and apply the semantic/support gates.
4. Produce the image exclusion and adjudication manifest.
5. Freeze recipe-family components using the accepted high-precision evidence.
6. Create a deterministic grouped 80/10/10 split and verify zero accepted cross-split components.
7. Remove the artificial `<UNK>` output.
8. Standardize aspect-ratio-preserving transforms for controlled model comparison.
9. Freeze metrics, calibration, thresholds, seeds, bootstrap procedure, and prior baselines in a benchmark data card.

## Limitations of this audit

- No full human relabeling was performed.
- Collision counts cover audited lexical signatures and are not a complete precision/recall estimate of the normalization pipeline.
- Perceptual-hash matches are candidates, not confirmed near-duplicates.
- The visual contact sheet is stratified and deterministic, not a random annotation study.
- The exact historical environment and saved normalization mapping are unavailable.
- No external dataset was used to measure domain shift or generalization.

## Sources and evidence

### Primary external source

- Weiqing Min et al., [*You Are What You Eat: Exploring Rich Recipe Information for Cross-Region Food Analysis*](https://openreview.net/pdf?id=F9oSOeGGkwP), IEEE Transactions on Multimedia 20(4), 2018, pp. 950–964.
- The authors' [official Yummly-66K repository](https://github.com/minweiqing/You-Are-What-You-Eat-Exploring-Rich-Recipe-Information-for-Cross-Region-Food-Analysis) and [VIPL resource page](https://vipl.ict.ac.cn/homepage/sqjiang/Resource/).

### Local evidence

- [`../../src_scratches/data_anlysis/yummly_audit.py`](../../src_scratches/data_anlysis/yummly_audit.py)
- [`../../src_scratches/data_anlysis/yummly_deep_audit.py`](../../src_scratches/data_anlysis/yummly_deep_audit.py)
- [`../../src_scratches/data_anlysis/outputs/yummly_deep_audit.json`](../../src_scratches/data_anlysis/outputs/yummly_deep_audit.json)
- [`../../src/raw2input/yummly/creation.py`](../../src/raw2input/yummly/creation.py)
- [`../../src/raw2input/yummly/recipes_merge.py`](../../src/raw2input/yummly/recipes_merge.py)
- [`../../src/raw2input/yummly/sort_recipes_as_img.py`](../../src/raw2input/yummly/sort_recipes_as_img.py)
- [`../../prev_attempts/attempt1/preprocessing_v2.py`](../../prev_attempts/attempt1/preprocessing_v2.py)
- [`../../src/data_processing/images_recipes.py`](../../src/data_processing/images_recipes.py)
- [`../../src/data_processing/labels_encoders.py`](../../src/data_processing/labels_encoders.py)
- [`../../src/data_processing/transformations.py`](../../src/data_processing/transformations.py)
