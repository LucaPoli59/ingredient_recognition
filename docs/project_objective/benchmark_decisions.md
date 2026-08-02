# Yummly benchmark decisions

**Created:** 2026-08-02  
**Last updated:** 2026-08-02

## Status and purpose

This document resolves the benchmark-design questions raised by the Yummly data audit. It is the decision record for data preparation and evaluation that must precede model discovery.

The current 65,146-record, 182-label dataset remains useful for diagnostics and historical comparisons, but it is not the final benchmark. Final model claims require a regenerated target, an exclusion manifest, a group-aware split, and a frozen evaluation protocol.

## Decision summary

| ID | Question | Decision |
|---|---|---|
| D1 | Repair or regenerate `ingredients_ok`? | Regenerate targets from the original ingredient lines and a versioned ontology. Do not patch the 182-label lists in place. |
| D2 | Predict all ingredients or only visible ones? | Predict the reviewed recipe-ingredient set as the primary task; evaluate visibility as separate instance-level slices. |
| D3 | Which aliases should merge? | Merge grammatical and measurement variants; keep distinct ingredients, plant parts, and named prepared products unless the ontology explicitly relates them. |
| D4 | How should recipes be grouped and split? | Exclude invalid images, build high-precision recipe families, then create a deterministic grouped 80/10/10 split with multi-label and cuisine balance. |
| D5 | Which images should be removed? | Exclude confirmed logos, advertisements, silhouettes, non-food placeholders, and mismatches through a machine-readable review manifest. Keep legitimate duplicate recipes in one split. |
| D6 | Keep 182 labels, use top-K, or rebuild? | Rebuild semantically; use training support and evaluation reliability gates, not a fixed legacy size or top-K cutoff. |
| D7 | How are thresholds and calibration selected? | Select them on validation only after model selection; require a global threshold and allow regularized per-label thresholds only with sufficient support. |
| D8 | Which metrics are primary? | Label-macro mAP is the primary ranking metric and micro F1 is the primary discrete-set metric. Both are required. |
| D9 | Should invisible ingredients be separate? | Yes in reporting, through per-recipe/per-label observability annotations on an audited test subset; not as silently removed training labels. |
| D10 | Can the current recipes be exactly regenerated? | No. The checked-in historical script is close but demonstrably not the exact generator of the current artifacts. |
| D11 | Should `<UNK>` remain an output? | No. Remove it from the closed-vocabulary task before any new experiment. |

## D1: regenerate the target from source ingredient lines

### Decision

Build a new target from `data/raw_input/yummly/metadata/*.json`. The new process must start from the original 66,615 metadata records and must not inherit the current `ingredients_ok` field as ground truth.

The existing 182 labels may be retained only as a named legacy target for regression tests and historical result interpretation.

### Why repair-in-place is rejected

The second-pass audit executed the normalization function from `prev_attempts/attempt1/preprocessing_v2.py` on all 144,010 unique original ingredient strings. Before the historical similarity merge:

- 174 labels survive the script's threshold, not 182;
- only 165 labels overlap the current vocabulary;
- exact per-recipe label sets match for 52,388 of 65,146 processed recipes, or 80.42%;
- micro F1 between regenerated and stored sets is 0.978, which shows close lineage but not exact identity;
- four checked-in recipe IDs are not eligible under the historical script even before its merge step. Because the merge can only collapse labels, it cannot make these records eligible later;
- the script would consider 65,170 source records eligible before merging, compared with 65,159 checked-in derived recipes.

The historical threshold called “minimum 50 occurrences” actually counts distinct raw ingredient strings that normalize to a value. It does not count recipes or ingredient occurrences. Approximate merge order is also driven by unordered multiprocessing completion. Neither behavior is a defensible or deterministic vocabulary rule.

### Required regeneration contract

The replacement normalizer must:

1. normalize Unicode, case, punctuation, quantities, units, and preparation text without deleting substrings inside unrelated words;
2. match aliases as token- or phrase-bounded rules;
3. map each retained raw line to a canonical ingredient ID, not only a display string;
4. store the raw line, matched rule, canonical target, rule version, and rejection reason;
5. reject or queue ambiguous mappings rather than guessing;
6. compute support by distinct recipes in the training split;
7. keep the source metadata immutable and write a new versioned benchmark tree;
8. include unit tests for every known collision and every accepted alias;
9. publish a vocabulary manifest and per-record mapping trace.

The original paper's 2,416-ingredient representation is not stored locally as a separate authoritative mapping. It can inform provenance, but it must not be described as recovered unless its exact source artifact is obtained and verified.

## D2 and D9: one recipe target, explicit observability slices

### Primary task

The primary task remains prediction of the canonical ingredients declared by the associated recipe from one image. It is deliberately a weakly supervised recipe-inference task, not visible-object detection.

Training two undocumented targets—“all ingredients” and “visible ingredients”—would make model comparisons difficult and would treat visibility as if it were constant for each ingredient. It is not. Onion may be directly visible in a salad, contextually implied in a soup, or visually unrecoverable in a smooth sauce.

### Observability protocol

Create a reviewed subset of the frozen test set with one annotation for each positive recipe-label pair:

- `direct`: visually identifiable evidence is present;
- `contextual`: the dish provides plausible category-level evidence but not direct identification;
- `not_inferable`: the ingredient is transformed, occluded, dissolved, or otherwise unsupported by the image;
- `uncertain`: image quality or ambiguity prevents a stable judgment.

At least two reviewers should independently annotate the subset. Report agreement and adjudicate disagreements. The sampling must cover all cuisines, frequent and rare labels, image-quality strata, and recipe cardinalities.

Headline results use the full reviewed recipe target. Results for `direct`, `contextual`, and `not_inferable` positives are mandatory slices. A separate visually grounded benchmark may be introduced later, but only as an explicitly named secondary task.

## D3: ontology and alias policy

### Merge rule

Merge strings when their difference is grammatical, orthographic, a measurement phrase, or a non-semantic preparation modifier. Do not merge merely because strings are close under edit distance or share a substring.

The following legacy groups are clear first-pass merges:

| Canonical label | Legacy variants | Unique recipes after merge |
|---|---|---:|
| `arugula` | `arugula`, `baby arugula` | 489 |
| `bay leaf` | `bay leaf`, `bay leaves` | 2,456 |
| `celery` | `celery`, `celery ribs`, `celery stalks`, `stalks celery` | 2,166 |
| `cherry` | `cherry`, `cherries` | 246 |
| `cilantro` | `cilantro`, `cilantro leaves`, `bunch cilantro` | 9,718 |
| `garam masala` | `garam masala`, `garam masala powder` | 2,315 |
| `leek` | `leek`, `leeks` | 569 |
| `lime` | `lime`, `limes` | 3,084 |
| `raspberry` | `raspberry`, `raspberries` | 449 |
| `strawberry` | `strawberry`, `strawberries` | 749 |
| `tomato` | `tomato`, `tomate` | 15,935 |

Applying only these obvious merges would reduce the legacy vocabulary from 182 to 168 labels. That number is diagnostic, not the target size of the regenerated ontology.

### Required non-merges

- Keep cilantro leaf and coriander seed concepts distinct unless a raw line is unambiguously an alias.
- Keep fennel bulb and fennel seed distinct.
- Keep jalapeño, poblano, serrano, ancho, and generic green chile distinct; an optional parent `chile pepper` may connect them hierarchically.
- Keep named prepared sauces distinct from a generic `sauce` label.
- Keep individual nuts distinct; an optional `nut` parent may support hierarchical evaluation.
- Keep ingredient identity separate from preparation state unless the prepared product is itself the purchased ingredient, such as pesto or pizza dough.

Every merge and non-merge must be represented in the ontology and covered by a regression test.

## D4: recipe families and the final split

### Group construction

Construct connected components before splitting. Union records when any high-precision rule holds:

1. identical SHA-256 image hash;
2. reviewed perceptual match, initially requiring agreement of dHash and pHash and manual confirmation for ambiguous groups;
3. identical normalized raw ingredient-line multiset;
4. identical canonical recipe name plus raw-line Jaccard similarity of at least 0.8;
5. a shared source-recipe family identifier, if a reliable source field is recovered.

Do not group by target label set alone: unrelated recipes commonly share frequent ingredient sets. Do not group by canonical name alone: names such as “French Toast” and “Salsa” are generic.

The high-precision candidate graph has 61,851 components for the 65,146 current records. It places 6,088 records in multi-record components, with a largest component of 27. This is small enough for grouped allocation.

Under the current split, the same graph identifies:

- 842 components crossing split boundaries;
- 1,953 records inside those cross-split components;
- 438 validation records, 8.41% of validation, grouped with at least one training record;
- 413 test records, 7.92% of test, grouped with at least one training record.

These are candidate family overlaps, not all confirmed near-duplicate images. They justify rebuilding the split rather than deleting individual evaluation records.

### Split allocation

After exclusions and target regeneration, allocate whole components to deterministic 80/10/10 train, validation, and test splits. Use a group-aware multi-label balancing procedure over:

- record count;
- cuisine distribution;
- positive support for each target;
- label cardinality;
- source/quality flags where available.

Run the allocation with a versioned seed, record the objective and achieved drift, then freeze the test manifest. Model selection, thresholds, calibration, and ontology revisions must not inspect test outcomes.

Acceptance requires zero exact-image, confirmed perceptual-image, exact-raw-list, or accepted recipe-family component crossing a split boundary.

## D5: image exclusions and duplicate handling

### Exclusion policy

Exclude a record when its image is confirmed as:

- an advertisement or publisher logo;
- a blank or silhouette placeholder;
- a generic category image unrelated to the specific recipe;
- a non-food image;
- a clear recipe-image mismatch;
- corrupt or unreadable.

Each exclusion must include record ID, image hash, reason code, reviewer, date, and review status. Borderline cases remain flagged and are included only under a documented policy.

Manual review of the largest exact-image groups has already confirmed eight exclusion groups covering 84 records: a Cooking Light advertisement, publisher logos, a BBC logo, a silhouette, and generic place-setting images. This is a verified minimum, not a complete quality estimate. Singleton images and the remaining duplicate groups still require review.

Legitimate repetitions of the same recipe image are not automatically invalid. Keep a representative or keep the family, depending on the benchmark's unit of analysis, but ensure the whole family remains in one split. Identical images attached to unrelated recipes or conflicting targets must be excluded or adjudicated, not treated as independent training examples.

Do not exclude an image only because one side is below 224 pixels. Preserve a resolution flag, avoid unnecessary upscaling, and compare transforms under a common aspect-ratio-preserving policy.

## D6: vocabulary size and record retention

The final vocabulary size is an output of the ontology and support rules, not an input target. Do not select 182 because it is current, 2,416 because it appears in the source paper, or 100 because it preserves 95.51% of legacy occurrences.

Use these gates:

1. the canonical concept is within the documented ingredient ontology;
2. mapping precision passes manual review on a stratified raw-line sample;
3. the label has at least 200 positive training recipes in the frozen grouped split for inclusion in headline macro metrics;
4. validation and test should each contain at least 20 positives; if grouping makes that impossible, move the label to an explicitly exploratory tail rather than changing the test set after modeling;
5. rare but important ingredients may remain as auxiliary outputs and per-label results, but not silently enter headline macro averages.

In the legacy split, 150 of 182 labels have at least 200 training positives and 147 have at least 20 positives in both validation and test. Eleven labels have fewer than 100 training examples. These figures justify a reliability gate but do not determine the regenerated vocabulary.

Rebuild from all 66,615 source records with valid images. Do not inherit the historical minimum-three-label filter. Keep a record with one or more valid in-vocabulary targets; exclude zero-target records with an explicit reason. This avoids retaining only recipes that happen to fit the old normalizer.

## D7: thresholds and calibration

### Required threshold path

1. Select model architecture and training configuration using threshold-free validation metrics.
2. Freeze the selected model.
3. Fit calibration parameters on validation logits only.
4. Select a single global decision threshold on validation to maximize micro F1. This is the mandatory, comparable thresholded result.
5. As a secondary analysis, fit per-label thresholds only for labels with at least 50 validation positives. Regularize or shrink them toward the global threshold; labels below that support use the global threshold.
6. Apply the frozen calibration and thresholds to test exactly once.

Report the threshold vector and calibration parameters with the checkpoint. A fixed 0.5 threshold may be included as a baseline but is not assumed optimal.

Because empirical F1 optimization can produce undesirable all-positive decisions for uninformative rare classifiers, per-label macro-F1 threshold maximization must not be used without support constraints and inspection. Threshold-free ranking metrics remain primary for model selection.

Use a simple global temperature or a regularized vector/Platt calibration baseline. Evaluate Brier score, reliability plots, and expected calibration error overall and by support tier. If probabilities are not calibrated, call them scores rather than probabilities.

## D8: evaluation protocol

### Primary metrics

- **Macro mean average precision:** primary threshold-free ranking metric, calculated over the fixed headline vocabulary. It prevents common labels from completely dominating model selection.
- **Micro F1:** primary discrete-set metric, calculated with the frozen validation-selected global threshold. It measures aggregate positive-label retrieval.

A model must report both. Neither may be replaced by raw label accuracy or a single weighted average.

### Required secondary metrics

- micro average precision;
- sample-wise F1;
- macro F1 with explicit zero-division policy;
- per-label AP, precision, recall, F1, prevalence, and confidence interval;
- precision@5, recall@5, precision@10, and recall@10;
- Hamming loss and exact match as secondary diagnostics;
- Brier score and expected calibration error when scores are interpreted probabilistically;
- results by label support, cuisine, course, image quality, recipe-family membership, and observability category.

Report mean and standard deviation over at least three training seeds. Use group-level bootstrap confidence intervals on the frozen test set so duplicate-family members are not resampled independently.

The all-negative, global-frequency, cuisine-prior, and simple visual baselines remain mandatory. The cuisine prior is diagnostic because it uses metadata unavailable to an image-only model.

## D10: reproducibility conclusion

The checked-in historical script establishes close lineage but cannot exactly reproduce the current raw recipe JSON files or the 182-label target. The mismatch is structural, not merely a missing random seed:

- the vocabulary sizes and members differ;
- exact recipe target sets differ for 19.58% of processed records before the script's similarity merge;
- four checked-in recipe IDs fail the script's pre-merge minimum-label rule and cannot be restored by a merge that only removes distinctions;
- similarity merging uses unordered multiprocessing output;
- saved mapping tables, checksums, dependency versions, and execution manifests are absent.

Therefore, the present `ingredients_ok` data must be versioned as a non-reproducible legacy artifact. The new benchmark must have a clean raw-to-output build with deterministic ordering and content hashes.

## D11: remove `<UNK>`

The task is closed-vocabulary. All 182 current labels occur in train, validation, and test, so the appended `<UNK>` column is always negative. It changes the output dimension from 182 to 183 and receives zero positive weight.

Replace `MultiLabelBinarizerRobust` with an encoder whose classes exactly match the frozen vocabulary, or change the robust encoder so `<UNK>` is used only for explicit ingestion diagnostics and never becomes a prediction target. Unknown raw ingredients are outside the current output space and must be recorded by preprocessing, not represented as a permanently negative model class.

## Benchmark readiness checklist

Model discovery may survey methods before these items are complete, but no result is a final benchmark claim until every item below passes:

- [ ] all raw metadata and images have a checksum manifest;
- [ ] the deterministic normalizer and ontology are versioned and unit-tested;
- [ ] mapping precision has been manually audited;
- [ ] image exclusions and adjudications are in a machine-readable manifest;
- [ ] recipe-family components are reviewed and frozen;
- [ ] the grouped 80/10/10 split has zero accepted cross-split components;
- [ ] the target vocabulary and support tiers are frozen;
- [ ] `<UNK>` is absent from model outputs;
- [ ] transforms are comparable and preserve aspect ratio;
- [ ] baselines, metrics, calibration, thresholds, seeds, and bootstrap procedure are frozen;
- [ ] the test set has not been used for model, ontology, threshold, or calibration selection.

## Evidence

### Local evidence

- [`yummly_data_audit.md`](yummly_data_audit.md)
- [`../../src_scratches/data_anlysis/yummly_audit.py`](../../src_scratches/data_anlysis/yummly_audit.py)
- [`../../src_scratches/data_anlysis/yummly_deep_audit.py`](../../src_scratches/data_anlysis/yummly_deep_audit.py)
- [`../../src_scratches/data_anlysis/outputs/yummly_deep_audit.json`](../../src_scratches/data_anlysis/outputs/yummly_deep_audit.json)
- [`../../src_scratches/data_anlysis/outputs/target_review.csv`](../../src_scratches/data_anlysis/outputs/target_review.csv)
- [`../../src_scratches/data_anlysis/outputs/duplicate_group_review.csv`](../../src_scratches/data_anlysis/outputs/duplicate_group_review.csv)
- [`../../src_scratches/data_anlysis/outputs/duplicate_group_review.jpg`](../../src_scratches/data_anlysis/outputs/duplicate_group_review.jpg)

### External evidence

- Min et al., [*You Are What You Eat: Exploring Rich Recipe Information for Cross-Region Food Analysis*](https://openreview.net/pdf?id=F9oSOeGGkwP), IEEE Transactions on Multimedia, 2018.
- Lipton, Elkan, and Narayanaswamy, [*Thresholding Classifiers to Maximize F1 Score*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/), 2014, explains threshold behavior for binary and multi-label F1, including failure modes for uninformative classifiers.
- Guo et al., [*On Calibration of Modern Neural Networks*](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017, provides the calibration baseline motivating temperature scaling.
