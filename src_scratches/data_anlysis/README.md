# Yummly data analysis workspace

**Created:** 2026-08-02  
**Last updated:** 2026-08-06

This scratch directory contains reproducible and legacy data-analysis artifacts for the Yummly dataset.

## Reproducible audit

Run the full metadata and image audit from the repository root:

```powershell
python src_scratches/data_anlysis/yummly_audit.py
```

Use `--skip-images` to refresh metadata statistics while retaining the image results from an existing report.

Run the decision-oriented second pass with:

```powershell
python src_scratches/data_anlysis/yummly_deep_audit.py
```

Its `--skip-images` mode refreshes target, taxonomy, support, and shortcut findings while preserving the prior image/grouping section.

Run the candidate `ingredients_target` vocabulary audit with:

```powershell
python src_scratches/data_anlysis/ingredient_vocabulary_audit.py
```

This audit reads the selected metadata generation without changing it and writes only aggregate target-support, co-occurrence, lexical-relationship, and source-example evidence. It deliberately does not persist a raw-line-to-target mapping.

Run the lightweight audit for a selected metadata file, split, and field with:

```powershell
python src_scratches/data_anlysis/metadata_field_audit.py `
    --metadata ingredients_target_v5_metadata.json `
    --split train `
    --field ingredients_target
```

The same command can inspect raw ingredient lines from any generation:

```powershell
python src_scratches/data_anlysis/metadata_field_audit.py `
    --metadata metadata.json `
    --split train `
    --field ingredients `
    --normalize-ingredients
```

The script is read-only and is intended for quick data questions.  Its
`audit.json` reports field validity, value and recipe-support counts,
cardinality distributions, support distributions, cuisine summaries, and
optional co-occurrence pairs.  `value_counts.csv` contains the complete
value list; `record_cardinality.csv` preserves the per-recipe counts.  Raw
`ingredients` remain visible as stored, while `--normalize-ingredients` adds
an explicitly separate summary using the current deterministic normalizer.

Run the train-only support-threshold comparison with:

```powershell
python src_scratches/data_anlysis/ingredient_threshold_sweep.py
```

It reads the original train `ingredients` field, applies the current deterministic normalizer, and writes aggregate vocabulary size, target-assignment retention, zero/one/two/three-or-more target recipe buckets, and named target transitions for each candidate support threshold. It does not generate metadata or choose a threshold. Re-run it after the controlled-vocabulary association rules are finalized before freezing a replacement generation.

Run the bounded fuzzy FoodOn research with:

```powershell
python src_scratches/data_anlysis/fuzzy_foodon_research.py --foodon-tsv <pinned-root-foodon-synonyms.tsv> --output-dir src_scratches/data_anlysis/outputs/fuzzy_foodon_research
```

It tests typo-only matching after exact FoodOn association and local fallback, reports only aggregate benchmark and corpus outcomes, and does not generate metadata or a raw-line mapping. The resulting decision is documented in `docs/plans/data_ingredient_refactor/controlled_vocabulary_evaluation.md`: fuzzy association is rejected from the standard pipeline.

Build the compact pinned FoodOn index from the root `foodon-synonyms.tsv`
export with:

```powershell
python scripts/build_foodon_index.py <pinned-root-foodon-synonyms.tsv>
```

Build the FoodOn-first Yummly generation with a dry run first:

```powershell
python scripts/build_yummly_foodon_metadata.py --foodon-index src/data_processing/resources/foodon_food_product_v2025_07_31.json
```

Add `--apply` only after reviewing the dry-run counts. The command writes the
new `ingredients_target_v5_metadata.json` generation and an aggregate report;
it does not modify `v1`–`v4` or legacy metadata.

Re-run the train-only threshold comparison after controlled association with:

```powershell
python src_scratches/data_anlysis/ingredient_threshold_sweep.py --foodon-index src/data_processing/resources/foodon_food_product_v2025_07_31.json --output src_scratches/data_anlysis/outputs/controlled_target_generation/threshold_sweep.json
```

Generated outputs are stored in `outputs/`:

- `yummly_audit.json` contains the complete structured report;
- `ingredient_frequency.csv` contains per-label statistics;
- `yummly_sample_contact_sheet.jpg` supports visual inspection.
- `yummly_deep_audit.json` contains target reproduction, collision, taxonomy, shortcut, and grouping evidence;
- `target_review.csv` supports per-label review;
- `duplicate_group_review.csv` and `duplicate_group_review.jpg` support duplicate and image-quality adjudication.
- `ingredient_vocabulary_audit/` contains the structured Work package 2.2a evidence and aggregate CSV views for the candidate target generation.
- `ingredient_threshold_sweep.json` records the train-only support-threshold comparison for the provisional current normalizer.
- `fuzzy_foodon_research/aggregate_report.json` records the bounded fuzzy FoodOn evaluation.
- `controlled_target_generation/report.json` records the v5 association, support, retention, and split summary.
- `controlled_target_generation/threshold_sweep.json` records the post-association train-only threshold comparison.
- `metadata_field_audit/<metadata>/<split>/<field>/` contains the lightweight selected-field audit outputs.

The temporary findings and open-question notes used during investigation were removed after their evidence and decisions were consolidated under `docs/project_objective/`.

## Legacy explorations

- `category_exploration.py` inspects split-level cuisine counts using an older directory layout.
- `images_shapes.py` samples image dimensions using an older directory layout.
- `image_exploration.ipynb` contains early manual image exploration.
- `ingredients_exploration.ipynb` contains early ingredient-frequency exploration.

The legacy artifacts are retained for history but are not the source of the current documented statistics.
