# Yummly data analysis workspace

**Created:** 2026-08-02  
**Last updated:** 2026-08-05

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

Run the train-only support-threshold comparison with:

```powershell
python src_scratches/data_anlysis/ingredient_threshold_sweep.py
```

It reads the original train `ingredients` field, applies the current deterministic normalizer, and writes aggregate vocabulary size, target-assignment retention, zero/one/two/three-or-more target recipe buckets, and named target transitions for each candidate support threshold. It does not generate metadata or choose a threshold. Re-run it after the controlled-vocabulary association rules are finalized before freezing a replacement generation.

Generated outputs are stored in `outputs/`:

- `yummly_audit.json` contains the complete structured report;
- `ingredient_frequency.csv` contains per-label statistics;
- `yummly_sample_contact_sheet.jpg` supports visual inspection.
- `yummly_deep_audit.json` contains target reproduction, collision, taxonomy, shortcut, and grouping evidence;
- `target_review.csv` supports per-label review;
- `duplicate_group_review.csv` and `duplicate_group_review.jpg` support duplicate and image-quality adjudication.
- `ingredient_vocabulary_audit/` contains the structured Work package 2.2a evidence and aggregate CSV views for the candidate target generation.
- `ingredient_threshold_sweep.json` records the train-only support-threshold comparison for the provisional current normalizer.

The temporary findings and open-question notes used during investigation were removed after their evidence and decisions were consolidated under `docs/project_objective/`.

## Legacy explorations

- `category_exploration.py` inspects split-level cuisine counts using an older directory layout.
- `images_shapes.py` samples image dimensions using an older directory layout.
- `image_exploration.ipynb` contains early manual image exploration.
- `ingredients_exploration.ipynb` contains early ingredient-frequency exploration.

The legacy artifacts are retained for history but are not the source of the current documented statistics.
