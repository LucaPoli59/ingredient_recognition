# Yummly data analysis workspace

**Created:** 2026-08-02  
**Last updated:** 2026-08-02

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

Generated outputs are stored in `outputs/`:

- `yummly_audit.json` contains the complete structured report;
- `ingredient_frequency.csv` contains per-label statistics;
- `yummly_sample_contact_sheet.jpg` supports visual inspection.
- `yummly_deep_audit.json` contains target reproduction, collision, taxonomy, shortcut, and grouping evidence;
- `target_review.csv` supports per-label review;
- `duplicate_group_review.csv` and `duplicate_group_review.jpg` support duplicate and image-quality adjudication.

The temporary findings and open-question notes used during investigation were removed after their evidence and decisions were consolidated under `docs/project_objective/`.

## Legacy explorations

- `category_exploration.py` inspects split-level cuisine counts using an older directory layout.
- `images_shapes.py` samples image dimensions using an older directory layout.
- `image_exploration.ipynb` contains early manual image exploration.
- `ingredients_exploration.ipynb` contains early ingredient-frequency exploration.

The legacy artifacts are retained for history but are not the source of the current documented statistics.
