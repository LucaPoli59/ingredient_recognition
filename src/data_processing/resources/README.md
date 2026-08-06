# Data-processing resources

**Created:** 2026-08-05  
**Last updated:** 2026-08-05

This directory contains compact, version-pinned resources required by
deterministic data-processing code. The FoodOn JSON is generated from the
root `foodon-synonyms.tsv` export of FoodOn v2025-07-31 at commit `7ede44c`.
The pinned source URL is
`https://raw.githubusercontent.com/FoodOntology/foodon/7ede44c/foodon-synonyms.tsv`.
It contains only the `food product` descendant branch and the exact preferred
or exact-synonym lexical surfaces used by the controlled Yummly target
generator. It is an offline lookup index, not a second project vocabulary:
metadata still stores only canonical strings in `ingredients_target`.

To rebuild it from the pinned upstream export:

```powershell
python scripts/build_foodon_index.py <path-to-foodon-synonyms.tsv>
```

The generated JSON records the source SHA-256 and branch root. A source
release or parsing-rule change requires a new resource filename and a new
metadata generation; existing generations remain immutable.
