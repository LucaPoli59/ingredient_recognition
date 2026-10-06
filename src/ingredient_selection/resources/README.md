# Frozen ingredient projections

**Created:** 2026-10-05
**Last updated:** 2026-10-06

This directory versions small, explicit vocabulary definitions, not split
metadata, model predictions or a replacement runtime default.

- [`ingredients_selected_v5_d6_v1.json`](ingredients_selected_v5_d6_v1.json)
  is the P6 shared 59-label projection of the 165-label FoodOn-first v5
  vocabulary. It retains base indices, ordered names, the 60 uncertain and
  46 below-floor exclusions, independent reasons, and approved evidence hashes.

Regenerate with `python scripts/ingredient_selection/export_projection.py
analysis_outputs/ingredient_selection/phase3-d1-v3` from the repository root.
The exporter requires the retained approved D6 artifacts and accepts only
identical existing bytes. Do not hand-edit membership or overwrite this
version for a later policy. No split metadata is read or generated.

The [implementation contract](../../../docs/implementation_details/ingredient_selection.md#p6-frozen-projection)
owns the format and usage; [D6 and its P6 handoff](../../../docs/project_objective/model_comparison_methodology.md#p6-shared-projection-freeze--2026-10-05)
own the policy. The [P7 runtime contract](../../../docs/implementation_details/ingredient_selection.md#p7-runtime-projection)
consumes this frozen definition explicitly through
`ExpConfig(dm_ingredient_projection="ingredients_selected_v5_d6_v1")`, preserving
the full default, original records and saved class-column identity. Do not
edit this JSON to configure an experiment or to bypass runtime checks.
