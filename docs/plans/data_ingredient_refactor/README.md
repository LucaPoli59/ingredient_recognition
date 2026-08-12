# Data ingredient refactor

**Created:** 2026-08-04  
**Last updated:** 2026-08-12

This directory contains the active implementation plan and its durable, project-specific research evidence for the Yummly ingredient-data refactor.

## Files

- [`yummly_data_phase.md`](yummly_data_phase.md) is the operational source of truth for Data Work packages 2.1b–2.4, their progress, dependencies, implementation tasks, and completion gates. Work package 2.1c defines the minimum retained November 2024 ingredient-selection evidence and executable compatibility anchors; its gate is closed by [`validate_legacy_experiments.py`](../../../scripts/validate_legacy_experiments.py) and the generated [`retention_manifest.json`](../../../src_scratches/ingredient_selection_reconstruction/retention_manifest.json).
- [`controlled_vocabulary_evaluation.md`](controlled_vocabulary_evaluation.md) records the Yummly-specific Work package 2.2c experiment, findings, and the contract implemented by Work package 2.2d.

## Working rule

Read both files before changing ingredient canonicalization, metadata generations, split construction, loaders, or runtime target defaults. Update the plan whenever a work-package status changes, and keep the evaluation as durable evidence rather than duplicating its measurements in other documents.

The reusable, dataset-independent vocabulary catalog remains in [`../../research/topics/ingredient_vocabularies/`](../../research/topics/ingredient_vocabularies/README.md). Project-wide state remains in [`../../general_plan.md`](../../general_plan.md).
