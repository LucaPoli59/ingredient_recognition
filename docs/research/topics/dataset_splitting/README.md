# Dataset splitting research

**Created:** 2026-08-06  
**Last updated:** 2026-08-06

## Context

This topic records general research that informs how multi-label datasets can
be partitioned. It is deliberately independent from the Yummly implementation
and remains applicable to later datasets, model families, and evaluation work.

## Scope

The research compares pure random sampling, multi-label stratification,
higher-order label preservation, group-aware splitting, and frozen benchmark
evaluation. Repository-specific implementation choices are documented under
`docs/technical_details/`.

## Contents

- [`split_strategy.md`](split_strategy.md) explains the research evidence and
  general methodological trade-offs.
