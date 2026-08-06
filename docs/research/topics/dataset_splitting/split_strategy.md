# Multi-label dataset splitting

**Created:** 2026-08-06  
**Last updated:** 2026-08-06

## Purpose

This note reviews general methods for splitting multi-label datasets. It is
independent of a particular dataset, model implementation, or benchmark
artifact. Its purpose is to distinguish the methodological roles of random
sampling, multi-label stratification, grouping constraints, and frozen
evaluation partitions.

## The splitting problem

In single-label classification, ordinary stratification can preserve the
frequency of each mutually exclusive class. In multi-label classification, one
example can carry several labels and label combinations may be rare. Preserving
one label's frequency can conflict with preserving another's or with preserving
co-occurrence patterns. No finite split can generally reproduce every aspect of
the original joint distribution.

## Random and stratified splitting

Pure random splitting is a valid sampling baseline, especially for large and
well-balanced datasets. It can nevertheless yield unstable label prevalences or
leave very rare labels poorly represented in one partition.

Multi-label stratification allocates examples while attempting to preserve the
prevalence of multiple labels. The original iterative-stratification study
compared multi-label stratification methods with random sampling and showed why
the multi-label structure must be considered rather than treated as an
ordinary single-label problem. [Sechidis, Tsoumakas, and Vlahavas (2011)](https://doi.org/10.1007/978-3-642-23808-6_10)

First-order stratification targets individual label marginals. Second-order
methods additionally target label-pair distributions, which can reduce the
variation of evaluation quality and improve label-pair representation, at the
cost of a more constrained allocation problem. [Szymański and Kajdanowicz
(2017)](https://proceedings.mlr.press/v74/szyma%C5%84ski17a.html)

## Group-aware splitting

Examples that share a source, subject, duplicate content, or another dependency
may need to remain in one partition. Group-aware splitting prevents direct
leakage through that dependency, but makes exact stratification harder or
impossible when group sizes and label composition are uneven. This trade-off is
also documented for group-stratified splitting in the official
[scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

The appropriate group definition depends on the research question. It should
be justified by an actual leakage mechanism; broad similarity heuristics can
change the target population and introduce subjective assumptions.

## Reproducibility and evaluation use

Determinism does not itself create sampling bias. A random procedure with a
recorded seed is deterministic on rerun, while a fully specified stratifier can
be deterministic without randomization. The relevant methodological questions
are what distributions and dependencies the procedure preserves, and whether
the procedure is fixed before model results are examined.

For a benchmark intended to compare several models, one frozen split makes the
comparison paired and reproducible: every model is evaluated on the same
examples. Training and validation support model development; the test
partition must remain outside model and hyperparameter selection. A separate
robustness study can use additional frozen splits or an external dataset, but
it answers a different question from the primary benchmark comparison.

## Limits

Stratification improves representation of the chosen variables; it does not
prove generalization to a different population, time period, source, or visual
domain. Nor does it remove all dependencies not represented by the chosen
groups. Reports should state the split method, grouping assumptions, retained
labels, split ratios, and whether results are in-distribution or external.

## Related project note

The repository-specific Yummly split contract, its SHA-256 grouping, validation
checks, and use rules are documented separately in
[`../../../technical_details/data/yummly_benchmark_split/explaination.md`](../../../technical_details/data/yummly_benchmark_split/explaination.md).

