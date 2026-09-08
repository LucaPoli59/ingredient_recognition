# Custom attention-model design research

**Created:** 2026-09-08
**Last updated:** 2026-09-08

## Purpose and scope

This collection investigates how established visual-attention components could
support ingredient-specific use of spatial evidence and dish context under weak
recipe-level supervision. It preserves the evidence for the four research
subphases in the [custom-model plan](../../../plans/custom_attention_model.md).
The component analysis is reusable for other closed-vocabulary multi-label
image tasks; the problem brief explicitly identifies the Yummly-specific limits.

Research recommendations here are provisional. The
[experimental portfolio](../../../project_objective/experimental_model_portfolio.md)
owns binding model decisions, and the feature plan owns execution status.
The collection does not establish local model accuracy or resource feasibility.

## Files and reading order

1. [Problem and evidence synthesis](problem_evidence_synthesis.md): reviewed
   input revisions, inherited constraints, portfolio gaps, bounded design
   objective, and questions passed to component research.
2. [Attention-component evidence](attention_component_evidence.md): primary
   evidence, counterevidence, implementation observations, provisional component
   dispositions, and the handoff to tensor-level synthesis. Its source register
   provides stable identifiers for later reuse.

The compatibility synthesis and three topology proposals will be added by
4A.4.3 and 4A.4.4, respectively. They do not exist yet.

## Method and evidence flow

The source cutoff and access date are **2026-09-08**. The review combines the
existing [candidate dossiers](../experimental_model_candidates/README.md),
current project inputs, original papers, and inspected official/library source.
It is a bounded component investigation, not a systematic or exhaustive survey.
The component record describes search families, reading depth, exclusions and
retrieval limitations.

Use the chain `requirement R# -> design question Q# -> component C# -> source
S#/I# -> later proposal`. Keep original source results in their task, metric,
resolution and training context. A food-paper precedent can be negative
evidence: the Nutrition5K comparison does not show a consistent ML-Decoder
advantage over global pooling.

When extending this collection, distinguish source facts, source-reported
empirical findings, mathematical deductions, project interpretations,
provisional recommendations, and unresolved questions. Cite the component
entry rather than copying its evidence into each later proposal. Once proposals
exist, add their links to the component reuse table. Update this index whenever
files or their roles change.

## Related authorities

- [Topic-research conventions](../README.md)
- [Documentation governance](../../../README_DOCS_ORGN.md)
- [Parent 4A plan](../../../plans/experimental_model_research.md)
- [Problem definition](../../../project_objective/problem_definition.md)
- [Benchmark decisions](../../../project_objective/benchmark_decisions.md)
- [Comparison methodology](../../../project_objective/model_comparison_methodology.md)
- [Requirements R1--R11](../../discovery/2026-08-28/problem_model_requirements.md)
