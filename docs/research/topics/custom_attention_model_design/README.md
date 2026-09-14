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

Research rationale and retained alternatives live here. The
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
3. [Architecture compatibility and scaling](architecture_compatibility_synthesis.md):
   complete residual/query routes, feature taps, padding/position and weight
   reuse contracts, scalar S/M/L estimates, exclusions, and integration risks.
4. [Three topology proposals](topology_proposals.md): comparable P1 residual,
   P2 query/context and P3 spatial-mixer/query designs; S/M/L scales, qualitative
   selection rationale, minimal controls and implementation handoff.

The earlier brief/component/compatibility documents retain their dated handoff
scope. The final comparison supports the portfolio's adoption of P2-S under
[4A-D2](../../../project_objective/experimental_model_portfolio.md#4a-d2--custom-attention-topology).
P1/P3 remain research alternatives, not extra selected models or required runs.
The four research artifacts do not establish implementation or measured gains.

## Method and evidence flow

The source cutoff and access date are **2026-09-08**. The review combines the
existing [candidate dossiers](../experimental_model_candidates/README.md),
current project inputs, original papers, and inspected official/library source.
It is a bounded component investigation, not a systematic or exhaustive survey.
The component record describes search families, reading depth, exclusions and
retrieval limitations.

Use the chain `requirement R# -> design question Q# -> component C# -> source
S#/I# -> proposal -> portfolio decision`. Keep original source results in their task, metric,
resolution and training context. A food-paper precedent can be negative
evidence: the Nutrition5K comparison does not show a consistent ML-Decoder
advantage over global pooling.

When extending this collection, distinguish source facts, source-reported
empirical findings, mathematical deductions, project interpretations,
provisional recommendations, and unresolved questions. Cite the component
entry rather than copying its evidence into each later proposal. The component
reuse table now links the proposals that retain or reject each mechanism.
Update this index whenever files or their roles change.

## Related authorities

- [Topic-research conventions](../README.md)
- [Documentation governance](../../../README_DOCS_ORGN.md)
- [Parent 4A plan](../../../plans/experimental_model_research.md)
- [Problem definition](../../../project_objective/problem_definition.md)
- [Benchmark decisions](../../../project_objective/benchmark_decisions.md)
- [Comparison methodology](../../../project_objective/model_comparison_methodology.md)
- [Requirements R1--R11](../../discovery/2026-08-28/problem_model_requirements.md)
