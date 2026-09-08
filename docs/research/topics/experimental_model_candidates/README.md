# Experimental model candidate dossiers

**Created:** 2026-08-28
**Last updated:** 2026-09-08
**Status:** Complete for Subphase 4A.2; subsequent 4A.3 decision linked

## Purpose and scope

This collection is the durable deep-research record for the five new
family/protocol candidates retained by Subphase 4A.1. It asks what the
literature, official implementations, and available checkpoints actually
establish, and what could be compared fairly on the frozen Yummly
`ingredients_target_v5` benchmark.

The collection does not select the two experiment families, choose
hyperparameters, execute training, or measure local accuracy. ResNet and
DINOv2 are kept as already-used comparison anchors; they are not new
candidates and therefore do not receive a new candidate dossier here.

## Method and cutoff

The review used the current problem-to-model matrix, the 4A.1 discovery, the
current repository model contract, original architecture/pretraining papers,
official repositories or maintained library implementations, official
checkpoint/model cards, and food-task evidence where available. The source
cutoff is **2026-08-28**. Claims are separated into verified facts, external
empirical evidence, project interpretation, recommendations, and unresolved
questions. Scores reported by a source remain in that source's dataset,
vocabulary, split, and metric context; they are not cross-family rankings.

## Common dossier schema

Every candidate file follows the same order:

1. research question and boundary;
2. family/protocol identity and architecture mechanism;
3. feature flow and adaptation boundary;
4. pretraining and task-relevant evidence;
5. project-fit matrix and transfer limits;
6. canonical image-only multi-label protocol and alternatives;
7. implementation, checkpoint, licence, dependency, and provenance path;
8. variants and resource envelope, separating known facts from unmeasured
   8 GB behaviour;
9. repository integration path;
10. risks, go/no-go uncertainties, and open questions; and
11. one falsifiable local hypothesis with a minimal fair comparison.

## Files

- [`efficientnet_v2.md`](efficientnet_v2.md) — C1, efficient convolutional
  family.
- [`swin_v2.md`](swin_v2.md) — C2, hierarchical/windowed transformer.
- [`siglip2.md`](siglip2.md) — C3, vision-language visual encoder.
- [`structured_multilabel_head.md`](structured_multilabel_head.md) — C4,
  query/set readout protocol paired with a declared backbone; includes the
  2026-09-08 source correction linked to the custom component research.
- [`maxvit.md`](maxvit.md) — C5, hybrid multi-axis attention family.
- [`comparative_synthesis.md`](comparative_synthesis.md) — common qualitative
  comparison, the historical 4A.2 → 4A.3 handoff, and the bounded 2026-09-07
  source-review addendum.

## Working rules and limitations

- A model variant is nested inside its family unless it changes the tested
  scientific mechanism. Parameter size, checkpoint generation, patch/window
  size, and input resolution are therefore not additional candidate slots.
- The canonical downstream contract is one RGB image to 165 independent
  logits, with sigmoid/loss handling owned by the training module. Text,
  recipe metadata, ingredient prompts, graph statistics, and autoregressive
  generation are separate interventions, not silent parts of a candidate.
- Food-domain or web-language pretraining is evidence about a representation,
  not proof that a label is visually observable. Source overlap, target-name
  leakage, and cuisine shortcuts remain explicit risks.
- Reported parameter counts, FLOPs, transforms, and licences are source or
  library facts. Peak memory, throughput, exact aspect-preserving transforms,
  checkpoint hashes, and dependency compatibility require the Macro-section 5
  implementation gate.
- The collection is a research owner. The adopted shortlist, protocols, and
  subsequent handoff are owned by the
  [experimental portfolio](../../../project_objective/experimental_model_portfolio.md).
  Dossier recommendations retain their pre-decision meaning; they do not
  override that decision.

## Related documentation

- [Adopted experimental portfolio](../../../project_objective/experimental_model_portfolio.md)
- [4A experimental-model plan](../../../plans/experimental_model_research.md)
- [2026-08-28 broad discovery](../../discovery/2026-08-28/README.md)
- [Problem-to-model requirements matrix](../../discovery/2026-08-28/problem_model_requirements.md)
- [Current model implementation contract](../../../implementation_details/models.md)
- [Problem definition](../../../project_objective/problem_definition.md)
- [Benchmark decisions](../../../project_objective/benchmark_decisions.md)
- [Comparative model methodology](../../../project_objective/model_comparison_methodology.md)
