# Experimental-model broad discovery — 2026-08-28

**Created:** 2026-08-28
**Last updated:** 2026-08-28
**Subphase:** 4A.1 broad model discovery
**Status:** Complete; handoff to 4A.2 is open

## Context and purpose

This discovery is the first execution artifact of the primary Subphase 4A
model-research stream. It converts the repaired Yummly problem into model
requirements, reviews the two preceding discovery records, and retains a
bounded set of family-level protocols for normalized deep research. It does not
select the final two experiment families, choose the Phase 3 selector, or run
any model.

The target remains **closed-vocabulary, recipe-level multi-label ingredient
inference from one RGB image of a finished dish**. Labels may describe hidden or
transformed ingredients, so a source about visible object detection,
segmentation, retrieval, or recipe generation is an adjacent/mechanistic source
unless its transfer boundary is stated.

## Relationship to earlier discoveries

The [2026-08-02 broad discovery](../2026-08-02/README.md) supplied the primary
state-of-the-art landscape: food ingredient inference, multi-label heads,
pretraining, long-tail and shortcut risks, calibration, and the 8 GB constraint.
The [2026-08-22 selector discovery](../2026-08-22/README.md) supplied reusable
family, source, checkpoint, licence, and instrumentation evidence. Its `M_ref`
intake tiers and selector dispositions remain owned by Subphase 4B and are not
copied as 4A decisions.

This record adds the 4A comparison boundary: candidates are family/protocol
tuples, close depth/width/checkpoint variants do not fill additional slots, and
head/label-side mechanisms remain explicit rather than being hidden inside a
backbone ranking. It also records that ResNet and DINOv2 are already-used local
baselines: their papers remain evidence, but they do not count as new candidates
in the selection.

## Method and cutoff

The search used primary papers, publisher records, official research pages,
maintained repositories, official model cards, and checkpoint documentation
available through **2026-08-28**. It covered:

1. supervised convolutional baselines and efficient CNN leads;
2. hierarchical/windowed transformers;
3. visual self-supervised representations;
4. vision-language representations;
5. class-query, set-decoding, and label-dependency heads; and
6. food-domain models as direct, adjacent, or provenance-sensitive leads.

No headline score was used to rank candidates across incompatible datasets or
metrics. No candidate was trained, memory-profiled, or evaluated locally.

## Files

- [`problem_model_requirements.md`](problem_model_requirements.md) records the
  mandatory input revisions and the requirement matrix used for intake.
- [`candidate_landscape.md`](candidate_landscape.md) records the five retained
  protocols, grouped leads/exclusions, hypotheses, and the formal 4A.1 → 4A.2
  handoff.
- [`source_catalog.md`](source_catalog.md) records claim-to-source links,
  implementation paths, evidence type, and transfer boundaries.

## Executive result

Five scientifically distinct **new** protocols are retained for 4A.2 dossier
review:

1. efficient convolutional network (EfficientNetV2);
2. hierarchical/windowed transformer (Swin V2);
3. vision-language visual encoder (SigLIP2, with CLIP/SigLIP fallbacks);
4. structured multi-label query/set head (ML-Decoder or Query2Label); and
5. hybrid multi-axis attention network (MaxViT).

ResNet and DINOv2 remain baseline anchors outside this count. C4 varies the
multi-label readout and must be paired with a declared backbone; it is retained
because local ingredient evidence and label-set structure are independent
hypotheses. This is a candidate set, not a ranking or an implementation
commitment.

### Shared conclusions

- EfficientNetV2 is the new efficient-convolution candidate; its food-task
  precedent and TorchVision path make it a practical comparison against the
  existing ResNet baseline.
- Swin V2 is a plausible local-plus-global representation, but its food transfer
  and 8 GB behavior are indirect/unverified.
- SigLIP2 is a useful contemporary semantic-prior candidate, but downstream text
  scoring is a separate experiment and provenance/overlap must be audited.
- MaxViT tests a hybrid local/global attention mechanism with a compact
  TorchVision path, but its square-input contract and archived research
  repository need verification.
- Query/set heads may expose label-specific spatial evidence, while dependency
  heads can exploit co-occurrence. Non-visual controls are therefore part of the
  later falsification protocol.
- DINOv2 and ResNet remain the already-used local anchors against which the new
  protocols are compared; neither is reselected as a new family.
- Food2K, Recipe1M+, segmentation, retrieval, and generative systems remain
  contextual leads until target, provenance, access, and resource conditions are
  re-established.

## Handoff to 4A.2

The next stage must create one normalized dossier for each of the five retained
new candidates. It should first resolve the open questions in the handoff table,
then freeze a concrete candidate tuple, implementation/checkpoint path, transfer
boundary, falsifiable local hypothesis, and go/no-go conditions. The dossiers
must include the B1 ResNet and B2 DINOv2 anchors as comparison references, but
must not reclassify them as new candidates. The stage must not train models,
select hyperparameters, compare test results, or silently promote a grouped
lead.

## Limitations

- This is a bounded discovery rather than a systematic review or meta-analysis.
- Direct Yummly-like evidence remains sparse; most sources are adjacent or
  mechanistic.
- Access, licence, APIs, and checkpoint availability can change and require a
  Phase 5 recheck.
- The one-declared-seed project budget limits later stability claims.

## Related documentation

- [Experimental-model research plan](../../../plans/experimental_model_research.md)
- [Problem definition](../../../project_objective/problem_definition.md)
- [Benchmark decisions](../../../project_objective/benchmark_decisions.md)
- [Comparative model methodology](../../../project_objective/model_comparison_methodology.md)
- [Current model implementation contract](../../../implementation_details/models.md)
- [Research directory governance](../../README.md)
