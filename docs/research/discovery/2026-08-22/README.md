# Reference-selector candidate discovery — 2026-08-22

**Created:** 2026-08-22
**Last updated:** 2026-08-22

## Context and purpose

This discovery supports Work package 4.6 in the
[general project plan](../../../general_plan.md). Its purpose is to map the
architecture, pretraining, adaptation, and multi-label-head families that could
act as the reference selector `M_ref` for the later ingredient-learnability
study.

The selector is a measurement instrument: it will help decide which labels in
the frozen `v5` vocabulary can be learned under one declared visual-learning
protocol. It is not automatically the final benchmark winner. This discovery
therefore asks what each candidate would measure, not which paper reports the
largest score on an unrelated dataset.

## Relationship to the preceding discovery

Only one earlier discovery directory existed at the start of this work, so the
project rule to review the two most recent records was satisfied to the extent
possible by reviewing the complete
[2026-08-02 discovery](../2026-08-02/README.md).

This record reuses its:

- closed-vocabulary, recipe-level multi-label task boundary;
- separation of image pipeline, representation, head, loss, and decision rule;
- warning that food, segmentation, retrieval, and general multi-label
  benchmarks are not directly comparable;
- 8 GB development-GPU constraint; and
- initial evidence for ResNet/ConvNeXt, DINO, vision-language encoders, and
  class-query heads.

The distinct contribution of this discovery is the reference-selector
perspective. It:

- broadens the catalog beyond models already implemented in the repository;
- distinguishes visual supervision, visual self-supervision, generic
  image-text pretraining, and food-specific pretraining;
- makes frozen probing, learned-head transfer, fine-tuning, and text-defined
  scoring separate protocols;
- identifies when label names or label dependencies add information that can
  be mistaken for image learnability; and
- applies the current one-declared-seed constraint rather than the older
  multi-seed recommendation.

## Scope and method

The search used original papers, publisher proceedings, official research
pages, and official repositories available through **2026-08-22**. It covered:

1. supervised convolutional and transformer backbones;
2. visual self-supervised and masked-image foundation models;
3. generic contrastive vision-language encoders;
4. food- and recipe-domain pretrained representations;
5. independent, spatial class-query, dependency-aware, and set-decoding heads;
6. recent food vision-language systems as emerging leads; and
7. checkpoint, licence, compute, and integration questions that must be
   resolved before a candidate becomes eligible.

The search is broad and decision-oriented, not a systematic review or
meta-analysis. Reported scores were not used to rank candidates because the
source tasks, label semantics, data, and compute budgets differ.

## Files

- [`candidate_landscape.md`](candidate_landscape.md) defines the measurement
  protocols, catalogs candidate families, records their interpretation risks,
  and hands a bounded but non-ranked set to the repository inventory.
- [`candidate_integration_inventory.md`](candidate_integration_inventory.md)
  completes R0.2 by mapping the catalog to the frozen `v5` contract, current
  instrumentation, dependency and checkpoint paths, and measured-versus-
  unverified 8 GB evidence.
- [`source_catalog.md`](source_catalog.md) records the primary sources and the
  transfer boundary attached to each source.

## Executive synthesis

### No model is a universal learnability oracle

An ingredient can be accessible to one pretrained representation and absent
from another. It can also become learnable after fine-tuning even when it is not
linearly available in frozen features. Every future statement must therefore
mean “learnable relative to the frozen `M_ref` protocol.”

### Pretraining is admissible, but changes the claim

A visually self-supervised encoder measures transfer from general image
structure. A vision-language encoder also transfers concepts associated with
web text. A food-domain encoder can transfer recipe, dish, or ingredient
knowledge even more directly. These priors may make the selector more sensitive
to weak visual evidence, but they do not establish learnability from the local
images and labels alone.

### Downstream use of text is a separate intervention

Training an ordinary learned sigmoid head on a CLIP or SigLIP image encoder
still tests a representation with language-supervised pretraining. Scoring
images directly against ingredient names, initializing label queries from
their text embeddings, or using prompt ensembles introduces the label
semantics again at downstream time. Those modes require separate reporting and
must not be merged with an image-encoder transfer result.

### The head can confound the measurement

An independent per-label head is the clean control. Spatial class-query heads
such as ML-Decoder or Query2Label may reveal small, label-specific evidence that
global pooling misses. Dependency-aware heads such as C-Tran or graph models can
instead succeed through co-occurrence or cuisine context. They remain relevant
research candidates, but cannot by themselves prove visual recognizability.

### Food-specific pretraining is promising but not yet an eligible default

Food2K, Recipe1M+, VLPCook, and FoodSeg/ReLeM demonstrate useful food-domain
pretraining routes. Their supervision is also closer to the target, which
raises stronger provenance, vocabulary-overlap, recipe-overlap, licence, and
semantic-transfer questions. R0.2 did not resolve a proportionate maintained
path for them, so they remain deferred until the inventory's named re-entry
conditions are satisfied.

### R0.2 reduced the catalog through implementation evidence

The [integration inventory](candidate_integration_inventory.md) found that no
candidate currently has the complete per-label AP, raw-score, seed, environment,
and checkpoint-provenance path required by Phase 3. That is a shared
instrumentation prerequisite rather than an architecture-specific failure.

R1 receives verified intake paths for torchvision ResNet-50, the repairable
frozen DINOv2 B/14-register wrapper, and a maintained-library pool of ConvNeXt
Tiny v1, EfficientNetV2-S, and Swin V2 Tiny. Compact DINOv3 and SigLIP 2 Base
FixRes 224 remain conditional on explicit access/dependency/checkpoint and 8 GB
gates. Food-domain, structured-head, exact ConvNeXt V2/FCMAE, and generative
paths are deferred with named re-entry conditions. No `M_ref` is selected.

## Limitations

- No new model was trained or benchmarked for this discovery.
- The catalog is deliberately family-level; it is not an exhaustive list of
  every published backbone or checkpoint.
- Checkpoint access, licence terms, APIs, and dependency compatibility can
  change and must be rechecked when R2 creates a concrete candidate record.
- Food-domain datasets may overlap with web-recipe sources in ways that cannot
  be inferred from paper descriptions alone.
- The one-seed project constraint means later comparison can report
  configuration and temporal sensitivity, but not seed-level stability.

## Related documentation

- [Reference-selector plan](../../../plans/reference_selector_research.md)
- [Comparative model and vocabulary methodology](../../../project_objective/model_comparison_methodology.md)
- [Problem definition](../../../project_objective/problem_definition.md)
- [Current model implementation contract](../../../implementation_details/models.md)
