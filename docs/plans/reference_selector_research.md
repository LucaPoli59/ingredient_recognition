# Reference-selector research and decision plan

**Created:** 2026-08-12  
**Last updated:** 2026-08-22
**Linked macro-section and work package:** [Macro-section 4, Work package 4.6](../general_plan.md#4-model-research)  
**Overall status:** In progress

## Objective

Choose and freeze the reference selector, M_ref, used to define the
model-conditional meaning of an image-learnable ingredient before Macro-section
3 starts the new v5 selection study. M_ref is a measurement instrument for
vocabulary selection, not the automatically preferred final model category.

The binding cross-phase design remains in
[model_comparison_methodology.md](../project_objective/model_comparison_methodology.md).
This plan records how the prerequisite M_ref decision will be researched,
compared, justified, and handed off.

## Scope

This work package will:

- establish which current or research-backed model categories are eligible to
  act as the selector through a broad candidate-landscape discovery, not only
  through the models already implemented in this repository;
- freeze an evidence protocol and decision rubric before candidate-specific
  analysis;
- research each eligible candidate's scientific fit, per-label diagnostic
  capability, representativeness, resource cost, and integration risk;
- inspect the current code and configuration path needed to produce comparable
  multi-label per-label trajectories; and
- record the decision and its limits in the project objectives, then release
  Macro-section 3 to freeze its selector-specific protocol.

## Non-goals

This work package does not:

- tune a candidate, run the v5 learnability campaign, or generate V_selected;
- choose the final benchmark winner or replace Work package 4.5's broader
  model-shortlist decision;
- compare test-set outcomes, access the test split, or use downstream
  selected-vocabulary results to choose M_ref;
- add model implementations solely to make them selector candidates; or
- claim seed-level stability. The project has budgeted one declared seed per
  selection configuration.

Existing ResNet, DenseNet, and DINO-related paths are starting evidence, not a
closed candidate list. A candidate is not adopted merely because it appears in
the broad discovery.

## Progress tracker

**Overall status:** In progress
**Current task:** R1 — freeze the eligibility gates, evidence labels, and tie
rule before candidate-specific evaluation.
**Next action:** Predeclare the R1 hard gates and qualitative rubric, including
the shared instrumentation gate, supervised-representative rule, conditional-
candidate deadline, maximum integration effort, and measured 8 GB smoke
contract.

| # | Task | Status | Evidence or result |
| --- | --- | --- | --- |
| R0 | Complete the broad candidate discovery and map the frozen v5 data contract, current model inventory, available logging, and eligible candidate families. | **Done** | The dated [discovery and integration inventory](../research/discovery/2026-08-22/README.md) retain the broad landscape and the repository-backed reduction. No candidate is recommended. |
| R0.1 | Conduct a broad, catalog-only discovery of architecture, training, and pretraining families that could act as an ingredient-learnability selector. | **Done** | [2026-08-22 discovery](../research/discovery/2026-08-22/README.md) catalogs supervised CNN/transformer, visual SSL, generic VLM, food-domain, and multi-label-head families with explicit claim boundaries. |
| R0.2 | Map candidates with a credible path to the current v5 task and record their instrumentation and integration state. | **Done** | The [candidate and instrumentation inventory](../research/discovery/2026-08-22/candidate_integration_inventory.md) records the common observability gaps, dependency/checkpoint/licence paths, measured-versus-unverified compute evidence, intake tiers, and explicit re-entry conditions. |
| R1 | Freeze the hard eligibility gates, evidence sources, and qualitative decision rubric. | **Pending** | — |
| R2 | Create source-backed candidate evidence records and inspect their current integration paths. | **Pending** | — |
| R3 | Compare eligible candidates against the frozen rubric and record trade-offs, exclusions, and residual risks. | **Pending** | — |
| R4 | Freeze and document M_ref, including its role, limitations, and handoff requirements. | **Pending** | — |
| R5 | Synchronize the decision with Macro-section 3 and release its deferred protocol work. | **Pending** | — |

## Dependencies and assumptions

| Dependency or assumption | Status | Consequence |
| --- | --- | --- |
| The FoodOn-first v5 base vocabulary and split contract are frozen. | Available | Candidates are assessed for the same multi-label task, not for different vocabularies. |
| Macro-section 3's learnability decision profile is retained. | Available | The selector must support train-AP and validation-AP per-label trajectories, fixed-policy F1 diagnostics, and complete provenance. |
| The final benchmark model shortlist is not yet frozen. | Pending | M_ref can be selected without declaring a final model winner; Work package 4.5 remains separate. |
| Selection training has one declared seed per configuration. | Binding constraint | The decision must favor protocol observability and reproducibility, while Macro-section 3 reports temporal/configuration sensitivity rather than seed-level stability. |
| Test outcomes remain unavailable. | Binding constraint | No test metric, threshold, or downstream ranking may influence M_ref. |

## Research protocol

### R0. Candidate landscape and instrumentation inventory

#### R0.1 Broad candidate-landscape discovery

Conduct an extensive, catalog-only external discovery before constraining the
selector to the repository's current implementations. The discovery must cover
candidate families relevant to imbalanced multi-label ingredient recognition
under partial visual observability, including where justified:

- supervised convolutional baselines and modern convolutional families;
- vision transformers and hybrid convolutional-transformer architectures;
- self-supervised or masked-image vision foundation models;
- contrastive vision-language foundation models that can be adapted to the
  canonical image-to-multi-label task;
- food- or ingredient-domain pretraining, when its sources, licence, and
  transfer boundary are traceable; and
- multi-label or label-query heads that may materially improve per-label
  evidence without changing the frozen vocabulary or split.

The purpose is to make the candidate set explicit, not to select a model from a
single performance claim. Each discovery entry must distinguish a reported
architecture or checkpoint from a credible candidate for this repository under
the available compute and implementation constraints. Follow the project
discovery rule: inspect the two preceding discovery records when they exist,
then retain the source catalogue and broad findings in a new dated
`docs/research/discovery/<date>/` record.

Treat pretraining as a declared part of a possible M_ref protocol, not as an
automatic advantage. For every relevant pretraining family, record:

1. the source data and training objective;
2. whether the selector would be frozen, linearly probed, or fine-tuned;
3. the information it may contribute beyond the v5 labels and images; and
4. the consequent interpretation boundary: a label would be judged learnable
   relative to the declared pretrained selector, rather than demonstrably
   learnable from scratch.

The later decision must weigh whether prior visual or vision-language knowledge
helps reveal a visually recognizable ingredient against the risk that it turns
the selector into a measure of transferred semantic knowledge. It must state
that trade-off explicitly and must not infer data leakage without evidence.

#### R0.2 Candidate and instrumentation inventory

After R0.1 and before candidate-specific evaluation, map each candidate family
with a credible project path:

- its implementation status, checkpoint/pretraining dependencies, licence or
  access constraints, and expected compute;
- its canonical input, output logits or scores, and multi-label loss path;
- whether the maintained training path can log label order, train and
  validation AP trajectories, fixed-policy F1 diagnostics, raw scores, and the
  configuration/provenance needed by Macro-section 3;
- how it connects to the frozen v5 DataModule and transforms; and
- any missing engineering work that would make its use as a selector
  disproportionate or non-reproducible.

The inventory may include existing ResNet, DenseNet, and DINO-related paths,
but it must not presume that any is selected. A newly discovered family becomes
eligible for R2 only when the broad research record and a credible maintained
integration path are available.

**Completed result.** The dated
[candidate and instrumentation inventory](../research/discovery/2026-08-22/candidate_integration_inventory.md)
found that the shared training path is the first material gate: it does not yet
persist per-label train/validation AP trajectories, bounded raw scores, an
explicit label manifest, a declared reproducible seed, code/environment
identity, or exact external-checkpoint provenance. No current wrapper can pass
the intended evidence gate without this common layer.

R1 receives:

- verified intake paths for torchvision ResNet-50, frozen DINOv2 B/14-register
  after bounded reproducibility repair, and a maintained-library pool of
  ConvNeXt Tiny v1, EfficientNetV2-S, and Swin V2 Tiny;
- conditional intake paths for compact DINOv3 and SigLIP 2 Base FixRes 224,
  subject to access/dependency/checkpoint and measured 8 GB gates; and
- deferred paths with explicit re-entry conditions for the currently broken or
  redundant DenseNet wrapper, exact ConvNeXt V2/FCMAE, other masked-image and
  food-domain checkpoints, structured/dependency heads, and generative food
  VLMs.

This is an integration-credibility reduction, not a model ranking. R1 must
decide the representative count and evidence rules before R2 inspects
candidate-specific performance evidence.

### R1. Hard eligibility gates and comparison rubric

Freeze the following gates before writing a candidate recommendation. A
candidate must:

1. train end-to-end on the canonical v5 multi-label image task without changing
   the vocabulary or split;
2. expose per-label continuous scores and support epoch-level train and
   validation AP trajectories, with the declared label order and full run
   provenance;
3. fit the available compute and schedule for a bounded, single-seed
   selector campaign;
4. have a credible maintained configuration, loss, transform, checkpoint, and
   logging path; and
5. make the decision without test-set access or implicit tuning on
   selected-vocabulary outcomes.

For candidates that pass the gates, predeclare how the following dimensions
are judged. The rubric must make trade-offs visible rather than reduce an
unjustified decision to one accuracy number:

| Dimension | Research question |
| --- | --- |
| Scientific fit | Does the architecture and pretraining/fine-tuning regime plausibly measure visual learnability of imbalanced multi-label ingredients rather than an unrelated proxy? |
| Evidence fidelity | Can the maintained path deliver the per-label optimization, generalization, threshold-diagnostic, support, and provenance evidence required by the Phase 3 profile? |
| Representativeness | Is it a defensible reference instrument for the later model portfolio without being treated as the final winner? |
| Resource robustness | Can its expected training, storage, and diagnostic costs be completed within the declared single-seed budget? |
| Integration maturity | Are data contract, loss weighting, transforms, checkpoints, and experiment tracking sufficiently understood and maintainable? |
| Interpretability of limitations | Can likely architecture, pretraining, or resolution biases be stated so that M_ref-conditional vocabulary decisions are not overstated? |

R1 must also predeclare the scoring scale or qualitative evidence labels and
the rule for handling a tie. It may not alter those rules after a candidate's
evidence has been inspected.

### R2. Focused evidence to collect

For every eligible candidate, collect and cite:

1. primary architecture and, where relevant, pretraining sources;
2. primary or authoritative evidence relevant to multi-label visual
   classification, transfer/fine-tuning, calibration, resolution, and
   class-imbalance behavior;
3. current repository evidence for the actual data, loss, metrics, logging,
   configuration, checkpoint, and compute path; and
4. research evidence about practical failure modes that could make a label
   appear non-learnable only because of the selector's limitations.

Store reusable source-backed candidate findings under
docs/research/topics/reference_selector/ and link the resulting records here.

### R3. Candidate comparison and bounded engineering checks

Create a candidate matrix containing the frozen rubric, source links,
repository evidence, expected resources, exclusions, and unresolved risks.
Separate facts from inferences.

An engineering smoke check is allowed only when it resolves a concrete
integration uncertainty that sources and code inspection cannot resolve. It
must use the canonical data contract, preserve a reproducible configuration,
avoid the test split, and be reported as engineering evidence rather than
comparative performance evidence. No candidate is selected from an untracked
single chart or a downstream selected-vocabulary result.

### R4. Decision record

The recommendation must name one M_ref and document:

- why it passed every hard gate and why the chosen trade-offs fit the
  selection objective;
- candidates excluded or deferred, including the evidence and limitation that
  drove the exclusion;
- the selector's architecture, initialization/pretraining state, maintained
  training path, and assumptions that Macro-section 3 must freeze; and
- the boundary of the claim: V_selected will be learnable relative to this
  declared M_ref protocol and not universally learnable for every architecture.

Record the binding result in
[model_comparison_methodology.md](../project_objective/model_comparison_methodology.md)
and [benchmark_decisions.md](../project_objective/benchmark_decisions.md). Keep
the full research rationale in the topic records and this plan.

### R5. Handoff to Macro-section 3

After R4, update the project plan and
[recognizable_ingredient_selection.md](recognizable_ingredient_selection.md)
to replace the M_ref dependency with the chosen selector. Macro-section 3 can
then freeze its configuration panel, logging contract, one-seed protocol,
bounded pilot, and decision-profile thresholds before training begins.

The broader model shortlist remains in Work package 4.5. It may include M_ref,
but it must record a separate final-model hypothesis and must not retroactively
change the frozen selector.

## Validation and completion criteria

This plan is complete only when:

- the candidate inventory and frozen rubric are retained;
- every considered candidate has source-backed and repository-backed evidence,
  with exclusions explained;
- one M_ref passes all gates, has an explicit limitation statement, and is
  recorded in the binding methodology and benchmark decision record;
- the M_ref handoff requirements are linked from the Macro-section 3 plan; and
- no test outcome or selected-vocabulary result was used for the decision.

## Decision and change log

| Date | Change | Rationale |
| --- | --- | --- |
| 2026-08-12 | Created Work package 4.6 plan. | Macro-section 3 is deferred until a research-supported reference selector is frozen independently from the final model shortlist. |
| 2026-08-22 | Opened R0.1 broad candidate-landscape discovery and R0.2 technical inventory. | The selector search must not be limited to existing ResNet, DenseNet, and DINO implementations; pretraining is evaluated as an explicit part of the selector protocol and its interpretation boundary. |
| 2026-08-22 | Completed R0.1 and started R0.2. | The dated discovery found no universal selector: architecture, pretraining, adaptation, downstream label text, and head structure define different measurements. The non-ranked handoff set now requires repository and compute verification before R1. |
| 2026-08-22 | Completed R0.2 and handed the intake tiers to R1. | Static code, artifact, dependency, official-source, and workstation inspection identified a shared instrumentation/provenance prerequisite; retained only credible or explicitly conditional paths and gave every deferred family a re-entry condition without choosing `M_ref`. |
