# Reference-selector research and decision plan

**Created:** 2026-08-12
**Last updated:** 2026-08-27
**Linked macro-section and subphase:** [Subphase 4B, Reference-selector research](../general_plan.md#4b-reference-selector-research)
**Overall status:** In progress

## Objective

Choose and freeze one reference selector, `M_ref`, before Macro-section 3
starts the new `v5` ingredient-selection study. The selector defines the
model-conditional meaning of “image-learnable”; it is a measurement instrument,
not the automatically preferred final benchmark model.

The binding cross-phase design remains in
[model_comparison_methodology.md](../project_objective/model_comparison_methodology.md).
This plan owns only Subphase 4B and the bounded decision needed to release Macro-section 3.

## Relationship to Subphase 4A

Subphases 4A and 4B may cite the same broad discoveries, primary sources, model-family descriptions, implementation audits, and resource evidence. Reusable evidence remains in the research records rather than being copied into both plans.

This plan applies selector-specific criteria only. A 4B exclusion does not remove a model from the 4A experiment shortlist, a 4A shortlist decision does not select `M_ref`, and the final outputs remain independently justified.

## Why the remaining work is intentionally small

R0 already surveyed the relevant model and pretraining families and mapped them
to the repository, the frozen data contract, and the 8 GB constraint. Repeating
that survey as one dossier per candidate would add work without changing the
decision.

The remaining question is narrower: which credible protocol is sufficiently
sensitive, interpretable, reproducible, and affordable to act as the common
selector? The plan therefore limits the decision to at most three finalist
protocols, checks only uncertainties that can change the choice, and combines
the decision record with the handoff.

## Scope

This subphase will:

- reduce the completed R0 inventory to at most three meaningfully different
  finalist protocols;
- compare those finalists using a short set of mandatory gates and an explicit
  decision priority;
- run only bounded technical checks needed to confirm the selected protocol on
  the current WSL environment and 8 GB GPU; and
- freeze the exact `M_ref` protocol, its interpretation boundary, and the
  requirements handed to Macro-section 3.

## Non-goals

This subphase does not:

- reopen broad model discovery unless every credible R0 path fails a mandatory
  gate;
- produce a systematic review or a separate research document for every model
  family considered in R0;
- train or tune candidates as a performance tournament;
- run the `v5` learnability campaign or generate `V_selected`;
- choose the final benchmark winner or replace Subphase 4A;
- implement a new architecture solely to keep it in the selector comparison;
- access the test split or use selected-vocabulary outcomes to choose
  `M_ref`; or
- claim seed-level stability.

## Progress tracker

**Overall status:** In progress
**Current task:** R1 — reduce the completed R0 inventory to a bounded shortlist
and state the selector decision priority.
**Next action:** Retain at most three scientifically distinct protocols that
pass the mandatory gates; do not create candidate dossiers or run comparative
training.

| # | Task | Status | Evidence or result |
| --- | --- | --- | --- |
| R0 | Discover candidate families and map them to the frozen `v5` task, repository, instrumentation needs, and compute boundary. | **Done** | The dated [discovery and integration inventory](../research/discovery/2026-08-22/README.md) retain the broad landscape and repository-backed intake tiers. No selector was chosen. |
| R0.1 | Conduct the broad candidate-landscape discovery. | **Done** | The [2026-08-22 discovery](../research/discovery/2026-08-22/README.md) covers supervised, visual self-supervised, vision-language, food-domain, and structured multi-label families with explicit interpretation boundaries. |
| R0.2 | Map credible candidates to the current `v5` task and integration path. | **Done** | The [candidate and instrumentation inventory](../research/discovery/2026-08-22/candidate_integration_inventory.md) records verified, conditional, and deferred paths plus the common observability gap. |
| R1 | Freeze a shortlist of at most three distinct protocols and the decision priority. | **Pending** | Use the existing R0 evidence; group exclusions by reason instead of producing one dossier per rejected candidate. |
| R2 | Verify the finalists only as needed, compare them, and choose `M_ref`. | **Pending** | Produce one concise comparison table. Use current-source inspection and a bounded load/forward/resource smoke where an uncertainty can change the decision; do not run comparative training. |
| R3 | Freeze the selected protocol and hand it to Macro-section 3. | **Pending** | Record the exact model, weights/pretraining, trainability, transforms, head, resource boundary, limitations, and Phase 3 instrumentation requirements; synchronize the binding methodology and plans. |

## Dependencies and fixed constraints

| Dependency or constraint | Status | Consequence |
| --- | --- | --- |
| FoodOn-first `v5` vocabulary and split | Available | Every finalist targets the same 165-label task and class order. |
| Phase 3 decision profile | Available | `M_ref` must be able to provide named-label train and validation AP evidence after the shared instrumentation is added. |
| One declared seed per configuration | Binding | The later campaign cannot claim seed-level stability. This does not require candidate training during Subphase 4B. |
| Test isolation | Binding | No test result, selected-vocabulary size, or downstream ranking may influence the choice. |
| 8 GB development GPU and thesis schedule | Binding | A protocol that requires disproportionate integration or campaign cost is not eligible. |
| Final benchmark shortlist | Independent and pending | Selecting `M_ref` does not declare the final model winner. |

## Simplified decision protocol

### R0. Completed evidence base

R0 established three useful intake groups:

- verified paths: torchvision ResNet-50, repairable frozen DINOv2 B/14-register,
  and a maintained-library pool containing ConvNeXt Tiny, EfficientNetV2-S,
  and Swin V2 Tiny;
- conditional paths: compact DINOv3 and SigLIP 2, only if their access,
  dependency, checkpoint, interpretation, and 8 GB issues are resolved without
  a one-off pipeline; and
- deferred paths: broken or redundant DenseNet, unavailable food-domain
  checkpoints, structured/dependency heads, and disproportionate generative
  models.

These groups are retained as discovery evidence, not carried forward as a
requirement to compare every entry.

### R1. Bounded shortlist and decision priority

A finalist must pass all of these mandatory gates:

1. **Task fit:** consume the canonical `v5` images and produce 165 independent
   continuous label scores without changing the split or using downstream
   label-text prompts or dependency reasoning.
2. **Evidence path:** have a credible maintained route to named-label train and
   validation AP, reproducible configuration, and run provenance. The common
   instrumentation may be implemented once in Macro-section 3; it need not be
   duplicated for each finalist now.
3. **Operational fit:** fit the 8 GB GPU and available schedule using one
   declared configuration and seed.
4. **Reproducibility:** use traceable weights, transforms, dependencies, and
   trainability state, with no test access or downstream selected-vocabulary
   feedback.

Retain at most three protocols that represent genuinely different measurement
choices, for example supervised continuity, a modern supervised visual model,
and a visually self-supervised pretrained representation. A conditional
candidate enters only if it removes a clear limitation of the verified paths
and its prerequisites can be satisfied proportionately.

Compare finalists in this priority order:

1. scientific meaning for image-based ingredient learnability;
2. capacity to expose useful per-label visual signal under the declared
   protocol;
3. reproducibility and interpretability of pretraining and limitations; and
4. campaign cost and integration risk.

Do not invent a numeric score. If two finalists remain effectively equivalent,
prefer the maintained, lower-cost, easier-to-audit protocol instead of opening
another experiment campaign solely to break the tie.

### R2. Decision-relevant verification and selection

Create one concise table for the finalists containing:

- exact architecture and initialization/pretraining;
- frozen, partially trained, or fully trained backbone policy;
- input transform and independent multi-label head;
- what “learnable” would mean under that protocol;
- current integration and provenance gaps;
- licence/checkpoint constraints; and
- current 8 GB evidence.

Inspect current source for every finalist. Run a bounded technical smoke only
where code inspection or existing evidence cannot settle a decision-relevant
question. A smoke may confirm loading, forward/backward compatibility, output
shape, and peak resource use; it is not comparative accuracy evidence.

Choose one `M_ref` from this table. Briefly group excluded alternatives by the
gate or trade-off that mattered. If no finalist passes, reopen only the failed
gate or intake category rather than repeating the broad discovery.

### R3. Freeze and hand off

Record the selected protocol with enough precision for Macro-section 3 to use
it without reinterpretation:

- model variant and exact weight/checkpoint identity;
- pretraining source and the resulting claim boundary;
- trainability policy, input resolution/transforms, independent output head,
  loss family, and resource boundary;
- known biases and limitations; and
- the shared per-label AP, label-manifest, score-audit, seed, configuration,
  code/environment, and data-provenance artifacts that Phase 3 must implement.

Update
[model_comparison_methodology.md](../project_objective/model_comparison_methodology.md),
[benchmark_decisions.md](../project_objective/benchmark_decisions.md),
[recognizable_ingredient_selection.md](recognizable_ingredient_selection.md),
and the [general plan](../general_plan.md) at this completion checkpoint.
Macro-section 3 then resumes and owns instrumentation, the bounded pilot,
selection thresholds, campaign execution, and `V_selected`.

## Expected artifacts

- this plan with the R1 shortlist rationale and R2 finalist comparison;
- bounded smoke evidence only when required for the choice;
- one binding `M_ref` decision with an explicit interpretation boundary; and
- a synchronized Macro-section 3 handoff.

New topic-research documents are created only when R1 or R2 produces a reusable
finding not already owned by the R0 discovery. They are not mandatory
per-candidate paperwork.

## Validation and completion criteria

This plan is complete only when:

- the completed R0 discovery and inventory remain linked as the evidence base;
- no more than three scientifically distinct finalists were carried into R2;
- the chosen protocol passes every mandatory gate and any decision-relevant
  technical uncertainty has been checked on the current environment;
- the exact `M_ref` protocol and its model-conditional limitation are recorded
  in the binding methodology;
- Macro-section 3 receives the instrumentation and execution handoff; and
- no test outcome, selected-vocabulary result, or comparative candidate-tuning
  campaign influenced the choice.

## Decision and change log

| Date | Change | Rationale |
| --- | --- | --- |
| 2026-08-12 | Created the plan as former Work package 4.6, now Subphase 4B. | Macro-section 3 is deferred until a research-supported reference selector is frozen independently from the final model shortlist. |
| 2026-08-22 | Opened R0.1 broad discovery and R0.2 technical inventory. | The selector search must not be limited to existing ResNet, DenseNet, and DINO implementations; pretraining is part of the selector protocol and changes its interpretation. |
| 2026-08-22 | Completed R0.1. | The discovery found no universal selector; architecture, pretraining, adaptation, downstream label text, and head structure define different measurements. |
| 2026-08-22 | Completed R0.2 and handed intake tiers to the decision stage. | The inventory identified credible and conditional paths, deferred disproportionate ones, and isolated a shared instrumentation/provenance gap without selecting `M_ref`. |
| 2026-08-27 | Compressed the remaining R1–R5 sequence into R1–R3. | R0 already provides broad evidence. The decision only needs a bounded shortlist, decision-relevant verification, one frozen selector, and its Phase 3 handoff; exhaustive candidate dossiers, a numeric rubric, and separate decision/synchronization stages do not advance the objective. |
| 2026-08-27 | Reclassified the plan as Subphase 4B and separated it from Subphase 4A. | Shared discoveries and technical evidence may support both streams, but this plan owns only the selector criteria, `M_ref` decision, and Phase 3 handoff. |
