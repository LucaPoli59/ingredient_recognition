# Reference-selector research and decision plan

**Created:** 2026-08-12  
**Last updated:** 2026-08-12  
**Linked macro-section and work package:** [Macro-section 4, Work package 4.6](../general_plan.md#4-model-research)  
**Overall status:** Pending

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
  act as the selector;
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

## Progress tracker

**Overall status:** Pending  
**Current task:** Not started  
**Next action:** Complete R0 and R1: register the available candidate families,
their current instrumentation status, and the frozen decision rubric before
candidate-specific research begins.

| # | Task | Status | Evidence or result |
| --- | --- | --- | --- |
| R0 | Map the frozen v5 data contract, current model inventory, available logging, and eligible candidate families. | **Pending** | — |
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

### R0. Candidate and instrumentation inventory

Record each realistically available candidate family before external comparison:

- its implementation status, checkpoint/pretraining dependencies, licence or
  access constraints, and expected compute;
- its canonical input, output logits or scores, and multi-label loss path;
- whether the maintained training path can log label order, train and
  validation AP trajectories, fixed-policy F1 diagnostics, raw scores, and the
  configuration/provenance needed by Macro-section 3;
- how it connects to the frozen v5 DataModule and transforms; and
- any missing engineering work that would make its use as a selector
  disproportionate or non-reproducible.

The initial inventory may include existing ResNet and DINO-related paths, but
it must not presume that either is selected. A new family is eligible only when
the research record and a credible maintained integration path are available.

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

Before starting a new model-discovery source, review the two preceding
discoveries when they exist, following the project research governance. Store
reusable source-backed findings under
docs/research/topics/reference_selector/ and link the resulting records here.
Use a dated discovery only if the candidate set must be expanded beyond the
existing model-research evidence.

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
