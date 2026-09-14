# Custom attention-model feature plan

**Created:** 2026-09-07
**Last updated:** 2026-09-08
**Linked macro-section and subphase:** [Subphase 4A, Experimental-model research](../general_plan.md#4a-experimental-model-research), Stage 4A.4
**Overall status:** Done

## Objective and boundary

This feature plan governs the **four research subphases** that lead to one
project-specific attention-model topology. It is not a single literature
review and it does not implement a model. Its outcome is an evidence-backed,
implementation-ready design handoff: one selected topology, its small/medium/
large (S/M/L) scaling rules, initial scale and fallback, input/output and
initialization contracts, and the engineering risks that Macro-section 5 must
test.

The term *custom* means a topology assembled and justified for this project;
it does not claim scientific novelty. The two established families remain
EfficientNetV2-S and MaxViT-T under [4A-D1](../project_objective/experimental_model_portfolio.md).
This stage does not reopen that decision, choose the Phase 3 reference selector
`M_ref`, alter the benchmark, train a candidate, or use validation/test
performance to choose a design.

## Progress tracker

**Overall status:** Done
**Current task:** Research complete; implementation handoff available
**Next action:** After the remaining Data 2.4 readiness checks, prepare the
Phase 5 implementation plan from [4A-D2](../project_objective/experimental_model_portfolio.md#4a-d2--custom-attention-topology)
and its shape, artifact, diagnostic and resource gates. No model is implemented.

| # | Subphase | Status | Evidence or result |
| --- | --- | --- | --- |
| 4A.4.1 | Problem and evidence synthesis | **Done** | [Design brief](../research/topics/custom_attention_model_design/problem_evidence_synthesis.md): reviewed revisions, historical/current reconciliation, R1--R11 mapping, one primary objective, two secondary constraints, failure conditions and Q1--Q7 handoff. |
| 4A.4.2 | Intensive attention-network component research | **Done** | [Component evidence](../research/topics/custom_attention_model_design/attention_component_evidence.md): 32 primary references, eight implementation entries, counterevidence, provisional alternatives, compatibility limits and 4A.4.3 handoff. |
| 4A.4.3 | Component compatibility and architecture synthesis | **Done** | [Compatibility synthesis](../research/topics/custom_attention_model_design/architecture_compatibility_synthesis.md): residual/query routes, explicit feature and label contracts, intact-weight reuse, S/M/L scalar estimates, exclusions and diagnostic integration obligations. |
| 4A.4.4 | Three topology proposals and binding decision | **Done** | [Three proposals](../research/topics/custom_attention_model_design/topology_proposals.md) compare residual, query/context and spatial-mixer/query topologies; [4A-D2](../project_objective/experimental_model_portfolio.md#4a-d2--custom-attention-topology) adopts P2-S, its initialization, same-S frozen fallback and Phase 5 gates. |

## Fixed inputs and design boundaries

Every subphase uses the following records as authoritative inputs; the first
subphase records their visible dates/revisions and any material tension between
them. It may summarize constraints but cannot redefine them.

- [Problem definition](../project_objective/problem_definition.md), [Yummly
  data audit](../project_objective/yummly_data_audit.md), and [ingredient
  vocabulary audit](../project_objective/ingredient_vocabulary_audit.md) define
  the weakly supervised, partially observable recipe-image task.
- [Benchmark decisions](../project_objective/benchmark_decisions.md) fix the
  image-only, 165-label `v5` task, split, output semantics, metrics and test
  isolation.
- [Comparative methodology](../project_objective/model_comparison_methodology.md)
  fixes the later Q1--Q4 comparison boundary, common vocabulary projections,
  random-reduction controls, and one-seed limitation.
- The [problem-to-model requirements matrix](../research/discovery/2026-08-28/problem_model_requirements.md), [candidate
  collection](../research/topics/experimental_model_candidates/README.md), and
  [4A-D1 portfolio](../project_objective/experimental_model_portfolio.md)
  provide the reusable evidence, exclusions, established-family strengths and
  open gaps.
- The current [model contract](../implementation_details/models.md) defines
  raw `num_classes` logits, model-owned transforms, serialization and
  visualization hooks. It is an implementation fact, not a restriction against
  proposing a future supported extension.

The custom protocol must remain image-only at inference and produce one raw
logit per ordered class. It must preserve the common aspect-preserving,
224-by-224 padded input policy unless a binding portfolio revision explicitly
changes that comparison contract. It cannot silently obtain validation/test
statistics, recipe text, ingredient names, cuisine metadata or label graphs.
Any train-only label-dependency mechanism must name its source, its control,
and the interpretation limit: it is not direct visual evidence.

The 8 GB development-GPU boundary, one declared seed per configuration, and
limited thesis schedule are fixed constraints. During this theoretical stage,
parameter, activation and compute figures are estimates; only Macro-section 5
may establish local loading, gradient, peak-memory or throughput facts.

## Evidence flow and durable artifacts

The feature plan is the operational source of truth for stage status. Evidence
is retained in the indexed [custom-design research collection](../research/topics/custom_attention_model_design/README.md),
rather than copied into this plan. Subphases 4A.4.1--4A.4.3 created its index,
design brief, component record and compatibility synthesis. The final subphase
adds the topology proposals and updates the same collection:

| Research artifact | Produced by | Owns |
| --- | --- | --- |
| `problem_evidence_synthesis.md` | 4A.4.1 | Current constraints, established-family gaps, bounded design objective, and open-question register. |
| `attention_component_evidence.md` | 4A.4.2 | Claim-to-source evidence for retained and rejected component choices. |
| `architecture_compatibility_synthesis.md` | 4A.4.3 | Interfaces, tensor resolutions, data flow, initialization options, incompatibilities and pre-implementation estimates. |
| `topology_proposals.md` | 4A.4.4 | The three comparable proposals, selection rationale and the 4A.4-to-Phase-5 handoff. |

Each component evidence entry records: the component's role in the design
brief; original paper and, when relied on, official or maintained code/weights;
the supported claim and evidence type; source-task transfer boundary; interface
and resource consequences; known counterevidence or limitations; and a link to
the proposal that reuses it. A proposal cites those entries by stable heading or
source identifier instead of reproducing a disconnected bibliography. The
selected decision itself belongs in the existing
`project_objective/experimental_model_portfolio.md` record; it links the
evidence and proposal rather than duplicating them.

Unreviewed search results, extraction notes and calculations belong under
`src_scratches/`. The collection retains only reviewed, interpretable findings.

## 4A.4.1 — Problem and evidence synthesis

### Question

Which unresolved, image-relevant limitation of the established portfolio is
worth addressing with one custom attention architecture, without changing the
task or turning label co-occurrence into a hidden answer source?

### Work

1. Re-read the fixed inputs above, including the selected EfficientNetV2 and
   MaxViT dossier details and their 4A-D1 handoff.
2. Open the indexed research collection and write `problem_evidence_synthesis.md`.
   Separate verified task facts, inherited research evidence, project
   interpretation, assumptions and open questions.
3. Map the unresolved design gaps to the existing R1--R11 requirement rows.
   State what the selected established families already control well, so the
   custom model does not merely recreate either one under a new name.
4. Formulate one bounded primary design objective and, at most, two secondary
   constraints. It must name the expected visual mechanism, its likely failure
   mode, and the simplest later comparison that could weaken its interpretation.

### Completion and handoff

This subphase is complete when the design brief explains *why* a custom model
is warranted, identifies the minimum functions the architecture must supply,
and records questions that the component research must answer. It must not
choose components, widths, depths, a loss, or a topology.

### Completion checkpoint — 2026-09-08

**Status:** Done. The [brief](../research/topics/custom_attention_model_design/problem_evidence_synthesis.md)
records the mandatory input revisions at `e0cddbe`, reconciles historical audit
counts with the active `v5` contract, and maps the remaining portfolio question
to R1--R11. Its primary objective is ingredient-specific aggregation of spatial
evidence while retaining dish context; computation and traceable information
use are the two secondary constraints. A same-trunk pooled control and failure
conditions make the hypothesis testable without scheduling another campaign.
Q1--Q7 were passed to component research. No topology or loss was selected.

## 4A.4.2 — Intensive attention-network component research

### Question

Which established attention-network components are sufficient and justified to
serve the functions in the approved design brief, and which attractive
components should be excluded?

### Work

Research only the component families needed for a complete end-to-end route:

1. input representation and early spatial processing: convolutional, patch or
   hybrid tokenization; feature scales; positional information;
2. spatial interaction: local, windowed, grid, global, axial, channel or
   cross-attention, including their small-image and activation-memory limits;
3. readout: global pooling, class/query attention or another independent-logit
   decoder, with any label-dependency boundary made explicit; and
4. supporting blocks: residual paths, normalization, feed-forward layers,
   fusion, regularization, initialization and trainability choices that are
   necessary for the selected route.

For every retained component, inspect the original research source and an
official or maintained implementation when the proposed design depends on its
behavior. Seek direct food/multi-label evidence and independent mechanism
evidence when available; retain an evidence gap rather than filling it with an
unrelated headline score. Record the source task, input regime, supervision,
metric and comparison context before making a transfer interpretation.

Stop when there is one coherent, evidence-supported component route for each
design function and a recorded exclusion for material alternatives. This is not
an exhaustive catalog of every attention mechanism and does not create a
candidate dossier for each layer type.

### Completion and handoff

`attention_component_evidence.md` must identify compatible candidates and
known incompatibilities, distinguish verified mechanism facts from the local
recommendation, and state the questions that only tensor-level synthesis can
resolve. Component choice remains provisional until the next subphase.

### Completion checkpoint — 2026-09-08

**Status:** Done. The [component record](../research/topics/custom_attention_model_design/attention_component_evidence.md)
covers a complete functional route: spatial convolutional/hybrid features,
optional bounded spatial interaction or fusion, residual/class-query readout,
supporting layers and raw logits. It preserves simpler alternatives and
exclusions, primary-source reading depth and transfer limits, inspected
implementation revisions, and arithmetic memory examples distinct from GPU
measurements.

The original Nutrition5K comparison supplies negative evidence against assuming
ML-Decoder improves every encoder. The source audit also distinguishes learned
Query2Label embeddings from fixed random standard ML-Decoder queries. These
findings are linked back from the existing C4 research; they do not change the
adopted established-family portfolio.

**Work discovered for 4A.4.3:** resolve feature scales/projections, context and
position paths, mask transport or all-canvas processing, exact query/group
semantics across vocabulary sizes, and honest weight reuse/normalization.
The CSRA code-reuse provenance remains conditional. Reference decoder mask/API
behavior and actual peak memory belong to Phase 5 if the relevant route is
selected. No candidate was executed, no weights downloaded and no validation
or test outcomes used. The [handoff](../research/topics/custom_attention_model_design/attention_component_evidence.md#handoff-to-4a43)
is ready; 4A.4.3 and 4A.4.4 remain Pending.

## 4A.4.3 — Component compatibility and architecture synthesis

### Question

Can the retained components form a reproducible, interpretable and plausibly
8-GB-feasible network before any model is proposed or implemented?

### Work

1. Define the end-to-end representation path from padded RGB image to `L=165`
   raw logits. For each transition, state tensor layout, channel/embedding
   dimension, spatial resolution or token count, fusion operation and residual
   path.
2. Resolve interactions between tokenization, positional information, attention
   grouping, multi-scale fusion, normalization, decoder and output head. Mark
   which interactions are inherited from evidence and which remain a project
   hypothesis.
3. Compare initialization paths honestly: a new topology may start from a
   documented random initialization, or reuse only weights whose architecture
   and licence actually permit it. It must not claim pretrained benefit for
   untransferable pieces. State the consequent comparison interpretation.
4. Define topology-preserving S/M/L scaling axes (for example depth, width,
   embedding dimension, head count or repeated blocks), provisional parameter
   bands relative to the selected families, and an initial feasibility order.
5. Estimate parameter count, largest activations and dominant attention cost at
   the fixed input. Reject combinations with no credible 8 GB route. Estimates
   are planning evidence, not successful resource validation.

### Completion and handoff

`architecture_compatibility_synthesis.md` must provide one or more compatible
design routes with explicit interfaces and a bounded scaling envelope. It must
also retain rejected combinations and their reason. Only then may the next
subphase package routes as three full model proposals.

### Completion checkpoint — 2026-09-08

**Status:** Done. The [synthesis](../research/topics/custom_attention_model_design/architecture_compatibility_synthesis.md)
provides two complete readout routes, a conditional late spatial operator,
source-derived feature taps, explicit all-canvas/position conventions and
learned one-query-per-label semantics. An intact EfficientNetV2-S trunk anchors
honest weight reuse and the bounded S/M/L envelope; MaxViT remains a separately
identified representation alternative. These are provisional research
specifications, not three final proposals or a portfolio decision.

**Verification:** a [standard-library scalar calculator](../../src_scratches/custom_attention_design/estimate_design_envelope.py)
reconciles the trunk with upstream parameter metadata, verifies stride-derived
taps and checks head arithmetic for multiple vocabulary sizes. It constructs
no model/tensor and accesses no dataset, checkpoint or GPU. Counts distinguish
new versus total parameters, dominant MACs and individual storage examples;
no 8 GB peak-memory claim is made.

**Newly discovered implementation work:** the existing feature-factorization
consumer assumes a classifier applicable to standalone concept vectors. Query
and attentive fused heads need an explicitly qualified adapter or a
capability-aware dashboard path; exposing an unrelated linear layer is not a
valid fix. Phase 5 also owns exact transform realization, checkpoint/licence,
gradient, reconstruction and full-step resource checks. No code integration
or training was performed. The [4A.4.4 handoff](../research/topics/custom_attention_model_design/architecture_compatibility_synthesis.md#handoff-to-4a44)
is ready, and 4A.4.4 remains Pending.

## 4A.4.4 — Three topology proposals and binding decision

### Question

Which one of three coherent, distinct custom topologies best tests the bounded
design objective while remaining reproducible and feasible enough to hand to
Macro-section 5?

### Work

Produce exactly three topology-level proposals. Each proposal must state:

- a distinct mechanism-specific hypothesis, affected requirement rows and a
  result that would weaken it;
- complete image-to-logit flow, spatial resolutions/tokens, attention locations
  and evidence path;
- output/readout, probability semantics and any label-structure boundary;
- input transform, initialization/pretraining and trainability contract;
- retained components, rejected alternatives and the source-backed reason for
  each choice;
- one topology-preserving S/M/L scale definition, initial scale and fallback;
- parameter/activation/compute estimate, dependencies, licence and expected
  integration work; and
- the narrowest necessary later control or ablation. Macro-sections 5--7 decide
  whether and how that control is run; the proposal must not silently schedule
  an additional training campaign.

Compare the proposals qualitatively, without inventing an aggregate numeric
score. A selected topology must pass every gate below:

1. it addresses the 4A.4.1 primary objective more directly than a renamed
   EfficientNetV2 or MaxViT variant;
2. it remains compatible with the frozen image-only logits and comparison
   contract;
3. its major components, initialization and dependencies have auditable
   provenance;
4. it has a credible, but still unmeasured, route under the 8 GB constraint;
5. its S/M/L variants preserve the same topology and scientific hypothesis; and
6. it names a falsifiable claim and avoids presenting label co-occurrence or
   context as proof of direct ingredient visibility.

### Completion and handoff

The selected design is recorded as a new decision in
`experimental_model_portfolio.md`, not as an implementation fact. Update this
plan, the parent 4A plan, the topic index and the general plan at that completed
subphase. The binding handoff to Macro-section 5 contains the selected topology,
initial and fallback scale, exact interfaces, initialization, dependencies,
required shape/gradient/preprocessing/checkpoint/resource smoke tests, and
remaining engineering risks.

### Completion checkpoint — 2026-09-08

**Status:** Done. [P1/P2/P3](../research/topics/custom_attention_model_design/topology_proposals.md)
each have a distinct graph/hypothesis, complete image-to-logit specification,
source-linked retained/rejected components, S/M/L axes, initial/fallback,
resource estimates, failure conditions and the narrowest later control.
The qualitative six-gate comparison supports **P2-S**, a dual-scale learned
ingredient-query readout plus pooled context, adopted in 4A-D2. Its intact
EfficientNetV2-S trunk and new modules have distinct initialization provenance.
The fallback remains S with frozen-encoder adaptation; it is not a smaller
scale or equivalent full-tuning experiment. M/L are optional capacity designs,
not an extra campaign.

**Verification:** bounded recheck of decisive primary papers and official
source/API documentation; rerun of the scalar calculator, without models,
tensors, data, weights or GPU access; documentation/link and diff checks.
All custom-plan, proposal, portfolio and index links resolve. The scan also
retains three unrelated, pre-existing broken references in the general plan's
Data entries (two image-loading-document links and one DataModule test link),
confirmed present at `993a166`; no new unresolved link was introduced.
The input brief's benchmark and methodology contracts remain unchanged.
No model-performance or seed-stability conclusion is drawn.

**Engineering handoff:** exact transform realization, artifact/initialization
and configuration round trip, shape/gradient/label-order tests, normalization
and attention-backend checks, full-step memory/timing, and capability-aware
visualization integration. The custom selection does not release 4B/Phase 3
or satisfy Data 2.4. All four research subphases and this feature plan are
complete; the parent 4A plan and general tracker are synchronized.

## Dependencies, assumptions and open decisions

| Item | Status | Consequence |
| --- | --- | --- |
| 4A-D1 established-family portfolio | Available | Supplies components, gaps, comparison controls and the common input/output starting boundary. |
| Data Work package 2.4 runtime smoke completion | In progress | Does not block theory; blocks standard benchmark-model execution. |
| Subphase 4B and Macro-section 3 | Independent/in progress | Neither selects the custom topology; their future shared vocabulary changes the later ablation, not this design brief. |
| Exact custom initialization policy | Adopted in 4A-D2 | Intact pretrained feature weights plus explicitly new query/context modules; Phase 5 verifies artifact loading and initialization without overwriting reused weights. |
| Exact S/M/L parameter bands | Specified; scalar estimates only | P2-S is initial; frozen S is the bounded adaptation fallback, M/L are capacity options. Phase 5 verifies counts and measures feasibility. |
| Local feasibility, optimization stability and calibration | Unverified | Cannot be inferred from literature or estimates and cannot decide the research selection through unrun performance. |

## Validation and completion criteria

This plan is complete only when:

- all four subphases retain the reviewed evidence in the indexed topic
  collection and preserve a source-to-decision trail;
- the design brief, component evidence and compatibility synthesis distinguish
  facts, interpretations, recommendations and uncertainties;
- exactly three topology proposals exist, each with topology-preserving S/M/L
  rules and an explicit failure condition;
- one custom topology and initial scale are adopted in the portfolio with a
  bounded fallback and Macro-section 5 handoff;
- no candidate implementation, HPO, validation/test outcome or unrecorded
  source influenced the selection; and
- all plan/index/portfolio/general-plan links resolve and no planned model is
  represented as implemented.

## Decision and change log

| Date | Change | Rationale |
| --- | --- | --- |
| 2026-09-07 | Created the dedicated 4A.4 feature plan with four research subphases. | The parent 4A plan requires a staged problem synthesis, component investigation, compatibility analysis, and three-proposal decision rather than one undifferentiated custom-model search. |
| 2026-09-08 | Completed 4A.4.1 and 4A.4.2; compatibility synthesis is ready to start. | The indexed brief and component evidence now carry an explicit objective, source/counterevidence trail, provisional routes and open interface questions. Negative food results and query-initialization corrections prevent inherited optimism from becoming a design assumption. |
| 2026-09-08 | Completed 4A.4.3; three-proposal comparison is ready to start. | Complete tensor/initialization routes and reproducible scalar estimates replace unresolved component combinations. Padding/grouping are explicit, and the visualization compatibility limitation is handed to Phase 5 without implementing a model or choosing a topology. |
| 2026-09-08 | Completed 4A.4.4 and this feature plan; adopted P2-S in 4A-D2. | Exactly three source-linked topologies, topology-preserving scales and explicit controls support the decision; implementation/resource validation remains Phase 5 work, and no extra training campaign or selector decision is implied. |
