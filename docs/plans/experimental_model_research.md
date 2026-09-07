# Experimental-model research plan

**Created:** 2026-08-28
**Last updated:** 2026-09-07
**Linked macro-section and subphase:** [Subphase 4A, Experimental-model research](../general_plan.md#4a-experimental-model-research)
**Overall status:** In progress

## Objective

Build a traceable, literature-supported experiment portfolio for Macro-sections
5–7. Subphase 4A will:

1. discover and retain three to five scientifically distinct, accessible model
   families with credible use in this problem or a clearly transferable adjacent
   problem;
2. investigate every retained family through the same deep-research protocol;
3. select two established families for implementation and controlled comparison;
   and
4. design one project-specific attention model through a separate, dedicated
   feature plan that first compares three coherent architecture proposals.

The intended final portfolio is therefore **two literature-derived model
families plus one selected custom attention architecture**. Required non-visual
and simple visual baselines remain part of the later benchmark contract, but do
not count toward the three-to-five discovery candidates unless they also earn a
place as an experimental family.

This plan owns the research sequence, evidence handoffs, and shortlist decision.
It does not own model implementation, hyperparameter optimization, training, or
final test comparison.

## Relationship to Subphase 4B

Subphase 4A selects the model categories used in the thesis experiments.
Subphase 4B independently selects the single reference selector `M_ref` used by
Macro-section 3. They may cite the same architecture, pretraining, checkpoint,
licence, integration, and resource evidence, but they apply different decision
criteria and own different outputs.

A model excluded from one subphase is not automatically excluded from the other.
The 4A shortlist does not select `M_ref`, and the 4B decision does not determine
the experiment portfolio.

## Progress tracker

**Overall status:** In progress
**Current task:** 4A.4 — custom attention-model research
**Next action:** Execute 4A.4.1 in the dedicated
[custom attention-model feature plan](custom_attention_model.md): re-read the
problem and accumulated evidence, then create the bounded design brief before
component research.

| # | Task | Status | Evidence or result |
| --- | --- | --- | --- |
| 4A.1 | Broad model discovery and three-to-five-family handoff | **Done** | The 2026-08-28 discovery records the input revisions, requirements matrix, five new retained family/protocol candidates, the already-used ResNet/DINOv2 baseline anchors, grouped exclusions, and the formal 4A.1 → 4A.2 handoff. |
| 4A.2 | Deep research for every retained family | **Done** | Five normalized dossiers and a common comparative synthesis are recorded in [`research/topics/experimental_model_candidates/README.md`](../research/topics/experimental_model_candidates/README.md); no implementation or training was part of this stage. |
| 4A.3 | Select two established model families | **Done** | [4A-D1](../project_objective/experimental_model_portfolio.md) adopts EfficientNetV2-S and MaxViT-T with common input/readout starting protocols, explicit fallbacks, candidate dispositions, hypotheses, and 4A.4/Phase 5 handoffs. |
| 4A.4 | Research and select the custom attention model | **In progress** | The [dedicated feature plan](custom_attention_model.md) separates problem/evidence synthesis, component research, compatibility synthesis, and three topology proposals before one binding selection. |

## Mandatory input and re-reading gate

Every 4A stage must use the current project problem rather than a generic image
classification brief. Before 4A.1 begins, read the following inputs in this
order and record the exact document revisions or last-updated dates in the new
discovery:

1. [Yummly data audit](../project_objective/yummly_data_audit.md): data lineage,
   low image resolution, label imbalance, duplicate evidence, cuisine shortcuts,
   and supervision ambiguity.
2. [Problem definition](../project_objective/problem_definition.md): task,
   scope, non-goals, research questions, evaluation principles, and success
   criteria.
3. [Benchmark decisions](../project_objective/benchmark_decisions.md): frozen
   `v5` target, split, vocabulary, test-isolation, metric, and compatibility
   contracts.
4. [Comparative methodology](../project_objective/model_comparison_methodology.md):
   separation of model-category comparison, vocabulary ablation, random controls,
   and local adaptation.
5. [Candidate vocabulary audit](../project_objective/ingredient_vocabulary_audit.md):
   target granularity, support, semantic ambiguity, and the boundary between
   taxonomy repair and later learnability selection.
6. [Current model implementation contract](../implementation_details/models.md):
   models, interfaces, transforms, known integration constraints, and historical
   continuity available in the repository.
7. The [2026-08-02 broad discovery](../research/discovery/2026-08-02/README.md)
   and the [2026-08-22 selector-oriented discovery](../research/discovery/2026-08-22/README.md),
   including their source catalogs. Reuse evidence, not their old priorities or
   selector-specific dispositions.
8. This plan and the current [general plan](../general_plan.md), so the active
   scope, dependencies, and handoff remain explicit.

Before 4A.4 begins, repeat the problem-definition, benchmark, comparative-
methodology, selected-family dossier, and selection-synthesis reading. The
custom design must be derived from the actual problem and accumulated evidence,
not from whichever attention component is newest or easiest to combine.

The first 4A.1 artifact is a **problem-to-model requirements matrix**. It must
translate the inputs into model-relevant requirements without redefining their
authority. At minimum it covers:

- one RGB image and a fixed 165-label recipe-level output;
- weak supervision and partial visual observability;
- long-tail support, sparse positives, and label co-occurrence;
- cuisine, dish, plating, and source shortcuts;
- small, mostly 3:2 landscape images and possible local ingredient evidence;
- independent probability/ranking outputs, calibration, and per-label analysis;
- one declared seed per configuration and no seed-stability claim;
- validation-only selection and complete test isolation;
- the 8 GB development-GPU boundary and limited thesis schedule; and
- reproducible checkpoint, transform, licence, dependency, and artifact access.

## Scope

Subphase 4A includes:

- primary-source discovery across food ingredient inference, food vision,
  weakly supervised multi-label recognition, long-tail multi-label learning,
  local or class-specific evidence, label dependence, and relevant visual
  pretraining;
- family-level candidate definition rather than variant counting;
- architecture, pretraining, head, adaptation, access, and resource analysis;
- current official implementation and checkpoint verification;
- a common deep-research dossier for every retained candidate;
- evidence-based selection of two established families with falsifiable local
  hypotheses; and
- custom attention-model research, three design proposals, size scaling rules,
  and one implementation handoff.

Source, official-artifact, and repository inspection are allowed to verify the
documented access and integration path. Subphase 4A does not add wrappers or run
models; load, forward/backward, peak-memory, and throughput checks belong to the
Macro-section 5 implementation gate and cannot be treated as research evidence
in this theoretical phase.

## Non-goals

Subphase 4A does not:

- choose `M_ref` or execute ingredient selection;
- change the benchmark, selected vocabulary, metrics, or test-isolation rules;
- rank models by copying headline scores from incomparable datasets;
- perform model training, HPO, accuracy tournaments, or test evaluation;
- count depth, width, checkpoint, or close version variants as distinct
  candidates;
- exhaustively survey every available architecture;
- assume that the newest or largest model is the best choice;
- combine backbone, head, loss, preprocessing, and label-dependency changes
  without a stated hypothesis;
- claim that the custom model is novel in the scientific literature unless a
  separate novelty search supports that claim; or
- create model deep dives before a selected architecture is actually integrated
  into the repository.

## Candidate identity and portfolio rules

A candidate is a **model family and experimental protocol**, not merely a model
name. Its record must make the following tuple explicit:

\[
C = (\text{family},\ \text{pretraining},\ \text{representation},
\text{adaptation},\ \text{multi-label head},\ \text{input policy}).
\]

Depth, width, patch size, checkpoint generation, and small/base/large variants
remain nested options of one candidate when the core architecture and hypothesis
are unchanged. For example, DenseNet depth variants form one DenseNet candidate;
they cannot fill multiple shortlist slots. A close successor such as a new
checkpoint or training recipe becomes a separate candidate only when it changes
the scientific mechanism being tested, not merely because it is newer.

The broad scan may describe more families, but its formal handoff contains only
three to five candidates. Those candidates must be scientifically distinct
enough that comparing them can answer different hypotheses. The two selected
established families should also be complementary; selecting two near-equivalent
families requires an explicit continuity or ablation rationale.

## Research evidence and recording system

### Durable owners

| Information | Durable owner |
| --- | --- |
| Problem, benchmark, scope, and binding comparison rules | Existing documents under [`project_objective/`](../project_objective/README.md) |
| 4A execution status, stage sequence, gates, and next action | This plan |
| Broad search, candidate landscape, problem-to-model matrix, and source catalog | A new date-stamped folder under [`research/discovery/`](../research/discovery/README.md) |
| Candidate deep-research dossiers and reusable comparative synthesis | A new indexed topic collection under [`research/topics/`](../research/topics/README.md) |
| Adopted two-family shortlist and, later, the selected custom design | [`experimental_model_portfolio.md`](../project_objective/experimental_model_portfolio.md), created at the 4A.3 decision checkpoint; custom topology remains pending |
| Detailed custom-model execution sequence | [`custom_attention_model.md`](custom_attention_model.md) |
| Attention-component and custom-design evidence | The indexed `research/topics/custom_attention_model_design/` collection created by 4A.4.1 under the [custom feature plan](custom_attention_model.md) |
| Temporary search notes, extracted tables, scripts, or unreviewed outputs | [`src_scratches/`](../../src_scratches/) or an appropriate experiment workspace, never as the final durable claim owner |
| Verified model contract after implementation | [`implementation_details/`](../implementation_details/README.md) and, where appropriate, [`models_deepdive/`](../models_deepdive/README.md) |

Do not copy the same conclusion into every artifact. Research records retain
evidence; the binding decision record states what the project adopts and links
that evidence; this plan records execution state and handoff; the general plan
records only project-level status.

### Common evidence schema

Every broad record, candidate dossier, and custom-design record must distinguish:

- **verified facts:** directly supported by the original paper, official code,
  official model card, licence, checkpoint metadata, or inspected repository;
- **external empirical evidence:** reported results with the source task,
  dataset, split, metric, and comparison context preserved;
- **project interpretation:** why the mechanism may or may not transfer to this
  weakly supervised ingredient task;
- **recommendation:** a proposed local use, adaptation, or exclusion;
- **assumption or uncertainty:** missing evidence, access, overlap, resource, or
  implementation questions; and
- **decision:** only after the responsible project-objective record adopts it.

Each substantive research document must include:

1. research question, boundary, cutoff date, and search method;
2. family/protocol identity and considered variants;
3. a claim-to-source table with direct primary links and evidence type;
4. the transfer boundary from the source task to this project;
5. the problem-fit matrix rows affected by each finding;
6. checkpoint, licence, official implementation, dependency, and provenance
   information where applicable;
7. facts separated from interpretations and open questions;
8. limitations and negative or contradictory evidence; and
9. a concise handoff section naming what the next stage may reuse and what it
   must not infer.

Secondary surveys, blog posts, benchmark aggregators, and search-result snippets
may locate evidence but cannot be the sole support for a candidate or design
claim. Prefer the original architecture/pretraining paper, official repository
or maintained library, official checkpoint/model card, and independent
peer-reviewed task evidence. For every candidate, seek at least:

- the original model or pretraining paper;
- one traceable official or maintained implementation/checkpoint path; and
- two independent task-relevant or mechanism-relevant empirical sources when
  available.

If that bundle does not exist, retain the gap explicitly and lower the evidence
confidence rather than substituting adjacent headline scores.

### Stage handoff packets

Each stage ends with one short handoff table. It carries only the information
needed by the next stage:

| Handoff | Required content |
| --- | --- |
| 4A.1 → 4A.2 | Candidate family ID, distinct hypothesis, qualifying sources, accessible implementation/checkpoint path, candidate variants, unresolved questions, and grouped exclusion reasons. |
| 4A.2 → 4A.3 | Common dossier fields, evidence confidence, project-fit findings, feasibility boundary, falsifiable experiment hypothesis, and explicit go/no-go uncertainties. |
| 4A.3 → 4A.4 | Two adopted families, rejected alternatives, reusable architectural components, complementary strengths, failure modes, and design gaps worth addressing in the custom model. |
| 4A.4 → Macro-section 5 | Selected custom topology, S/M/L scaling rules, preferred initial scale and fallback, tensor/interface contract, initialization/pretraining plan, implementation dependencies, required smoke tests, and unresolved engineering risks. |

An open-question register remains in the current research owner. A question is
closed only by linked evidence or a recorded project decision; it is not silently
dropped between stages.

## 4A.1 — Broad model discovery

### Research question

Which three to five established and accessible model families are the strongest
scientific candidates for recipe-level multi-label ingredient inference under
the frozen problem and resource constraints?

### Search boundary

The discovery must consider direct and adjacent evidence in three tagged tiers:

1. **Direct:** image-to-ingredient, inverse-cooking ingredient prediction, or
   closely matching food-image multi-label tasks.
2. **Adjacent:** food classification, recipe retrieval, food segmentation,
   weakly supervised localization, or other food tasks that test a transferable
   representation or attention mechanism.
3. **Mechanistic:** general multi-label classification, long-tail recognition,
   class-query attention, structured label prediction, or efficient visual
   pretraining that directly addresses a project requirement.

The scan is intentionally broad across supervised convolutional networks,
hierarchical or patch-based transformers, visual self-supervised models,
vision-language or food-domain pretraining, spatial/class-query models, and
label-dependency approaches. These are search strata, not mandatory shortlist
slots.

Recent work is preferred when it supplies a materially useful mechanism or
better supported implementation, but recency alone is not evidence of quality.
Every retained candidate must have sufficient research maturity, reproducible
access, and a plausible 8 GB path. Older classic families remain eligible when
they provide a strong, interpretable, or historically important control.

### Mandatory eligibility gates

A family enters the three-to-five handoff only if it has:

1. a distinct and falsifiable reason to help with at least one row of the
   problem-to-model matrix;
2. credible primary evidence from a direct, adjacent, or mechanistic task, with
   the transfer limit stated;
3. an official or maintained implementation and a traceable checkpoint or
   initialization path;
4. licence and access conditions compatible with reproducible thesis work;
5. a plausible fixed-vocabulary multi-label adaptation that does not silently
   add forbidden inference inputs;
6. a credible useful-resolution path under the 8 GB boundary; and
7. enough architectural distinctness that it does not merely pad the candidate
   count with a close variant.

### Outputs and stopping rule

Create a new dated discovery with, at minimum:

- `README.md` for method, cutoff, file map, synthesis, and limitations;
- `problem_model_requirements.md` for the input-derived requirement matrix;
- `candidate_landscape.md` for the broad scan, family grouping, and exclusions;
- `source_catalog.md` for primary and official evidence; and
- the 4A.1 → 4A.2 handoff table.

Stop broadening once three to five families pass every gate and the major
scientific strata have been considered. Do not continue collecting candidates
that answer the same hypothesis.

### 4A.1 completion checkpoint — 2026-08-28

**Status:** Done.

Evidence is recorded in the new
[2026-08-28 discovery](../research/discovery/2026-08-28/README.md):

- the mandatory problem and benchmark revisions are listed in the
  [requirements matrix](../research/discovery/2026-08-28/problem_model_requirements.md);
- the candidate landscape covers supervised CNN, hierarchical transformer,
  visual self-supervision, vision-language pretraining, and structured
  multi-label readout strata;
- five **new** family/protocol candidates pass the broad intake gates:
  EfficientNetV2, Swin V2, SigLIP2, a structured query/set head, and MaxViT,
  with hypotheses, evidence tiers, access paths, grouped variants, exclusions,
  and uncertainties;
- the already-used ResNet and DINOv2 models are recorded as baseline anchors,
  with their papers and implementation evidence retained but excluded from the
  new-candidate selection count;
- the primary-source and official-implementation links are catalogued in the
  [4A.1 source catalog](../research/discovery/2026-08-28/source_catalog.md); and
- the formal 4A.1 → 4A.2 handoff is complete without selecting an experiment
  family, freezing hyperparameters, running a model, or using test outcomes.

The handoff deliberately retains the structured query/set head as a protocol
candidate paired with a declared backbone. This keeps the head hypothesis
visible without pretending that it is a sixth independent backbone family. The
same handoff treats ResNet and DINOv2 as already-used anchors rather than
selection candidates, avoiding a false impression of five new alternatives.

**Newly discovered work for 4A.2:** verify one concrete representative and
checkpoint path per candidate; resolve licence, dependency, provenance/overlap,
native-aspect transform, and useful-resolution questions; and decide which
within-stratum alternatives (if any) deserve dossier-level treatment. A broad
lead may be demoted only with a recorded evidence, access, resource, or
interpretation reason.

**Next stage:** 4A.2 candidate deep research, using one normalized dossier
schema for C1–C5 and producing an evidence-comparable handoff to 4A.3.

## 4A.2 — Candidate deep research

### Research question

For each retained family, what does the literature and official implementation
actually establish, which project problems can it address, and what concrete
protocol could be implemented and fairly compared here?

### Dossier structure

Create one candidate file in a dedicated indexed research-topic collection for
each of the three to five families. Every dossier uses the same structure:

1. family identity, core architecture, original objective, and close variants;
2. tensor/feature flow and the mechanism that differs from other candidates;
3. pretraining data and supervision, including semantic or food-domain priors;
4. evidence on direct, adjacent, and mechanistic tasks, with contradictory or
   negative evidence retained;
5. fit to partial observability, local evidence, long tail, label co-occurrence,
   calibration, and shortcut risk;
6. multi-label head, adaptation, input, and initialization alternatives;
7. official implementation, checkpoint, licence, dependency, and offline
   reproducibility path;
8. plausible S/M/L or equivalent variants, while treating them as one family;
9. parameters, resolution, published compute metadata, and explicit distinction
   between known facts and unmeasured 8 GB behavior;
10. repository integration path and components that can be shared rather than
    copied;
11. risks, unresolved questions, and conditions that would invalidate the
    candidate; and
12. one falsifiable local benchmark hypothesis plus the simplest comparison
    that can test it in Macro-sections 6–7.

The dossiers are research artifacts, not implementation specifications. They
must not choose hyperparameters, report unrun local performance, or turn one
paper's score into a cross-family ranking.

### Completion gate

4A.2 is complete when every retained family has a reviewed dossier under the
same schema, all material claims link to primary or official evidence, open
questions are explicit, and the 4A.2 → 4A.3 handoff permits a like-for-like
qualitative comparison.

### 4A.2 completion checkpoint — 2026-08-28

**Status:** Done.

The indexed topic collection
[`experimental_model_candidates`](../research/topics/experimental_model_candidates/README.md)
contains one common-schema dossier for each retained new candidate:

- C1 EfficientNetV2;
- C2 Swin Transformer V2;
- C3 SigLIP2 image-only adapter, with text-conditioned variants explicitly
  separated;
- C4 structured query/set head, treated as a head protocol paired with a
  declared backbone; and
- C5 MaxViT.

The collection also records a qualitative comparison and the formal 4A.2 →
4A.3 handoff. Each dossier includes primary architecture evidence, an official
or maintained implementation path, the direct/adjacent/mechanistic transfer
boundary, project-fit judgments, resource metadata, open questions, and one
falsifiable local hypothesis. The direct food-ingredient comparison evidence
from the 2025 Recipe1M study is retained only in its source context and is not
used to rank candidates on Yummly.

No model was trained, tuned, measured for local accuracy, or evaluated on the
test split. Peak memory, exact aspect-preserving transforms, checkpoint hashes,
and dependency compatibility remain Phase 5 verification items.

**Next stage:** 4A.3 established-family selection. Use the comparative
[synthesis](../research/topics/experimental_model_candidates/comparative_synthesis.md)
to apply the hard gates and choose exactly two complementary families without
reopening an unbounded discovery search.

## 4A.3 — Established-family selection

### Selection question

Which two established families provide the strongest, complementary, and
feasible tests of the thesis hypotheses on the common benchmark?

### Decision method

Apply hard eligibility gates first. A finalist must retain:

- scientific relevance to the documented problem;
- sufficient evidence quality and a defensible transfer argument;
- traceable model/checkpoint/licence availability;
- a plausible implementation and 8 GB execution path;
- compatibility with the common image-only, fixed-vocabulary output contract;
- a falsifiable comparison against simpler baselines; and
- acceptable provenance, shortcut, and interpretation risks.

Compare the eligible families qualitatively across:

1. fit to the highest-priority problem requirements;
2. complementarity with the other selected family and required baselines;
3. strength and directness of empirical evidence;
4. expected information value of the local comparison;
5. implementation and training feasibility;
6. reproducibility and maintenance quality;
7. resource efficiency; and
8. useful architectural or methodological recency.

Do not collapse the comparison into a pseudo-precise numeric score. Use a common
table with `strong`, `moderate`, `weak`, or `uncertain` judgments, each linked to
the dossier evidence. Scientific fit and evidence quality take priority;
recency breaks a close decision only when access, feasibility, and
reproducibility remain at least equivalent.

Select exactly two established families. Record:

- why each family is selected;
- the distinct hypothesis it contributes;
- the preferred concrete variant/protocol and one feasible fallback;
- why the remaining candidates are not selected;
- which components or ideas should inform the custom model; and
- which uncertainties are deferred to implementation smoke tests rather than
  hidden.

The evidence remains in the research topic. The adopted shortlist belongs in
the binding project-objective decision record and must be linked from the
project-objective index, this plan, and the general plan. No test outcome or
local candidate accuracy run may influence the selection.

### 4A.3 completion checkpoint — 2026-09-07

**Status:** Done.

The [experimental portfolio](../project_objective/experimental_model_portfolio.md)
records 4A-D1, selecting EfficientNetV2-S and MaxViT-T as the two established
families. It includes the eligibility outcomes, a common qualitative matrix,
preferred checkpoint/adaptation/input/readout protocols, one within-family
fallback each, reasons for not selecting C2–C4, falsifiable comparisons, and
the handoff to 4A.4 and Phase 5. ResNet and DINOv2 remain existing anchors;
`M_ref` remains the independent 4B decision.

The bounded [source-review addendum](../research/topics/experimental_model_candidates/comparative_synthesis.md#bounded-source-review--2026-09-07)
clarifies the MaxViT pretrained input restriction, classifier boundary and
BatchNorm checkpoint note, EfficientNet's native transform difference, and
the local square-input interface. These checks were read-only; no models were
constructed, weights downloaded, or candidate training/test evaluation run.

**Newly specified implementation work:** implement and serialize the common
aspect-preserving square transform and pooled linear readout; pin exact weight
artifacts and normalization policy; measure full-training memory and throughput;
exercise a declared fallback only if needed; and preserve the distinction
between a model-protocol comparison and an isolated architecture claim.
Existing research recommendations remain historical evidence where 4A-D1
adopts a different concrete protocol.

**Next stage:** execute 4A.4.1 in the dedicated
[custom attention-model feature plan](custom_attention_model.md). Use the
portfolio's component/gap handoff to produce a bounded problem and evidence
synthesis before attention-component research. 4A as a whole remains In
progress.

## 4A.4 — Custom attention-model research

This stage is intentionally larger and must receive its own feature plan before
research begins. The child plan must preserve the evidence and handoff rules in
this document while defining its more detailed stages, tracker, source strategy,
and completion gates.

At minimum, the custom plan must cover:

### 1. Problem and evidence synthesis

Re-read the mandatory problem inputs, candidate dossiers, selection synthesis,
and current model interfaces. Extract:

- model requirements that existing candidates satisfy well;
- unresolved weaknesses or conflicting trade-offs;
- reusable components supported by research;
- components rejected by evidence or resource constraints; and
- a bounded design objective for the new model.

### 2. Intensive attention-network construction research

Study how research architectures construct attention-based visual multi-label
networks and which existing, validated components can be reused. The search must
cover, where relevant:

- patch, convolutional, or hybrid tokenization;
- local, windowed, global, channel, spatial, cross-, and class-query attention;
- hierarchical feature scales and feature-pyramid or multi-resolution fusion;
- positional information, normalization, residual paths, feed-forward blocks,
  pooling, and regularization;
- independent logits versus learned label queries and carefully bounded label
  dependence;
- efficient attention, parameter-efficient adaptation, activation memory, and
  small-image behavior;
- pretrained backbones or component initialization that can be accessed and
  audited; and
- known training-stability, calibration, shortcut, and interpretability limits.

The objective is not to collect components. Every component must have a stated
role, primary evidence, interface contract, and reason it is preferable to a
simpler alternative for this problem.

### 3. Component compatibility and architecture synthesis

Before proposing a full network, specify tensor shapes, feature resolutions,
data flow, initialization, trainability, and component interactions. Identify
which combinations are supported by prior work, which are project hypotheses,
and which introduce confounding changes. Estimate parameters and activation
pressure before implementation and reject combinations that lack a plausible
8 GB path.

### 4. Three model proposals and decision

Produce exactly three coherent topology-level architecture proposals. Each
proposal must include:

- one distinct scientific hypothesis and its expected failure mode;
- end-to-end tensor and component flow;
- attention locations and their intended evidence access;
- output head, label-dependency boundary, and probability interpretation;
- pretraining/initialization and input-transform contract;
- proposed ablations against the selected established families and simpler
  controls;
- implementation dependencies, licences, and reuse path;
- parameter, memory, and compute estimates; and
- `small`, `medium`, and `large` scale rules.

The three sizes of one proposal must preserve the same network topology and
scientific hypothesis. They may scale declared width, depth, embedding size,
number of heads, or repeated blocks, but they do not count as three model
proposals.

After review, select one topology for Macro-section 5. Record why it is preferred,
which evidence is inherited, what remains novel only at the project-design
level, and which initial scale should be implemented first. Phase 5 may move to
the declared fallback scale after measured resource smoke tests without
reopening the topology decision.

## Dependencies, assumptions, and open decisions

| Item | Status | Consequence |
| --- | --- | --- |
| Frozen FoodOn-first `v5` task and split | Available | Research targets one common 165-label benchmark. |
| Data Work package 2.4 runtime smoke completion | In progress | Does not block theoretical research; must pass before benchmark training. |
| Subphase 4B and Macro-section 3 | Independent/in progress | Do not block 4A research; the selected vocabulary is required later for the Phase 6 transferred ablation. |
| One declared seed per configuration | Binding | Candidate research must not promise seed-level comparisons or require repeated-seed promotion. |
| Current 8 GB development GPU | Binding | Literature and implementation paths must remain plausible within this boundary. |
| Exact HPO spaces and budgets | Deferred to Macro-section 6 | 4A defines model hypotheses and feasible protocols, not tuning values. |
| Established-family implementation handoff | Available in [4A-D1](../project_objective/experimental_model_portfolio.md) | Phase 5 still needs data readiness, artifact/interface checks, and measured resource validation. |
| Exact S/M/L parameter bands for custom proposals | Open until 4A.4 | The child plan must define them relative to selected baselines and measured Phase 5 feasibility. |

## Validation and completion criteria

Subphase 4A is complete only when:

- the mandatory input revisions and problem-to-model matrix are recorded;
- a new dated discovery retains the broad search, primary-source catalog,
  exclusions, and exactly three to five family-level candidates;
- every retained candidate has a complete, consistently structured deep dossier;
- exactly two established families are adopted through a binding decision, each
  with a falsifiable benchmark hypothesis and credible implementation/resource
  path;
- a dedicated custom-model plan is created and completed;
- the custom research records component evidence, interfaces, limitations, and
  three topology-level proposals with S/M/L scaling rules;
- one custom topology and an initial implementation scale are adopted;
- no test result, candidate HPO, or unrecorded source influenced either
  selection;
- Macro-section 5 receives implementation-ready handoffs for all three selected
  model categories; and
- this plan and the general plan are synchronized with the completed evidence
  and decisions.

## Decision and change log

| Date | Change | Rationale |
| --- | --- | --- |
| 2026-08-28 | Created the dedicated Subphase 4A plan with four research stages. | Separate primary experiment-model research from Subphase 4B, preserve an extensive source/evidence flow, select two established families from three to five candidates, and govern the larger custom attention-model research through its own feature plan. |
| 2026-08-28 | Completed 4A.1 and opened 4A.2. | The problem-to-model matrix and primary-source discovery now retain five new family/protocol candidates; ResNet and DINOv2 are explicitly preserved as already-used baseline anchors and excluded from the selection count. The next work is normalized dossier verification, not broader unbounded searching or training. |
| 2026-08-28 | Completed 4A.2 and opened 4A.3. | Five common-schema candidate dossiers and a qualitative handoff now cover architecture mechanisms, transfer limits, access/provenance, resources, and falsifiable hypotheses. The next decision is exactly two complementary established families; no local training or test outcome was used. |
| 2026-09-07 | Completed 4A.3; dedicated 4A.4 planning is next. | 4A-D1 adopts EfficientNetV2-S and MaxViT-T with explicit common-protocol choices, fallbacks and limitations; research evidence is retained, and custom topology, measured feasibility, and the independent selector remain undecided. |
| 2026-09-07 | Opened the dedicated 4A.4 feature plan. | The child plan has four explicit research subphases--problem/evidence synthesis, component research, compatibility synthesis, and three topology proposals--so custom design is not treated as one undifferentiated literature review. |
