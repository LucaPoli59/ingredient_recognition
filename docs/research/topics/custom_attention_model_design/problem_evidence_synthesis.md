# Problem and evidence synthesis for custom attention design

**Created:** 2026-09-08
**Last updated:** 2026-09-08
**Scope:** 4A.4.1 research brief; no topology or benchmark decision

## Question and method

What useful question remains for a custom attention model after selecting an
efficient convolutional family and a hybrid spatial-attention family?

The strongest bounded opportunity is **ingredient-specific aggregation of
spatial evidence while retaining dish context**. This is a project hypothesis,
not an observed defect in the selected models: neither has been evaluated on
the repaired benchmark. The brief was formed by re-reading the inputs below
at repository commit `e0cddbe`, inspecting the current model interface, and
separating historical dataset evidence from the current contract. The later
[component review](attention_component_evidence.md) tests whether the proposed
functions have credible implementations; it does not retroactively turn the
brief into an architectural decision.

## Inputs reviewed and authority reconciliation

Dates identify the visible last-updated revision on 2026-09-08.

| Input | Revision | What is inherited |
| --- | --- | --- |
| [Problem definition](../../../project_objective/problem_definition.md) | 2026-08-02 | Recipe-ingredient inference from one image; partial observability; no ingredient localization target. |
| [Yummly audit](../../../project_objective/yummly_data_audit.md) | 2026-08-02 | Historical evidence of small landscape images, ambiguous supervision, imbalance and cuisine/source shortcuts. |
| [Vocabulary audit](../../../project_objective/ingredient_vocabulary_audit.md) | 2026-08-04 | Semantic granularity and the distinction between label association, synonymy and visual distinguishability. |
| [Benchmark decisions](../../../project_objective/benchmark_decisions.md) | 2026-08-27 | `ingredients_target_v5`, 165 ordered outputs without `<UNK>`, exact-image-safe split, validation-only selection and calibration. |
| [Comparison methodology](../../../project_objective/model_comparison_methodology.md) | 2026-09-07 | Q1--Q4, one shared vocabulary per comparison, one declared seed per configuration, separately interpreted vocabulary controls. |
| [Requirements matrix](../../discovery/2026-08-28/problem_model_requirements.md) | 2026-08-28 | Stable R1--R11 requirement identifiers. |
| [Candidate collection](../experimental_model_candidates/README.md) and [comparative synthesis](../experimental_model_candidates/comparative_synthesis.md) | 2026-09-07 | Five research candidates, existing anchors, source limitations and the subsequent portfolio decision. |
| [EfficientNetV2](../experimental_model_candidates/efficientnet_v2.md) and [MaxViT](../experimental_model_candidates/maxvit.md) dossiers | 2026-09-07 | Convolution/SE and block/grid interaction mechanisms; implementation and transform boundaries. |
| [Swin V2](../experimental_model_candidates/swin_v2.md), [SigLIP2](../experimental_model_candidates/siglip2.md), [query/set-head](../experimental_model_candidates/structured_multilabel_head.md) dossiers | 2026-08-28 at intake | Alternative spatial/readout mechanisms and pretraining limitations. The query-head dossier receives a separately dated source correction in this change. |
| [4A-D1 portfolio](../../../project_objective/experimental_model_portfolio.md) | 2026-09-07 | EfficientNetV2-S and MaxViT-T; common pooled linear readout and 224-square aspect-preserving padded input; custom opportunity and Phase 5 gates. |
| [Current model contract](../../../implementation_details/models.md) | 2026-08-27 | Model-owned transforms, raw logits, reconstruction and visualization hooks. |
| [Parent plan](../../../plans/experimental_model_research.md), [child plan](../../../plans/custom_attention_model.md), [general plan](../../../general_plan.md) | 2026-09-07 at intake | Research sequence, independent 4B ownership and implementation/training boundaries. |

The older problem definition and audits contain 182-label legacy counts,
209-label candidate counts, provisional support rules and an unresolved
`<UNK>` discussion. These are historical evidence, not current design inputs.
The benchmark decision resolves the task to 165 `v5` labels without `<UNK>`;
the later comparison methodology resolves the seed policy. No new dataset
audit or validation/test inspection was performed here.

The portfolio supersedes the MaxViT dossier's old suggestion to retain its
stock pooled projection: the adopted established-family head is GAP plus a
new linear layer for both families. The previous SigLIP2 total-checkpoint
parameter display cannot establish the size or feasibility of its vision
tower. Neither historical candidate recommendation overrides 4A-D1.

Local inspection of [BaseModel](../../../../src/models/commons.py) confirms
square inputs, configurable transforms, serialization and visualization hooks.
The [DINOv2 wrapper](../../../../src/models/dinov2.py) currently returns the
adapted hub classifier's logits; a spatial-token consumer would require an
explicit new interface. A token count alone does not establish such integration.

## What the task permits and cannot resolve

The input is one prepared-dish RGB image. The output describes the recipe's
normalized ingredient set, including transformed or hidden ingredients. A
useful representation can therefore use local appearance and contextual cues;
forcing every positive label to occupy a visible region would impose false
supervision. There are no ingredient boxes or masks from which to train that
assumption.

Historical images have median dimensions 360 by 240 pixels and are mostly
3:2 landscapes. Those measurements describe the audited legacy collection,
not a newly measured `v5` distribution. Under the adopted fit-and-pad transform,
an exact 3:2 image occupies approximately 224 by 149 pixels. This geometry
calculation suggests limited local detail and substantial padding; it does not
show that a particular feature stride will fail.

High ingredient co-occurrence is not semantic equivalence. Likewise, strong
prediction of a dissolved ingredient may be useful contextual inference without
being direct recognition. The custom architecture cannot resolve annotation
ambiguity or identify all hidden ingredients by adding attention layers.

## Portfolio strengths and remaining questions

| Existing evidence or protocol | What it already supplies | Remaining question; project interpretation |
| --- | --- | --- |
| ResNet anchor and selected EfficientNetV2 | Local convolutional processing, nonlinear feature channels and a simple pooled classifier; efficient representation is already a portfolio question. | Does equal spatial averaging underuse some label-specific cues? More CNN depth alone would not isolate this. |
| Selected MaxViT | Convolution plus local block and global grid attention before pooling. | Does richer spatial interaction suffice, or is the final aggregation itself worth changing? Adding attention generically is already covered. |
| DINOv2 anchor | A visual self-supervised representation with a learned output adapter. | Better external pretraining is a different intervention from better local evidence aggregation. |
| Swin V2 reserve | Shifted-window interaction and hierarchical feature scales. | Useful component evidence, but reproducing another pooled spatial backbone adds limited contrast. |
| Query/set-head dossier | Generic multi-label and food-task precedents for adaptive readout. | Which mechanism is warranted for only 165 labels and small images? Food precedent must be checked for negative results. |
| SigLIP2 dossier | Pretraining, geometry and semantic-prior lessons. | Text-conditioned queries would add an external information source and require a different claim. |

Global average pooling does not make a network spatially ignorant: preceding
channels can already encode localized patterns and broad context. What GAP
does remove is the explicit arrangement of the final feature locations, using
the same averaging weights for every class. The unresolved hypothesis concerns
**adaptive aggregation**, not an assertion that the established models cannot
recognize small ingredients.

## Bounded design objective

**Primary objective O1:** investigate whether label-dependent use of local
image features together with dish-level context improves recipe-ingredient
ranking relative to a matched class-agnostic pooled representation.

The expected mechanism is that different ingredients can draw evidence from
different spatial features, while a contextual path remains available when the
target has no visible instance. The brief does not prescribe queries, a fusion
operator, an extra branch, a particular loss, feature strides or a topology.

Two secondary constraints shape this objective:

1. **Bounded computation:** preserve useful spatial information within a
   plausible 8 GB route and the single-seed thesis schedule. Complexity must
   earn its cost relative to a simple readout.
2. **Traceable information use:** expose raw per-label scores and identify
   the spatial/contextual paths, initialization and any dependency between
   labels. Diagnostic maps must remain distinguishable from visibility evidence.

These supplement, rather than replace, all inherited benchmark constraints.

## Requirement coverage and minimum functions

| Requirements | Consequence for the brief | Function required from component research |
| --- | --- | --- |
| R1, R7 | Ordered per-label ranking and later calibration; no class softmax or generated ingredient sequence. | A readout producing `(B, L)` raw logits for configurable `L`, initially 165. |
| R2, R5 | Local evidence can be small or absent; preserving context is necessary. | Spatial representation plus access to dish context, with explicit downsampling and padding implications. |
| R3 | Positives are sparse and support varies. | Keep per-label scoring and support analysis possible; do not claim a head fixes imbalance. |
| R4, R6 | Priors can dominate and external semantics change the experiment. | Explicit query/dependency provenance; no hidden text, graph or metadata source. |
| R8, R10 | Memory and study time are limited. | Compact operators, auditable reuse/initialization and a credible resource estimate. |
| R9, R11 | The custom model must answer an identifiable question. | A removable or replaceable proposed mechanism and a matched simpler comparison. |

## Failure conditions and minimal later comparison

The mechanism may fail because local cues disappeared during cooking or
resizing; because final features are too coarse; because class-specific
aggregation overfits background/cuisine cues; or because the pooled backbone
already encodes the available information sufficiently well. A capacity or
pretraining difference may also explain an apparent improvement.

The narrowest useful control is the **same visual trunk and initialization**
with class-agnostic pooling versus the proposed aggregation, holding data,
input, loss and adaptation policy fixed. Compare label-macro AP, paired micro
F1 under the common threshold policy, per-label/support slices and resource
cost. This is a prospective control for Phases 6--7 to budget, not an additional
training campaign authorized by this brief.

If it is not run, the comparison with EfficientNetV2 and MaxViT can establish
relative model-protocol performance but cannot isolate the custom aggregation
mechanism. A gain confined to common/contextual labels does not support the
narrower local-evidence interpretation. An inconclusive single-run result is
not proof of equivalence. Image shuffling and targeted perturbation can help
diagnose image dependence; distribution shift and correlation prevent them
from proving ingredient visibility. Observability slices require independent
annotations, not attention-derived groups.

## Questions handed to component research

| ID | Question | Evidence needed; subsequent owner |
| --- | --- | --- |
| Q1 | What is the least complicated way to make aggregation label-dependent? | GAP, residual spatial attention and query-head comparisons; 4A.4.2. |
| Q2 | Which spatial scales retain useful cues without unnecessary activation cost? | Tokenization/hierarchy/fusion evidence; component options in 4A.4.2, exact interfaces in 4A.4.3. |
| Q3 | How can a label use global context without forcing visible localization? | Readout and context-path mechanisms with absent-label limitations; 4A.4.2--4A.4.3. |
| Q4 | Do 165 labels warrant grouped queries or label-to-label interaction? | Original query-head ablations, query provenance and cost; 4A.4.2. |
| Q5 | Which initialization and normalization remain compatible with reuse and small physical batches? | Original/library evidence in 4A.4.2; exact trainability and weight mapping in 4A.4.3. |
| Q6 | How should padding and position information reach the readout? | Operator/masking behavior in 4A.4.2; geometry, masks and serialization in 4A.4.3 and Phase 5. |
| Q7 | What would distinguish useful attention from extra capacity or priors? | Counterevidence and simple control in 4A.4.2; proposal-specific claim in 4A.4.4. |

The [component handoff](attention_component_evidence.md#handoff-to-4a43)
records the resulting answers and unresolved parts of these questions. The
brief is complete as a research input. No component selection is binding and
no performance, feasibility or novelty claim has been established.
