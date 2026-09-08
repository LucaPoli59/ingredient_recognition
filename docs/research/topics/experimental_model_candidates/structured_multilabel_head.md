# C4 — Structured multi-label query/set head dossier

**Created:** 2026-08-28
**Last updated:** 2026-09-08
**Candidate:** C4 — Structured multi-label readout protocol
**Status:** Research dossier; no local training performed

**Subsequent source review:** The [4A.4.2 component record](../custom_attention_model_design/attention_component_evidence.md)
qualifies the original favorable intake below: direct food evidence includes
negative decoder outcomes, and standard ML-Decoder uses fixed random queries.
The original recommendation to audit it first is not a binding preference for
the custom model.

## Research question and boundary

Can a class-query or set-decoding head extract label-specific spatial evidence
and use ingredient-set structure more effectively than global average pooling
with independent logits? C4 is a head/protocol candidate, not a sixth
backbone family. It must be paired with one declared visual backbone and
compared against the same backbone with a simple independent head.

The canonical C4 claim uses query embeddings and image features only; query
trainability is representative-specific.
Ingredient names, recipe text, external ontologies, autoregressive generation,
and graph statistics derived from validation/test labels are not implicit parts
of the protocol.

## Candidate identity and mechanism

| Tuple field | Proposed C4 value |
| --- | --- |
| Family/protocol | Structured query/set multi-label head |
| Representatives | One of ML-Decoder or Query2Label after implementation/access inspection |
| Backbone | Declared in 4A.3; the same backbone must receive an independent GAP-head control |
| Representation | A spatial feature map or patch-token sequence from the backbone |
| Readout mechanism | Class/group queries cross-attend to visual features; Query2Label learns queries, while standard ML-Decoder fixes random queries and can group classes |
| Adaptation | Train the head from scratch; backbone frozen/partial/full tuning is a separate axis |
| Output | One logit per of the 165 labels, interpreted independently after sigmoid for metrics |

Query2Label uses a Transformer decoder in which class-specific label embeddings
query a visual feature map through cross-attention. ML-Decoder removes redundant
self-attention and introduces group decoding so that attention-based heads scale
to thousands of classes. Both mechanisms can preserve class-specific spatial
readout while returning a fixed multi-label vector; neither guarantees that a
recipe-only label is visually present.

## Feature flow and adaptation boundary

Let a backbone expose `F ∈ R^(B×N×C)` (flattened spatial tokens) and let
learned queries be `Q ∈ R^(L×D)`, where `L=165`. A query decoder projects
queries and features to a common dimension, performs cross-attention, and maps
each class query to one logit. The exact tensor order, number of decoder
layers, query grouping, and pooling are representative-specific implementation
choices and are not frozen by this research dossier.

The output remains a set of independent logits for compatibility with
`BCEWithLogitsLoss`, mAP, calibration, and per-label trajectories. Query-to-query
self-attention or graph dependencies may be useful, but they increase the risk
that the head learns recipe co-occurrence instead of image evidence. Such
dependencies must be paired with image-shuffle and non-visual baselines.

## Pretraining and task-relevant evidence

| Source | Evidence type | Finding used here | Transfer boundary |
| --- | --- | --- | --- |
| [Query2Label, Liu et al.](https://arxiv.org/abs/2107.10834) | Primary multi-label mechanism | Class queries and decoder cross-attention adaptively probe class-related regions and outperform prior methods on several generic multi-label datasets. | Generic datasets and label embeddings do not establish ingredient observability or Yummly performance. |
| [Official Query2Label repository](https://github.com/SlongLiu/query2labels) | Official implementation path | Provides a concrete reference implementation and MIT licence for inspection. | The code's dependency age, tensor interface, and memory behaviour must be audited before reuse. |
| [ML-Decoder, Ridnik et al., WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Ridnik_ML-Decoder_Scalable_and_Versatile_Classification_Head_WACV_2023_paper.html) | Primary efficient-head evidence | Query-based classification and group decoding improve spatial use and scale to thousands of labels; the paper reports generic multi-label and single-label results. | Reported MS-COCO/ImageNet scores are not food evidence and group counts are not chosen here. |
| [Official ML-Decoder repository](https://github.com/Alibaba-MIIL/ML_Decoder) | Official implementation/licence path | Public MIT-licensed code and examples provide a maintained-enough reference for an adapter audit. | Repository/API revision and integration with the current Lightning contract remain open. |
| [Food Ingredients Recognition through Multi-label Learning](https://arxiv.org/abs/2210.14147) | Direct food multi-label evidence, including negative results | A Nutrition5K study compares global-pooling and ML-Decoder; the [subsequent full-text review](../custom_attention_model_design/attention_component_evidence.md#task-relevant-empirical-evidence) qualifies the original positive interpretation. | Different dataset, resolution, metric aggregation and training; precedent for testing the question, not evidence of a reliable decoder advantage. |
| [Inverse Cooking, Salvador et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Salvador_Inverse_Cooking_Recipe_Generation_From_Food_Images_CVPR_2019_paper.pdf) | Direct food set-structure evidence | Ingredient prediction is treated as an unordered set within a food-image/recipe system. | The system is generative and uses recipe targets; it is evidence for set structure, not a drop-in classifier. |

## Project fit and transfer limits

| Requirement | Assessment | Reason and limitation |
| --- | --- | --- |
| R1 fixed 165-label output | Strong | Query heads naturally emit one score per declared class. |
| R2 partial observability/local evidence | Strong hypothesis, moderate evidence | Cross-attention can seek class-specific regions, but hidden/dissolved ingredients remain weakly supervised. |
| R3 sparse positives/long tail | Moderate-high | Class queries isolate label gradients; decoder complexity is bounded for 165 labels, but imbalance still needs the common loss policy. |
| R4 label co-occurrence/shortcuts | Uncertain | Query self-attention or label graphs may exploit co-occurrence; only train-only structures and explicit non-visual controls are admissible. |
| R5 small 3:2 inputs | Moderate | Spatial features are retained, but token resolution and feature-map extraction depend on the paired backbone. |
| R6 provenance/leakage | Strong for randomly learned queries | No external text/ingredient prior is required; external backbone provenance still applies. |
| R7 ranking/calibration | Strong | The head can expose raw per-class logits; calibration remains a validation-only operation. |
| R8 reproducibility/8 GB | Moderate | 165 labels are small, but cross-attention memory scales with token count and implementation details. |
| R9 fair comparison | Strong if paired | Same backbone, data, and transforms permit a clean head ablation. |
| R10 one declared seed | Strong | The protocol does not require repeated-seed evidence at intake. |
| R11 falsifiability | Strong | A paired GAP head and non-visual control directly test whether query attention adds image evidence. |

## Canonical protocol and alternatives

The recommended C4 protocol is:

1. choose one representative head after the 4A.3 feasibility audit;
2. pair it with a declared backbone and expose a spatial feature map;
3. initialize queries without ingredient-name embeddings and declare whether
   they are learned or fixed under the chosen representative;
4. emit 165 logits with no autoregressive decoding; and
5. compare against the same backbone plus a simple pooling/linear head.

ML-Decoder is the preferred first implementation candidate because its group
decoding is explicitly designed for scalable multi-label heads; Query2Label is
the reference if class-specific cross-attention visibility is more important
than implementation simplicity. This is a recommendation, not an adopted
decision. Graph-only, word-query, autoregressive, or recipe-text variants are
separate interventions.

## Access, licence, dependency, and provenance

Both official repositories expose MIT licences ([ML-Decoder](https://raw.githubusercontent.com/Alibaba-MIIL/ML_Decoder/main/LICENSE),
[Query2Label](https://raw.githubusercontent.com/SlongLiu/query2labels/main/LICENSE)).
Neither repository's original environment should be copied wholesale into the
project. Phase 5 must pin a compatible implementation or reimplement the
smallest well-understood head under the repository's existing PyTorch/Lightning
stack, while preserving attribution and the documented algorithm.

Unlike C3, learned C4 queries do not require an external text checkpoint. The
paired backbone still determines image-pretraining provenance, and any graph
or label embedding source would reintroduce an additional prior that must be
audited separately.

## Resource envelope

For a feature map with `N` tokens and query dimension `D`, a straightforward
cross-attention layer has memory and compute that grow with `L×N×D` (here
`L=165`). ML-Decoder's grouped queries reduce the effective decoder burden for
large label sets. Exact memory depends on the paired backbone feature
resolution, number of layers, mixed precision, and implementation; no 8 GB
claim is made here.

The head has no required external checkpoint. Query and projection weights are
random-initialized in the canonical protocol; standard ML-Decoder freezes the
query embeddings while learning projections. A larger decoder, label-text
initialisation, or graph prior would change both resource and scientific
interpretation.

## Repository integration path

Phase 5 should add a reusable head module under `src/models/` or a clearly
owned subpackage, with an explicit feature-map interface and a `num_classes`
argument. Backbone wrappers must expose the tensor before global pooling. The
head must implement `forward(features) -> logits` and integrate with the
existing Lightning loss/metric code without embedding sigmoid or thresholding.

Required smoke checks are feature-map shape contracts for every paired
backbone, output shape `(B, 165)`, deterministic checkpoint reconstruction,
Grad-CAM/attention-target traversal where supported, one forward/backward pass,
and peak-memory measurement at the declared token resolution. The paired GAP
control must use identical data and training instrumentation.

## Risks, go/no-go conditions, and open questions

**Go for 4A.3 comparison:** C4 has direct food attention-decoder precedent,
strong generic multi-label evidence, MIT reference code, and a falsifiable
same-backbone head ablation.

**Go/no-go questions:**

- Which representative (ML-Decoder or Query2Label) can be integrated without
  importing obsolete dependencies or hidden text/label inputs?
- Does the chosen backbone expose a sufficiently fine spatial feature map at
  the fair input resolution?
- Do query gains survive image-shuffle, frequency, cuisine-prior, and
  support-matched controls?
- Does grouped decoding alter logits or calibration enough to require a
  separately documented output policy?

**Invalidating evidence:** if the head cannot be paired with a maintained
backbone under the common feature-map contract, or if gains are fully explained
by non-visual co-occurrence, C4 should be demoted as an experimental family
while its mechanism remains useful for the custom-model research.

## Falsifiable local hypothesis and minimal comparison

**Hypothesis H-C4:** on the same backbone, a learned query/set head will
improve validation AP for labels with spatially local cues over global average
pooling, while the improvement will shrink on image-shuffled and
co-occurrence-only controls.

The minimal test is one declared-seed paired comparison: `backbone + GAP head`
versus `backbone + C4 head`, using identical split, transforms, loss, budget,
and output metrics. Report macro/micro AP, per-label support and observability
slices, calibration, and resource cost. Do not interpret a higher score alone
as proof of visual localisation.

## Comparison anchors

| Anchor | Role |
| --- | --- |
| [ResNet local contract](../../../implementation_details/models.md) | Simple pooled-head control and possible paired backbone. |
| [DINOv2 local deep dive](../../../models_deepdive/dinov2.md) | Existing token-based visual anchor; its patch output may be a useful but separately declared pairing. |

## Handoff to 4A.3

Reuse the C4 head/backbone separation, representative options, control design,
and H-C4 hypothesis. 4A.3 must decide whether a head protocol is sufficiently
distinct and feasible to occupy one of the two established-family slots; it
must not count C4 as a second backbone family or hide the paired-backbone
choice.

## References

- [Query2Label paper](https://arxiv.org/abs/2107.10834)
- [Query2Label repository](https://github.com/SlongLiu/query2labels)
- [ML-Decoder paper](https://openaccess.thecvf.com/content/WACV2023/html/Ridnik_ML-Decoder_Scalable_and_Versatile_Classification_Head_WACV_2023_paper.html)
- [ML-Decoder repository](https://github.com/Alibaba-MIIL/ML_Decoder)
- [Food Ingredients Recognition through Multi-label Learning](https://arxiv.org/abs/2210.14147)
- [Inverse Cooking paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Salvador_Inverse_Cooking_Recipe_Generation_From_Food_Images_CVPR_2019_paper.pdf)
