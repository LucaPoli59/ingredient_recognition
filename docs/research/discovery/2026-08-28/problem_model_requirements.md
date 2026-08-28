# Problem-to-model requirements matrix

**Created:** 2026-08-28
**Last updated:** 2026-08-28
**Owner:** Subphase 4A broad model discovery
**Status:** 4A.1 baseline for candidate intake; not a model decision

## Purpose and authority

This matrix translates the current project objective and benchmark contract into
requirements that a candidate model protocol must satisfy or explicitly test.
It does not change any binding decision. The authority remains the linked
documents under [`project_objective/`](../../../project_objective/README.md),
while this file is the 4A.1 handoff artifact used to filter model families.

## Inputs reviewed

The following documents were read before the search. Dates are the revisions
visible in the repository on 2026-08-28.

| Input | Revision | Requirements extracted |
| --- | --- | --- |
| [`yummly_data_audit.md`](../../../project_objective/yummly_data_audit.md) | 2026-08-02 | Yummly-only provenance, small/non-square images, sparse labels, duplicate and cuisine shortcuts, and recipe-level supervision ambiguity. |
| [`problem_definition.md`](../../../project_objective/problem_definition.md) | 2026-08-02 | One RGB dish image to a normalized ingredient set; weak supervision and partial observability; no detection, segmentation, generation, or open-vocabulary target. |
| [`benchmark_decisions.md`](../../../project_objective/benchmark_decisions.md) | 2026-08-27 | Frozen `ingredients_target_v5`, 165 train-supported labels, exact-image-aware split, validation-only selection, mAP and micro-F1 contract, and test isolation. |
| [`model_comparison_methodology.md`](../../../project_objective/model_comparison_methodology.md) | 2026-08-28 | Common full/selected vocabularies, independent model-category comparison, random-reduction controls, and one declared seed per configuration. |
| [`ingredient_vocabulary_audit.md`](../../../project_objective/ingredient_vocabulary_audit.md) | 2026-08-04 | Target granularity, support and semantic ambiguity, and the separation of taxonomy repair from later learnability selection. |
| [`models.md`](../../../implementation_details/models.md) | 2026-08-27 | Existing model contract: a model returns one logit per class and the training module owns sigmoid/loss; current ResNet, DenseNet, and DINOv2 paths. |
| [`2026-08-02 discovery`](../2026-08-02/README.md) | 2026-08-28 | Broad evidence on food inference, multi-label heads, pretraining, losses, calibration, shortcuts, and 8 GB feasibility. |
| [`2026-08-22 discovery`](../2026-08-22/README.md) | 2026-08-27 | Selector-oriented family/source/checkpoint evidence reused only where its transfer boundary matches 4A; selector dispositions are not copied. |
| [`experimental_model_research.md`](../../../plans/experimental_model_research.md) | 2026-08-28 | Family-level candidate tuple, eligibility gates, stage handoffs, and the 4A.1 stopping rule. |

## Baseline boundary

The repository already contains and has used a ResNet protocol and a DINOv2
protocol. They remain mandatory local comparison anchors, and their papers,
checkpoints, wrappers, and historical evidence remain in the research record.
They are not counted as new candidates in the 4A.1 selection set; the matrix is
used to assess whether a new family adds a distinct hypothesis beyond those
anchors.

## Requirement matrix

| ID | Project requirement | Model/protocol implication | Evidence required before 4A.2 can recommend it | Main risk or non-goal |
| --- | --- | --- | --- | --- |
| R1 | **Fixed closed-vocabulary output:** one RGB image produces 165 recipe-level ingredient labels. | Adapt the candidate to a fixed multi-label sigmoid head with one logit per `ingredients_target_v5` label. | Shape and output semantics are documented; no text, recipe, or metadata is required at inference. | Open-vocabulary name scoring or recipe generation would answer a different question. |
| R2 | **Weak supervision and partial visual observability:** labels can be hidden, dissolved, transformed, or absent from pixels. | Preserve global context while allowing local/class-specific evidence; do not assume that localization equals label truth. | Source task contains a comparable ambiguity or the transfer argument states what is not observable. | Pixel-only segmentation or object-detection success cannot be treated as ingredient learnability. |
| R3 | **Sparse positives and long-tail support.** | Support independent per-label ranking/probabilities, robust loss variants, and per-label AP analysis; head cost must remain bounded. | Evidence discusses imbalanced multi-label behavior or provides a credible adaptation path. | A headline micro score can hide failure on rare ingredients. |
| R4 | **Correlated ingredients and cuisine/dish shortcuts.** | Label-dependency or query heads are admissible only as explicit hypotheses; any graph/statistics must be train-only and paired with non-visual controls. | Candidate mechanism and dependency source are named separately from visual evidence. | Co-occurrence success must not be reported as proof of visual recognition. |
| R5 | **Small, mostly 3:2 landscape inputs and possible local cues.** | Prefer aspect-preserving preprocessing and a useful-resolution path; hierarchical or spatial features are valuable if memory-feasible. | Official transforms/checkpoint assumptions and a plausible 8 GB configuration are recorded. | Silent square warping, aggressive crops, or high-resolution claims may remove the only cue or exceed memory. |
| R6 | **Shortcut and leakage control.** | Pretraining/data provenance, source overlap, and external label semantics must be auditable; all local comparisons use the frozen split. | Training corpus, licence, checkpoint provenance, and any text prior are explicit. | Food/recipe pretraining can leak near-duplicate images or target vocabulary knowledge. |
| R7 | **Ranking, calibration, and threshold diagnostics.** | Return raw logits and probabilities; support mAP macro/micro views, fixed-policy F1, calibration, and per-label trajectories. | The head exposes independent scores and does not require an unavailable decoder at evaluation time. | A single tuned threshold or a closed-set accuracy can obscure ranking and calibration. |
| R8 | **Reproducible, compute-bounded implementation.** | Use maintained code/checkpoints or a clear initialization path; record licence, dependency, transform, parameter, and resource assumptions. | Official/maintained access is traceable and an 8 GB path is plausible; unmeasured behavior is labelled unverified. | “Fits” cannot be claimed from paper-scale hardware; custom dependency drift can block reproduction. |
| R9 | **Fair category comparison.** | Keep benchmark, target projection, split, metrics, and selection boundary common; vary the model hypothesis rather than several uncontrolled axes. | Candidate tuple explicitly names backbone, pretraining, representation, adaptation, head, and input policy. | Combining a new backbone, loss, crop policy, and dependency graph would be an uninterpretable comparison. |
| R10 | **Single-run thesis budget.** | Plan for one declared seed per configuration; report temporal/configuration sensitivity and finite-validation uncertainty where feasible, without a seed-stability claim. | Research protocol does not require repeated-seed evidence to justify intake. | Published multi-seed robustness cannot be assumed locally. |
| R11 | **Scientific falsifiability and diagnostic value.** | Each retained family must state one mechanism-specific local hypothesis and a minimal later comparison against simpler controls. | The hypothesis identifies the affected requirement rows and a failure condition. | A model is not retained merely because it is newer, larger, or has a higher unrelated benchmark score. |

## Interpretation boundary

The matrix deliberately mixes hard compatibility requirements (R1, R5, R6,
R8, R9) with research opportunities (R2–R4, R7, R11). A candidate may be
retained when it addresses an opportunity through a testable hypothesis, but it
cannot waive a hard requirement. “Accessible” means that an official or
maintained implementation/checkpoint or a reproducible initialization path can
be named; it does not mean that local loading or peak memory has already been
measured. Those checks belong to Macro-section 5.

The matrix also keeps three kinds of evidence separate:

- **visual transfer:** what the image representation can encode;
- **label-side structure:** what dependencies or text priors can add; and
- **measurement instrumentation:** whether the protocol exposes comparable
  per-label scores and trajectories.

The later selector decision in Subphase 4B may reuse this matrix, but it must
apply its own measurement criteria and must not inherit the 4A shortlist.

## 4A.2 questions opened by the matrix

1. Which retained protocols can expose spatial evidence without making
   localization a hidden target?
2. For self-supervised and vision-language encoders, which adaptation mode is
   the fairest image-only comparison under the frozen vocabulary?
3. Does a class-query/set head improve rare-label ranking after non-visual and
   co-occurrence controls, or merely exploit label structure?
4. Which small/base variants preserve useful resolution under the 8 GB boundary
   once the repository's actual transforms and head are fixed?
5. What provenance and overlap checks are needed before a food-domain or
   language-supervised checkpoint can enter the common benchmark?
