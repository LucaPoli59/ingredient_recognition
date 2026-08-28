# 4A.1 candidate landscape

**Created:** 2026-08-28
**Last updated:** 2026-08-28
**Scope:** Broad model discovery for the experimental-model stream (4A)
**Cutoff:** 2026-08-28

## Research question and method

The question is not “which paper has the highest score?” It is:

> Which model families and protocols provide distinct, falsifiable hypotheses
> for recipe-level multi-label ingredient inference on the frozen Yummly
> benchmark, while remaining accessible and plausibly reproducible on an 8 GB
> development GPU?

The scan started from the [problem-to-model requirements
matrix](problem_model_requirements.md), the [2026-08-02 broad
discovery](../2026-08-02/README.md), and the [2026-08-22 selector
discovery](../2026-08-22/README.md). It then checked primary papers,
official project pages, maintained libraries, model cards, and checkpoint paths.
Evidence is tagged as:

- **D (direct):** image-to-ingredient or closely matching food multi-label
  evidence;
- **A (adjacent):** food classification, segmentation, retrieval, or other
  food-vision transfer evidence; and
- **M (mechanistic):** general multi-label, self-supervised, vision-language,
  or label-structure evidence that addresses a project requirement.

Reported scores are retained only in their original task context. They do not
rank candidates across datasets, vocabularies, splits, or metrics.

## Existing baselines versus new selection candidates

Two models already used in this repository are retained as **baseline anchors**,
not as candidates competing for the new-family selection:

| Baseline anchor | Role in 4A | Evidence retained |
| --- | --- | --- |
| **B1 — ResNet** | Historical supervised-CNN continuity and mandatory simple control. It is not a new candidate because the repository already uses it. | Original residual-learning paper, current model contract, and legacy experiment evidence. |
| **B2 — DINOv2** | Existing visual self-supervised representation and transfer control. It is not a new candidate because it is already implemented and used. | Official DINOv2 paper/repository/model card and current repository deep dive. |

Their papers and implementation evidence remain in the source catalog so that
new models can be compared against a known local anchor. “Not a candidate” does
not mean “remove from experiments”; it means that neither model fills one of the
new-family selection slots.

## Candidate identity and grouping

For 4A.1 a candidate is a **new family-level protocol**, represented as

`(backbone family, pretraining, representation, adaptation, head, input policy)`.

Depth/width/checkpoint variants remain nested choices. A candidate may be a
backbone family (C1, C2, C3, C5) or a multi-label solution/head family (C4), but
the tested mechanism must be explicit. C4 is paired with a declared backbone and
does not count as a sixth backbone. Existing B1/B2 are excluded from the new
candidate count even when a new candidate reuses their head or comparison
protocol.

## Formal retained set: five new candidates

The following five candidates pass the broad intake gates. “Pass” means that an
official or maintained access path and a plausible fixed-vocabulary adaptation
can be named; it does **not** claim that loading, peak memory, or local accuracy
has already been measured. Those are Macro-section 5 checks.

| ID | New family/protocol retained for 4A.2 | Distinct hypothesis | Evidence tier | Access and 8 GB plausibility | Initial confidence |
| --- | --- | --- | --- | --- | --- |
| C1 | **Efficient convolutional network** — EfficientNetV2-S (with smaller/larger variants nested), independent sigmoid head | Fused-MBConv and training-aware scaling may deliver a stronger parameter/compute trade-off than the existing ResNet control, which is valuable for sparse multi-label learning under the GPU budget. | D + M | Official PMLR paper and current TorchVision constructors/weights; S is a plausible starting point, but aspect/transforms and local memory remain to verify. | High |
| C2 | **Hierarchical/windowed transformer** — Swin Transformer V2-T/S, independent sigmoid head | Windowed multi-scale features may preserve local ingredient evidence and long-range dish context better than a pooled CNN at useful resolution. | A + M | Official Microsoft repository plus maintained library paths; Tiny/Small are the only plausible starting sizes until measured. | Medium-high |
| C3 | **Vision-language visual encoder** — SigLIP2 Base (CLIP/SigLIP as fallback continuity), image-only downstream sigmoid head | Language-supervised visual features may encode fine-grained food semantics that improve weak-label transfer, but the semantic prior and provenance must be reported separately. | A + M | Official/maintained SigLIP2 checkpoint path; dependency, checkpoint licence, and memory are conditional. Downstream label-text scoring is out of scope for the canonical protocol. | Medium |
| C4 | **Structured multi-label query/set head** — one ML-Decoder or Query2Label representative over a declared backbone | Label-specific cross-attention may recover local evidence and model a set of ingredients more effectively than pooled independent logits; a paired non-visual control is required to detect co-occurrence shortcuts. | D + M | Official papers and public implementations; head can be trained from a local backbone without a special external checkpoint. Head memory and representative choice remain open. | Medium-high |
| C5 | **Hybrid multi-axis attention network** — MaxViT-T (or a compact maintained equivalent) with an independent sigmoid head | Alternating local block attention and global grid attention may combine fine ingredient fragments with whole-dish context more efficiently than a purely windowed transformer. | A + M | Official Google repository and TorchVision `maxvit_t` weights; compact 31M-parameter path is plausible, but official repo maintenance and square 224-pixel preprocessing require verification. | Medium |

These are intentionally not a final ranking. C1 tests efficient convolutional
scaling, C2 and C5 test different spatial-attention backbones, C3 tests a
language-supervised representation, and C4 tests the multi-label readout. B1
ResNet and B2 DINOv2 remain mandatory comparison anchors but do not occupy these
five new-family slots.

## Candidate records

### C1 — EfficientNetV2 efficient convolutional network

**Protocol for deep research:** EfficientNetV2-S (with a smaller fallback) and
an independent 165-logit sigmoid head. Initialization, adaptation, and input
policy are explicit variables. The direct Nutrition5K ingredient study
evaluated EfficientNet-family encoders, providing a food-task precedent even
though it does not establish performance on the repaired Yummly target.

**Why it is retained:** EfficientNetV2 was designed through training-aware
architecture search and scaling with Fused-MBConv and progressive learning. It
is a genuinely new candidate relative to the existing ResNet control because
the efficiency/scaling mechanism, not only depth, is the hypothesis.

**Requirement rows:** R1, R3, R5, R7, R8, R9, R11.

**4A.2 questions:** whether the training recipe can be adapted without importing
uncontrolled augmentation; which resolution and aspect-preserving transform are
fair; and whether its efficiency matters when the bottleneck is sparse-label
optimization rather than throughput.

### C2 — Hierarchical/windowed transformer

**Protocol for deep research:** Swin V2 Tiny or Small with an independent
165-logit head. Patch/window size, input resolution, and adaptation mode must be
recorded as nested variables rather than new candidates.

**Why it is retained:** hierarchical windows expose multi-scale features without
requiring full global attention at every pixel. That is a plausible response to
small local cues plus dish-level context, but food-ingredient transfer evidence
is indirect and must not be overstated.

**Requirement rows:** R2, R3, R5, R8, R9, R11.

**4A.2 questions:** whether the available input resolution creates useful
patches; whether the memory cost is acceptable; and whether any improvement
survives matched preprocessing and non-visual controls.

### C3 — Vision-language visual encoder

**Protocol for deep research:** SigLIP2 Base (or a smaller maintained variant)
feeding an ordinary image-only sigmoid head. Prompted image-to-ingredient text
scoring, text-initialized label queries, and prompt ensembles are separate
semantic-prior interventions and cannot be merged with the canonical result.

**Why it is retained:** SigLIP/SigLIP2 offer a current contrastive image-text
representation with released checkpoints and a stated path to dense/native-
aspect features. Such pretraining may help fine-grained food semantics, but it
also imports web-language concepts and possible source overlap, so the transfer
claim is intentionally weaker than “visual recognition from Yummly alone.”

**Requirement rows:** R1, R2, R5, R6, R7, R8, R9, R11.

**4A.2 questions:** checkpoint licence and exact provenance; an 8 GB-compatible
adaptation mode; whether native-aspect features can be used without changing the
benchmark; and how to report the semantic prior without giving it downstream
label text.

### C4 — Structured multi-label query/set head

**Protocol for deep research:** choose one representative (ML-Decoder or
Query2Label) after checking the current implementation and memory path, then
pair it with a declared backbone. Learned class queries are the canonical
protocol; word-initialized queries, graph-only dependencies, and autoregressive
recipe generation are separate variants or exclusions.

**Why it is retained:** Query2Label explicitly uses label queries and
cross-attention to pool class-related spatial features; ML-Decoder provides an
efficient query-based multi-label head. Inverse Cooking supplies a food-domain
set-decoding precedent, but its recipe-generation objective is outside the
benchmark. This candidate therefore tests a readout hypothesis rather than
claiming that the head alone makes ingredients visually observable.

**Requirement rows:** R2, R3, R4, R7, R9, R11.

**4A.2 questions:** representative implementation, query count and complexity,
whether the head exposes comparable logits, and whether gains remain after
training-only co-occurrence and non-visual controls.

### C5 — MaxViT hybrid multi-axis attention network

**Protocol for deep research:** MaxViT-T at the maintained TorchVision path (or
the official TensorFlow checkpoint only if a fair PyTorch route is unavailable)
with an independent 165-logit head. Local block attention and global grid
attention remain the mechanism under test; the model is not merged with the C2
Swin family.

**Why it is retained:** MaxViT combines convolutional stages with block and grid
attention, exposing local and global interactions at different resolutions. The
official project documents compact checkpoints and TorchVision exposes a
30.9M-parameter `maxvit_t` weight, making it a useful hybrid candidate despite
the need to audit the archived research repository and its square-input
preprocessing.

**Requirement rows:** R2, R3, R5, R8, R9, R11.

**4A.2 questions:** whether the pretrained positional/input assumptions can
respect the project’s 3:2 images; which implementation and licence are durable;
and whether the hybrid attention offers information beyond Swin at comparable
resolution and memory.

## Broad leads not in the formal five

These leads were considered and remain reusable evidence, but they do not pad
the 4A.1 handoff with a duplicate hypothesis or a task-mismatched system.

| Lead | Disposition and re-entry condition |
| --- | --- |
| ResNet and DINOv2 | Existing baseline anchors B1/B2. Preserve their papers, wrappers, and local evidence, but do not count them as new selection candidates. |
| ConvNeXt V2 | Strong modern CNN/FCMAE lead, but its official repository is archived and its pretrained models carry separate non-commercial terms. Re-enter only if those terms and a maintained adapter are acceptable, or substitute it for C1 with an explicit mechanism rationale. |
| DenseNet, EfficientNet (pre-V2), MobileNet, Inception | Existing/adjacent supervised CNN alternatives. The Nutrition5K study is retained as evidence, but depth/width or legacy CNN changes do not create extra candidate slots. |
| MAE, I-JEPA, DINOv3 | Self-supervised alternatives. DINOv2 is already the local anchor; re-enter only with a compact, accessible checkpoint and a mechanism that adds information rather than a release-date variant. |
| CLIP and original SigLIP | Vision-language continuity leads nested under C3. They remain fallbacks if SigLIP2 dependencies or checkpoints fail the access gate. |
| Food2K, Recipe1M+, VLPCook, FoodSeg/ReLeM | Food-domain pretraining evidence, but dish classification, retrieval, segmentation, or recipe-text supervision changes the target and raises overlap/licence questions. Re-enter only with a verified checkpoint and a written provenance/semantic-overlap audit. |
| C-Tran, ML-GCN, dependency-only heads | Structured-label alternatives to C4. Re-enter only if a training-only graph/state can be compared against an independent-head control without turning co-occurrence into visual evidence. |
| Inverse Cooking, FIRE, Food-R1, open-vocabulary food segmentation | Valuable food research and mechanism sources, not direct benchmark candidates: generation, multi-task VLM, or pixel/open-vocabulary targets are outside the canonical output contract. |

## Grouped exclusions

The following exclusion reasons are methodological, not claims that the systems
are ineffective:

1. **Already used locally:** ResNet and DINOv2 remain baselines, so counting
   them as new candidates would overstate the diversity of the selection.
2. **Target mismatch:** segmentation, retrieval, recipe generation, or
   open-vocabulary prompting does not equal fixed recipe-level ingredient
   prediction.
3. **Duplicate mechanism:** a depth/width/checkpoint/release change does not
   create a new family-level hypothesis.
4. **Unresolved provenance:** food or web-language pretraining needs source
   overlap, licence, and semantic-prior checks before fair comparison.
5. **Resource uncertainty:** a paper-scale result is not an 8 GB claim; compact
   checkpoint and dependency evidence must be verified.
6. **Confounded label information:** word queries, graphs, and autoregressive
   decoders may add label-side information and therefore need separate controls.

## 4A.1 → 4A.2 formal handoff

| Candidate | Qualifying sources | Accessible path to verify | Variants kept together | Unresolved questions for 4A.2 |
| --- | --- | --- | --- | --- |
| C1 EfficientNetV2 | EfficientNetV2 paper; direct food multi-label EfficientNet-family study; TorchVision model docs | Current TorchVision constructors/weights and the paper's public design | V2-S/M/L and input/training recipes | Aspect-preserving transform, local memory, and fair efficiency comparison against B1 ResNet |
| C2 Swin V2 | Swin V2 paper and official Microsoft repository | Official code/checkpoints and maintained library fallback | Tiny/Small/Base and window/resolution settings | Peak memory at project resolution, feature extraction path, and direct food-transfer evidence |
| C3 SigLIP2 VLM | SigLIP and SigLIP2 papers; official/maintained checkpoint card | Official research/Hugging Face checkpoint path | CLIP/SigLIP/SigLIP2 releases and scales | Licence/dependency, 8 GB path, native-aspect support, and text-prior separation |
| C4 query/set head | Query2Label, ML-Decoder, Inverse Cooking, direct food multi-label evidence | Public paper repositories and head implementations; train from local backbone | ML-Decoder vs Query2Label representative; learned vs word queries kept separate | Representative head, memory, output/logit contract, and non-visual controls |
| C5 MaxViT hybrid | MaxViT paper and official repository; TorchVision MaxVit weights | Official TensorFlow path or maintained TorchVision `maxvit_t` route | Tiny/Small/Base and 224/other resolution settings | Square-input assumption, archived repository, licence, and benefit beyond Swin |

**Handoff decision:** retain all five new candidates for 4A.2 dossier review.
B1 ResNet and B2 DINOv2 are retained as already-used baseline anchors and are
excluded from the selection count. No family is selected for implementation, no
hyperparameter is frozen, and no local performance claim is made. A candidate
may be demoted during 4A.2 if an access, provenance, resource, or evidence gate
fails; the reason must be recorded rather than silently replaced.

## Limitations

- The scan is broad and source-catalogued, not a systematic review or
  meta-analysis.
- No candidate was trained or memory-profiled in 4A.1.
- Food-domain evidence is heterogeneous; most published scores use different
  labels and supervision.
- “Accessible” is a documented path, not a guarantee that every checkpoint will
  remain downloadable or compatible with the current environment.
- One declared seed per configuration limits later stability claims.
