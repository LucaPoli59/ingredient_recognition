# 4A.2 comparative synthesis and handoff

**Created:** 2026-08-28
**Last updated:** 2026-09-08
**Scope:** Handoff from Subphase 4A.2 to 4A.3
**Status:** Evidence synthesis; 4A.3 decision recorded separately

## Purpose and boundary

This record puts the five candidate dossiers on one qualitative comparison
frame. It is not a numerical leaderboard and does not adopt an experiment
family. The original 4A.2 handoff below is retained as evidence. The subsequent
4A.3 choice is owned by the binding
[experimental portfolio](../../../project_objective/experimental_model_portfolio.md);
its dispositions must not be inferred from this historical intake matrix.

All candidates are evaluated against the frozen image-only Yummly task:
`ingredients_target_v5`, one RGB image, 165 recipe-level labels, common split,
common metrics, test isolation, and one declared seed per configuration.
ResNet (B1) and DINOv2 (B2) are existing anchors, not new candidates.

## Method and evidence confidence

The synthesis reuses the five normalized dossiers in this collection, the
2026-08-28 discovery, the current repository model contract, and the mandatory
project-objective inputs listed in the 4A plan. Judgments use `strong`,
`moderate`, `weak`, or `uncertain`; each is a research interpretation of the
linked evidence, not a measured Yummly result. “Evidence confidence” reflects
the quality and directness of the source bundle, while “feasibility” remains
conditional until Phase 5 smoke tests.

## Common comparison matrix

| Candidate | Distinct mechanism/hypothesis | Direct food evidence | Access/provenance | 8 GB plausibility before measurement | Main transfer risk | Evidence confidence |
| --- | --- | --- | --- | --- | --- | --- |
| C1 EfficientNetV2-S | Training-aware efficient convolution and compound scaling; test AP/resource trade-off versus ResNet | Nutrition5K multi-label EfficientNet-family study; direct Recipe1M comparison | High through TorchVision; ImageNet provenance comparatively simple | Moderate-high for S; 384-square weight transform and activation memory open | Efficiency may not address weak-label observability; transform may erase local cues | High |
| C2 Swin V2-T/S | Hierarchical shifted windows and multi-scale context; test local-plus-global representation | Direct Recipe1M comparison; adjacent food segmentation/classification evidence | High via TorchVision and MIT Microsoft repo; checkpoint revision must be pinned | Moderate for T; square 256 transform and token activations open | Gains may be due to crop/resolution or cuisine shortcuts; overlap with C5 hypothesis | Medium-high |
| C3 SigLIP2 Base | Web-language-supervised visual semantics and dense/native-aspect features; test semantic prior under image-only head | Recent food/VLM Nutrition5K evidence is adjacent; no matched Yummly study | Technically high via Hugging Face; WebLI overlap and model-card accounting require audit | Uncertain for full tuning; frozen/adapter path may fit | Provenance, text/label leakage, and prompt confounding | Medium |
| C4 query/set head | Class-specific cross-attention and structured set readout; test spatial evidence beyond GAP | Nutrition5K attention decoder and Inverse Cooking set precedent | MIT reference code; paired backbone and dependency compatibility open | Moderate; cost grows with token count and decoder depth | Co-occurrence or label-query priors can masquerade as visual recognition | Medium-high |
| C5 MaxViT-T | Convolution plus local block and global grid attention; test hybrid local/global interactions | Direct Recipe1M comparison; adjacent food localisation evidence | TorchVision maintained path; official Google repo archived but Apache-2.0 | Moderate-high on paper; square partition and non-square policy open | C2 overlap, archived reference, and pooled head hiding spatial mechanism | Medium-high |

The direct food comparison source is [Visual Food Ingredient Prediction Using
Deep Learning with Direct F-Score Optimization](https://www.mdpi.com/2304-8158/14/24/4269).
Its Recipe1M metrics and fixed-threshold/F-score objective are retained only as
context: they do not select a Yummly family or transfer a score.

## Requirement-oriented synthesis

| Requirement rows | Strongest current evidence | Interpretation for 4A.3 |
| --- | --- | --- |
| R1, R7 fixed logits, ranking, calibration | C1, C2, C3 image-only adapter, C4, C5 | All candidates can expose 165 logits if their adapter is implemented without hidden text or thresholding. |
| R2, R5 partial/local evidence and small images | C2, C4, C5 mechanistically; C1 as efficient local baseline | C2 and C5 test spatial backbones; C4 tests the readout directly. Their transforms and memory must be compared before treating spatial claims as evidence. |
| R3 long tail and sparse positives | C4 for per-label query gradients; C1 for compact capacity; all require the common imbalance policy | No candidate removes the need for per-label AP, support slices, calibration, and non-visual controls. |
| R4 co-occurrence and shortcuts | C4 and C3 have the highest shortcut risk; C1/C2/C5 still learn dish context | Every selected protocol needs frequency/cuisine/image-shuffle controls; structured or text-conditioned gains cannot be called visual recognition without them. |
| R6 provenance and leakage | C1/C2/C5 ImageNet routes are easiest to audit; C3 is weakest; C4 depends on its backbone | A technically convenient checkpoint is not sufficient if source overlap or text priors are undocumented. |
| R8 resources and maintenance | C1, C5, C2-T have compact maintained TorchVision paths; C4 head is small but backbone-dependent | Phase 5 must measure peak memory and offline reload. “Fits on paper” is not a completion gate. |
| R9 fair category comparison | C1/C2/C5 with the same head; C4 through a same-backbone ablation; C3 only with image-only adapter | 4A.3 should avoid selecting two candidates that differ in backbone, text prior, head, and transform simultaneously. |
| R11 scientific information value | C4 (head mechanism), C3 (semantic prior), and either C2/C5 (spatial backbone) are complementary; C1 is a low-risk efficiency control | The final pair should answer different hypotheses and retain B1/B2 anchors. |

## Candidate-specific go/no-go register

| Candidate | Research-level go condition | Blocking uncertainty to close in Phase 5 or 4A.3 |
| --- | --- | --- |
| C1 | Keep if a compact, maintained CNN comparison is wanted beyond ResNet and the 384-square transform can be adapted transparently. | Aspect-preserving transform, checkpoint hash/terms, and measured memory. |
| C2 | Keep if hierarchical window attention is judged complementary to the chosen CNN or head protocol. | T/S token memory, fair square-compatible transform, and separation from C5. |
| C3 | Keep only if the thesis explicitly values a language-supervised representation and can document provenance/semantic-prior boundaries. | Exact model accounting, WebLI overlap, offline cache, and frozen/adapter feasibility. |
| C4 | Keep if a same-backbone readout ablation is scientifically more informative than adding another backbone. | Representative implementation, feature-map interface, dependency drift, and co-occurrence controls. |
| C5 | Keep if hybrid local/global attention offers information value beyond C2 and the TorchVision path supports the benchmark aspect ratio. | Square partition behaviour, checkpoint provenance, and whether the pooled head hides the intended mechanism. |

## 4A.2 → 4A.3 handoff packet

| ID | Concrete representative for review | Falsifiable hypothesis | Evidence confidence | Must not infer |
| --- | --- | --- | --- | --- |
| C1 | EfficientNetV2-S, independent 165-logit head | H-C1: better AP/resource trade-off than ResNet, especially for texture/colour labels, at matched policy | High | ImageNet or Recipe1M score transfer; automatic hidden-label learnability |
| C2 | Swin V2-T (S only if resources justify it), independent head | H-C2: hierarchical local/global features improve local-plus-context labels over ResNet without improving non-visual controls | Medium-high | That a transformer automatically localises ingredients or fits 8 GB |
| C3 | SigLIP2 Base image tower, image-only learned head; native-aspect variant separately named | H-C3: semantic visual pretraining improves AP, but gains shrink without text and under provenance controls | Medium | Prompted zero-shot scores as image-only evidence; WebLI semantic knowledge as visual observability |
| C4 | ML-Decoder preferred first audit; Query2Label fallback, paired with a declared backbone | H-C4: query/set readout beats GAP on local labels but not on image-shuffle/co-occurrence controls | Medium-high | C4 is an independent backbone family; query attention proves ingredient visibility |
| C5 | TorchVision MaxViT-T, independent head | H-C5: hybrid local/global attention improves AP over Swin/C1 at matched input and resource cost | Medium-high | Archived-paper results transfer; arbitrary-resolution claim removes the need for input smoke tests |

## Recommended 4A.3 decision order

The evidence supports the following decision order, without selecting a winner:

1. apply hard gates for output compatibility, access/licence, provenance,
   maintained implementation, and plausible resource path;
2. eliminate candidates whose hypothesis is redundant after accounting for the
   existing B1/B2 anchors and any C4 pairing;
3. prefer a complementary portfolio (for example, an efficient CNN plus one
   spatial/semantic or head hypothesis) over two near-equivalent attention
   backbones unless the comparison itself is the research question;
4. choose the smallest concrete variants that preserve the mechanism and defer
   exact tuning to Macro-section 6; and
5. record the two-family shortlist, rejected alternatives, hypotheses, and
   unresolved Phase 5 checks in a binding project-objective decision record.

No numeric score is assigned here. The 4A.3 decision must remain qualitative,
source-linked, and independent of local test outcomes.

## Limitations and unresolved questions

- Only C1, C2, C4, and C5 have a direct or near-direct food ingredient
  precedent; even those sources use different datasets, labels, and metrics.
- C3's semantic prior is scientifically interesting but carries the largest
  provenance and fair-comparison burden.
- C2 and C5 both expose spatial attention; 4A.3 must justify retaining both or
  select one and carry the other as custom-model evidence.
- The exact target vocabulary is frozen at 165 for current planning, but Phase
  3's later selected vocabulary is intentionally not used in model selection.
- One declared seed per configuration means no candidate can claim local
  seed-level stability from this research stage.

## Bounded source review — 2026-09-07

This addendum rechecks decision-relevant implementation facts against
TorchVision `v0.23.0` source, including the installed source files, without
constructing models or loading weights. It does not reopen broad discovery.

| Verified source fact | Consequence; not a measured result |
| --- | --- |
| [MaxViT constructor and weights](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/maxvit.py) enforce the pretrained `input_size=(224,224)`; block grids must be divisible by the partition size. | Arbitrary-resolution complexity in the paper does not guarantee arbitrary input in this constructor. An aspect-preserving image can be padded to its required square canvas. |
| MaxViT's stock classifier includes pooling, LayerNorm, a projection, Tanh, and a final output layer. | Replacing only the output preserves a different readout from a simple GAP/linear CNN. The comparison must declare the head boundary. |
| The MaxViT weight metadata records training BatchNorm momentum 0.99 instead of 0.01. | Retain this checkpoint-specific fact and verify fine-tuning normalization; do not silently repair saved statistics. |
| [EfficientNetV2-S source](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/efficientnet.py) exposes adaptive pooling and a dropout/linear classifier; its released weight transform is 384-square. | A shared smaller canvas is an explicit transfer intervention; metadata FLOPs at 384 cannot rank it against a 224-input model. |
| [Local BaseModel](../../../../src/models/commons.py) rejects non-square input shapes and supports configurable transform builders. | A square canvas with aspect-preserving content has an integration path without first requiring rectangular model interfaces. Actual preprocessing and serialization remain Phase 5 checks. |

The [Recipe1M primary article](https://doi.org/10.3390/foods14244269) remains
the existing direct-task precedent. Indexed primary text was available, but
live full-text retrieval was rate-limited; no new quantitative findings were
extracted. Its pooled-head ingredient formulation supports a clean readout
comparison, without transferring its loss, threshold, or scores to Yummly.

The [SigLIP2 model card](https://huggingface.co/google/siglip2-base-patch16-224)
was revisited. Total checkpoint size is not the isolated vision-tower count;
the dossier's accounting uncertainty remains open and cannot establish an
8 GB failure. More generally, published size/operation counts are not local
training-memory measurements.

The adopted family pair, exact starting protocols, exclusions, and 4A.4/Phase 5
handoff are in the [portfolio decision](../../../project_objective/experimental_model_portfolio.md).
The five dossiers and the original 4A.2 recommendations remain retained as
pre-decision evidence.

### Subsequent C4 evidence qualification — 2026-09-08

The [custom component review](../custom_attention_model_design/attention_component_evidence.md#task-relevant-empirical-evidence)
adds negative food-task evidence and a query-initialization correction to the
[C4 dossier](structured_multilabel_head.md). The historical intake matrix above
must not be interpreted as a verified attention-head advantage. This update
does not reopen the adopted C1/C5 pair or select a custom readout.

## Related documentation

- [Adopted experimental portfolio](../../../project_objective/experimental_model_portfolio.md)
- [C1 dossier](efficientnet_v2.md)
- [C2 dossier](swin_v2.md)
- [C3 dossier](siglip2.md)
- [C4 dossier](structured_multilabel_head.md)
- [C5 dossier](maxvit.md)
- [4A experimental-model plan](../../../plans/experimental_model_research.md)
- [2026-08-28 broad discovery](../../discovery/2026-08-28/README.md)
- [Problem-to-model requirements](../../discovery/2026-08-28/problem_model_requirements.md)
- [Current model contract](../../../implementation_details/models.md)
- [Comparative model methodology](../../../project_objective/model_comparison_methodology.md)
