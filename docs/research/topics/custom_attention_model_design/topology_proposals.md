# Custom attention-model topology proposals

**Created:** 2026-09-08
**Last updated:** 2026-09-08
**Scope:** 4A.4.4 research comparison and implementation handoff; no trained model

## Outcome and authority

Recommend **P2, dual-scale ingredient-query readout with pooled context**, at
**S** scale. It changes how each ingredient reads image features, retains the
same pretrained trunk as the EfficientNetV2 comparison, and does not add a
second spatial-interaction stack. This is a project design judgment, not an
empirical ranking. [Decision 4A-D2](../../../project_objective/experimental_model_portfolio.md#4a-d2--custom-attention-topology)
owns adoption, the exact starting protocol and its fallback.

Exactly three topologies are considered here: P1 fused residual readout, P2
dual-scale class queries with pooled context, and P3 spatial mixing followed
by that query/context readout. They are different **computation graphs and
hypotheses**, not three independent backbone families. Their common trunk is
intentional: changing pretraining as well would obscure the readout question.
P3's image-to-image interaction before label readout is present at every size;
it is not merely P2 with more query blocks. S/M/L variants are not extra proposals.

The word custom means an attributed composition for this project. None of the
three is claimed to invent attention, class queries or feature fusion. The
selected design is not an original backbone trained from scratch and is not
an exact reproduction of Query2Label or ML-Decoder.

## Inputs, evidence and decision method

The starting revision is `993a166`. The [brief](problem_evidence_synthesis.md),
[component evidence](attention_component_evidence.md),
[compatibility synthesis](architecture_compatibility_synthesis.md),
[requirements R1--R11](../../discovery/2026-08-28/problem_model_requirements.md),
portfolio and [comparison methodology](../../../project_objective/model_comparison_methodology.md)
were rechecked. The brief's audited problem/benchmark inputs remain unchanged.
Theoretical selection uses no local training, HPO, validation/test outcomes,
checkpoint downloads or GPU measurements.

A bounded primary-source recheck on 2026-09-08 confirmed the following decisive
points; source IDs resolve through the [existing register](attention_component_evidence.md#primary-source-register).
This is synthesis of the completed research, not a new exhaustive survey.

| Evidence | What it supports | What it does not support |
| --- | --- | --- |
| S02 [Query2Label](https://arxiv.org/pdf/2107.10834), Section 3.1 and high-resolution experiment description | Learned label queries and adaptive feature pooling; a source variant omits query self-attention. | A proven advantage for the proposed two-scale, pre-norm, 224-padded ingredient adaptation. |
| S03 [ML-Decoder](https://arxiv.org/pdf/2111.12933), Section 2 | A decoder without query self-attention is a credible simpler mechanism; grouping targets much larger label spaces. | A need for grouping at 165 labels, or an equivalence between its standard frozen random queries and P2's learned queries. |
| S04 [CSRA](https://arxiv.org/pdf/2108.02456), Sections 3 and 4.2 | A compact class-specific residual readout; the single-branch VOC setting uses inverse temperature 1 and residual coefficient 0.1. | Optimal coefficients after our fusion/normalization changes, or a matched ingredient-task win. |
| S01 [Food Ingredients Recognition through Multi-label Learning](https://arxiv.org/pdf/2210.14147), Sections III-D and IV | Negative food transfer evidence: ML-Decoder helps EfficientNetB0 slightly but worsens other pairings in that study. | A universal decoder benefit, a failure of this untested P2, or comparable project label-macro AP: its metric uses micro aggregation and 500 thresholds. |
| I1/K1 [TorchVision EfficientNet v0.23.0](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/efficientnet.py) | An intact pretrained feature route, source-derived taps and parameter anchor. | Local loading, full-step memory, or task performance. |
| I6/K4 [PyTorch 2.8 MHA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MultiheadAttention.html) | Public batch-first attention with optional diagnostic weights and an optimized execution route. | Guaranteed fused-kernel eligibility or measured speed on the local GPU. |

The remaining component sources, code revisions, licences and source-task
limits are inherited through C1--C7/I1--I8, not replaced by paper leaderboard
scores. No aggregate numerical decision score is invented.

## Common specification for all three proposals

The complete equations and arithmetic definitions are in the
[compatibility synthesis](architecture_compatibility_synthesis.md). The shared
contract below applies to **each** proposal and is not an optional default.

| Field | Common proposal contract |
| --- | --- |
| Input and target | One RGB image, shared aspect-preserving fit/center-pad to 224×224 and ImageNet normalization; initially 165 ordered `v5` labels, no `<UNK>`. Phase 5 serializes common rounding/fill/interpolation. |
| Encoder and taps | Run original EfficientNetV2-S `features` once; side outputs `features[5]`: `(B,160,14,14)` and `features[7]`: `(B,1280,7,7)`, named F16/F32. These are static source-derived shapes, not measured forward outputs. |
| Pretraining | Reuse only the intact feature extractor and its buffers from `EfficientNet_V2_S_Weights.IMAGENET1K_V1`. Discard the entire stock classifier. All added modules start new; no pretrained custom-head claim. |
| New modules | Bias-free spatial projections; affine token LayerNorm with `eps=1e-5`; Xavier-uniform gain-one conv/linear weights, zero biases, LayerNorm one/zero, learned query/scale embeddings independent normal `std=0.02`. Initialize Q/K/V projections separately before any packing so a packed library default does not silently change the convention. Never reinitialize the loaded trunk. |
| Padding | Process the entire canvas without a valid-content mask. No pad-color inference, hidden geometry input or claim that padding cannot influence logits. |
| Output | `(B,L)` signed raw logits. External training/evaluation owns loss, sigmoid, calibration and thresholds; no label-axis softmax. |
| Information boundary | Images and train-supervised parameters only: no text/name embeddings, recipe input, label graph, cuisine statistics or label-to-label attention. Shared learning can still encode co-occurrence. |
| Adaptation | First choice is full trunk/readout fine-tuning. The resource fallback keeps the same topology at S and freezes the intact encoder, including normalization state and stochastic layers; new modules remain trainable. |
| Initial and fallback scale | Start S for each proposal if adopted. S is already the smallest declared size, so the fallback scale is also S, with frozen-encoder adaptation; it is **not** a smaller or equivalent full-tuning configuration. M/L document capacity options, not required runs. |
| Dependencies | Existing PyTorch/TorchVision public operators; no additional research framework, custom kernel or second encoder. Source/artifact pinning and actual checkpoint notices remain Phase 5 gates. |

Pretraining distribution and the fixed 224 input remain confounders common to
these routes. A shared encoder identity does not equal matched pretrained
recipes across the whole portfolio. If full tuning cannot fit after bounded
physical-batch/precision checks, record the frozen adaptation explicitly;
comparisons must match adaptation or be reported as different model protocols.
If even that fallback is not useful or feasible, reopen the decision instead
of silently dropping an attention/context branch or substituting a backbone.

## P1 — Fused residual spatial readout

### Hypothesis and complete flow

**H-P1:** class-specific reweighting of one aligned intermediate/coarse feature
map can complement its mean evidence without a transformer decoder. This
addresses O1 and R2/R5 directly, subject to R4/R7/R8/R9/R11.

```text
(B,3,224,224) -> intact encoder -> F16 and F32
F16 -> Conv1x1(160,D) ---------------------> (B,D,14,14)
F32 -> Conv1x1(1280,D) -> nearest(14,14) ---> add to F16 projection
sum -> Conv3x3(D,D,pad=1) -> flatten -> LN -> GELU -> U(B,196,D)
U -> normalized class scores -> mean + residual spatial weighting -> (B,L)
```

For each class, `s[n] = normalize(w_class) dot U[n]` and
`z = bias + mean(s) + 0.1 * sum(softmax(s) * s)` over 196 locations. Normalize
class weights with L2 norm clamped at `1e-12`. The bias is applied once, after
aggregation. Fix `tau=1`, `lambda=0.1`, one residual branch at every size. These
are source-inspired reference constants, not empirically selected Yummly
settings; using the source constants does not reproduce the source model.

Attention occurs only at final spatial aggregation. The mean and the upsampled
coarse contribution preserve context; upsampling does not restore lost pixels.
There is no separate coarse classifier and no coordinate/scale embedding on
this aligned grid. Each class has its own output row, without mixing labels.
An attention peak exists even for an absent class; it is not a presence mask.

### Retained and rejected mechanisms

Reuse [C3 fusion](attention_component_evidence.md#c3--feature-scales-fusion-position-and-padding)
(S16/K3 lateral addition and explicit nearest upsampling) and
[C4 residual attention](attention_component_evidence.md#gap-and-class-specific-residual-attention)
(S04), with C1's intact representation and C5's explicit normalization.
Reject a query decoder here to test the cheaper score-weighting hypothesis.
Reject a multi-branch temperature ensemble, full pyramid and label graph:
none is necessary to define this minimal intervention.

Fusion, GELU, normalized weights and the single post-aggregation bias make this
an attributed adaptation, not a copied CSRA benchmark protocol. The
[I4 provenance warning](attention_component_evidence.md#implementation-register)
remains unresolved for code copying. A mathematical composition using public
operators is the intended route; no AGPL-derived implementation is approved
for import merely because another repository displays a permissive badge.
Phase 5 would need to document its actual implementation provenance.

### Scale, cost and falsification

S/M/L change only `D=128/256/384`; the two taps, one fusion map and one residual
branch stay fixed. Initial S; fallback S with frozen encoder under the common
policy. See the [cost table](#scale-and-resource-comparison) for parameters and
MACs. A `(4,165,196)` score tensor at two bytes/element is about 0.247 MiB;
fusion/backbone activations and backward add more memory.

The narrow attention-specific control is the same route with `lambda=0`,
trained with matched policy if separately budgeted. It retains fusion,
normalization and class weights; it is not the established F32 GAP model.
The latter is a whole-protocol control with additional head differences.
No useful ranking benefit over residual-off would weaken H-P1. Gains only
against raw-F32 pooling cannot establish the residual attention's contribution.
Overconfident rare-label peaks and sensitivity to background/padding are
specific failure modes. This control is prospective, not a scheduled campaign.

## P2 — Dual-scale ingredient-query readout with pooled context

### Hypothesis and complete flow

**H-P2:** each ingredient can benefit from a learned readout over separate
intermediate/coarse features while a direct pooled route retains dish context.
Unlike P1, deciding where to read uses learned query/key projections, not the
same scalar class score used as the final evidence. It primarily addresses
O1 and R2/R5, with explicit R1/R3/R4/R7 and R8--R11 constraints.

```text
(B,3,224,224) -> intact encoder -> F16 and F32
F16 -> Conv1x1(160,D) -> flatten -> LN + e16 -> U16(B,196,D)
F32 -> Conv1x1(1280,D) -> flatten -> LN + e32 -> U32(B,49,D)
concat[U16,U32] -> U(B,245,D) -----------------> key/value memory
learned E(L,D) -> expand over B -> Tq cross-attention/FFN blocks using U
              -> final LN -> class-wise linear -> query logits(B,L)
F32 -> GAP -> biased Linear(1280,L) ------------> context logits(B,L)
query logits + context logits -----------------> output(B,L)
```

Each block is the [route-Q pre-norm block](architecture_compatibility_synthesis.md#compatible-route-q-class-queries-with-a-pooled-context-path):
`Q1=Q+MHA(LNq(Q),U,U)`, then
`Qnext=Q1+Linear(4D,D)(GELU(Linear(D,4D)(LNff(Q1))))`.
MHA uses distinct biased Q/K/V/output projections, usual `1/sqrt(D/h)` scaling
and noncausal softmax over all 245 tokens. Blocks have distinct weights;
there is no additional per-block memory normalization or query self-attention.
The class-wise output is `Wout[l] dot LN(Qfinal[l]) + bout[l]`, not a dense
projection mixing all labels. The two final logit paths have fixed coefficient
one. Added dropout is zero in the reference design; backbone behavior is retained.

Use one **learned** query per label, `G(L)=L`, at every vocabulary size.
Learned e16/e32 identify scale; there are no new coordinate positions. Joint
softmax gives 196 versus 49 entries, not equal scale mass. It can select
spatial features but does not explicitly model their pairwise arrangement.
The trunk already supplies contextual information; neither branch is a pure
measurement of local visibility or context. Logit addition is not a calibrated
probability mixture, and there is no supervised or identifiable branch decomposition.

### Retained and rejected mechanisms

Reuse [C4 learned class queries](attention_component_evidence.md#query2label-style-class-queries)
(S02/I3) with the simpler no-query-interaction boundary motivated by S03,
C3's two-scale access, and C5's residual/LayerNorm/FFN construction. The
separate context path and exact two-scale/pre-norm combination are project
hypotheses, not a published ingredient-model recipe. S01 prevents assuming
generic multi-label success will transfer.

Reject label grouping, fixed random standard ML-Decoder queries, text-based
initialization, query self-attention and label graphs. They are not needed to
test O1 at 165 labels and would change label semantics or information use.
Reject P1's score-coupled weighting to test more flexible adaptive aggregation,
not because the simpler route has failed. Reject an added image self-attention
stage here: its incremental value beyond the CNN is a different question (P3).

Use public PyTorch modules in a newly documented composition; the inspected
MIT Query2Label implementation is a reference, not a launcher/dependency to
import wholesale. The selected route does not require CSRA code or its
unresolved copying path. Standard primitives do not remove checkpoint,
attribution or licence-notice obligations.

### Scale, cost and falsification

S/M/L use `(D,h,Tq)=(128,4,1)/(256,8,2)/(384,12,3)`, fixed head width 32 and
FFN ratio four. All sizes retain both taps, learned per-label queries, the
same two logit paths and no spatial mixer. Initial S; fallback is S with the
encoder frozen, not a hidden smaller topology. At 165 labels, S has
20,814,874 total parameters, including 637,386 newly initialized parameters.
The cost table and synthesis give larger sizes and separate activation/state
examples; none measures a full training step.

The narrowest practical first control is the selected EfficientNetV2-S with
GAP/linear readout, same trunk checkpoint and matched data, input, adaptation,
loss and training policy. It tests the **whole custom head protocol**, including
extra capacity and the second scale. No useful ranking/resource trade-off
against that control weakens H-P2. A gain explained only by prevalent/contextual
labels leaves the local-evidence interpretation unsupported.

Simply removing the query branch at evaluation is a diagnostic of a trained
additive model, not an independently trained pooled baseline. Likewise, an
independently tuned Q1 EfficientNet run is a valid portfolio comparison but not
automatically a matched mechanism control. If a specifically attention-based
causal claim is essential, Phases 6--7 must budget a matched operator control
(for example query-independent spatial weights with the remaining route held
fixed) and its residual capacity/optimization limits. This stage does not add
that experiment or claim the broader control isolates attention alone.

## P3 — Late spatial interaction before ingredient-query readout

### Hypothesis and complete flow

**H-P3:** learned image-to-image interaction between intermediate and coarse
locations *before* label-conditioned readout can encode useful combinations
that P2's direct query aggregation does not capture. This addresses R2/R5/R11
but adds a stronger R4 shortcut risk and R8/R10 resource obligation.

```text
image -> same intact encoder, projections and scale embeddings as P2
      -> U(B,245,D)
      -> ONE positional spatial MHA/FFN block -> Uout(B,245,D)
learned E(L,D) -> same Tq query blocks, now using Uout -> LN -> class-wise logits
F32 -> unchanged GAP/Linear context logits -> add -> output(B,L)
```

The new block is exactly the [late mixer](architecture_compatibility_synthesis.md#optional-late-spatial-interaction):
`V0=LNs(U)`, `U1=U+MHA(V0+P,V0+P,V0)`, followed by the pre-norm residual
ratio-four GELU FFN. P is the specified fixed 2D sine/cosine encoding from
normalized feature-cell centers, generated separately on the 14/7 grids.
It is added before Q/K projections, not to V. All 245 positions can exchange
information; scale embeddings remain. The context classifier still reads
original F32, not the mixed tokens. Query/output/label semantics are identical
to P2. This is image self-attention, not attention between ingredients.

### Retained and rejected mechanisms

Reuse P2's C1/C3/C4/C5 route and
[C2 bounded late self-attention](attention_component_evidence.md#c2--spatial-interaction-and-attention-alternatives)
(S07/S08), with the explicit C3/K4 position/API contract. MaxViT supplies
portfolio evidence for testing spatial interaction, not pretrained weights for
this new mixer or proof that one late block improves ingredient prediction.
The exact placement and position convention are project choices.

Reject early dense attention, repeated spatial mixers, detection pyramids and
cross-label graphs. A single late interaction point makes the additional
hypothesis inspectable without recreating an entire transformer backbone.
Dependencies, source attribution and new-module initialization are P2's public
operator route; the mixer is newly initialized, not inherited from MaxViT.

### Scale, cost and falsification

Use P2's S/M/L widths, heads and query depths. **One spatial block is present
in S, M and L**; its width/head count follow D/h and its FFN ratio stays four.
Initial S; fallback S with frozen encoder. Each size adds `12D²+13D` parameters
to P2. At S this gives 21,013,146 total parameters and about 0.121 G dominant
new-module MACs/image. A materialized B4/two-byte S spatial-score tensor is
1.832 MiB in addition to query scores and other saved state.

The narrowest useful trained control is P2 at the same D/h/Tq and matched
policy, removing only the spatial block/its positions. It tests the added
mixing package, not a parameter-matched isolation of self-attention. No useful
benefit over P2, excessive cost, or gains dependent on cuisine/background
patterns would weaken H-P3. Pooling-only comparisons cannot distinguish this
extra stage from the query branch. This additional control is more demanding
under the available budget; neither proposal is scheduled to run as a pair.

## Scale and resource comparison

Counts are deductions for the specified graphs at `L=165`, reproduced with the
existing [scalar calculator](../../../../src_scratches/custom_attention_design/estimate_design_envelope.py)
on 2026-09-08. No new calculator/model implementation was needed. M means
million parameters; G MAC counts dominant multiply-accumulates per image, not
wall time or training FLOPs. The common trunk has 20,177,488 parameters and
about 2.849 G convolution/SE MACs at 224.

| Proposal | Size | D / h / query blocks | Total M | New M | New-module G MAC |
| --- | --- | --- | ---: | ---: | ---: |
| P1 | S | 128 / n.a. / n.a. | 20.531 | 0.353 | 0.045 |
| P1 | M | 256 / n.a. / n.a. | 21.179 | 1.001 | 0.148 |
| P1 | L | 384 / n.a. / n.a. | 22.122 | 1.944 | 0.309 |
| P2 | S | 128 / 4 / 1 | 20.815 | 0.637 | 0.058 |
| P2 | M | 256 / 8 / 2 | 22.424 | 2.246 | 0.346 |
| P2 | L | 384 / 12 / 3 | 26.395 | 6.218 | 1.076 |
| P3 | S | 128 / 4 / 1 | 21.013 | 0.836 | 0.121 |
| P3 | M | 256 / 8 / 2 | 23.213 | 3.036 | 0.570 |
| P3 | L | 384 / 12 / 3 | 28.170 | 7.992 | 1.556 |

For comparison, the adopted common-head EfficientNetV2-S and MaxViT-T have
20.389 M and 30.229 M parameters. P2-S is 426,021 parameters above the pooled
EfficientNet protocol, not 637,386 above it: the pooled head itself has 211,365
parameters. P1's fixed trunk dominates its scale range; artificially increasing
its width to match P3 would not make it a better scientific comparison.

At physical B4 and two bytes/element, P2 query scores per block occupy
1.234/2.467/3.701 MiB at S/M/L; P3 adds spatial scores of
1.832/3.664/5.495 MiB. The synthesis separately estimates memory tokens, FFN
expansions, convolution outputs and optimizer state. These are **individual
storage examples**, not an aggregate, upper bound or 8 GB fit guarantee.
Optimized attention may avoid materializing scores; actual saved tensors,
precision, workspace and optimizer behavior must be measured in Phase 5.

## Qualitative gates and selection rationale

The feature plan's six gates are applied without a weighted sum. A theoretical
pass means a traceable and plausible route, never a passed runtime gate.

| Gate | P1 | P2 | P3 |
| --- | --- | --- | --- |
| Direct O1 intervention beyond a renamed backbone | Yes: class-score weighting on fused features | Yes: class-query readout plus pooled context | Yes: explicit spatial exchange before class-query readout |
| Image-only, ordered raw logits, common benchmark | Pass by specification | Pass by specification | Pass by specification |
| Auditable components and initialization | Mathematical route explicit; reference-code copying remains conditional | Public operator composition, intact encoder, attributed query mechanism | Same route plus a specified newly initialized mixer |
| Credible, unmeasured 8 GB route | Smallest head; S/frozen route | Bounded 245-token memory; S/frozen route | Bounded late block; S/frozen route, higher cost |
| Topology-preserving S/M/L | Width only | Width/heads/repeated query blocks | Same axes; spatial block always present |
| Falsifiable, visibility limits explicit | Residual-off and pooled controls distinguished | Whole-head and narrower operator claims distinguished | Incremental spatial-stage claim needs P2 control |

**Why P2:** it tests the brief's adaptive-readout question with independently
learned selection/value/output projections, keeps an explicit uncompressed
pooled-F32 route, and avoids requiring an extra spatial-interaction hypothesis.
Its S head is modest in scalar arithmetic, and the same encoder checkpoint as
C1 makes a useful comparison possible. These are design and information-value
arguments, not evidence that flexible heads outperform simple weighting.

**Why not P1:** it is a credible simpler contender, not a failed baseline. It
ties selection to the class score and fuses scales before reading them; P2
retains separate scale access and a directly removable query branch alongside
the common pooled form. That flexibility is chosen to investigate O1 despite
the extra parameters. The code-provenance caveat adds implementation work but
is not, by itself, scientific evidence against the published mechanism.
If simplicity alone were the overriding objective, P1 would be defensible.

**Why not P3:** the CNN already supplies spatial/contextual processing, while
MaxViT covers richer spatial interaction in the established portfolio. No
direct evidence establishes that this extra mixer is needed before ingredient
queries. It introduces another attribution question and a more demanding
incremental control. Its greater symbolic cost is not proof that it cannot fit.

**Why start S:** neither more query depth nor a wider head has demonstrated
value here. D=128, four heads and one query block already instantiate the full
hypothesis with 165 learned queries and both paths. M/L remain reproducible
capacity options, not an implicit three-size HPO sweep. Larger-scale adoption
requires a recorded pre-comparative protocol revision, not automatic promotion
because memory happens to be available. The frozen-S fallback changes
adaptation, not topology; no smaller undeclared size is implied.

Direct food counterevidence prevents declaring P2 the likely accuracy winner.
The decision selects a bounded experiment worth implementing. If later matched
evidence does not support the custom readout, retain that negative result
rather than successively testing P1/P3 until something improves the benchmark.

## Implementation and comparison handoff

The binding starting contract is in [4A-D2](../../../project_objective/experimental_model_portfolio.md#4a-d2--custom-attention-topology).
Its Phase 5 acceptance checklist must include the following risks; none is
resolved by completing this research.

| Owner | Required check or decision | Claim withheld until then |
| --- | --- | --- |
| Phase 5 artifact/interface | Pin source/packages and actual checkpoint URL/hash/notices; reconstruct offline; serialize graph, taps, scale, initialization, trainability and label order in both configuration directions. | No reproducible runnable custom model yet. |
| Phase 5 shapes/gradients | Verify F16/F32 once-only traversal, B1 and L=1/50/165, finite signed logits, label-row permutation/subsetting with shared weights, gradients through both paths, no stock classifier leftovers. | Shapes/counts are source-derived; gradient behavior is untested. |
| Phase 5 transforms/state | Verify common canvas geometry and exact transform serialization; full/frozen normalization/stochastic behavior; seed and initialization order; attention reference/backend agreement. | No certified padding invariance, kernel availability or training stability. |
| Phase 5 diagnostics | Use actual forward-path convolution hooks; preserve input gradients in frozen mode; optional per-head maps with scale boundaries. Make feature factorization capability-aware or supply an explicitly qualified adapter. | Query attention is not a localization target; existing standalone-concept factorization is not drop-in compatible. |
| Phase 5 resources | Measure full optimizer/loss/backward step, allocated/reserved peaks and time with recorded batch/precision; test bounded fallback only if needed. | No 8 GB feasibility or throughput claim. |
| Phase 6 | Freeze loss, augmentation, stopping, HPO budgets and adaptation comparison; decide whether the minimal matched control is affordable. One declared seed/configuration. | No source loss or larger-size sweep is silently adopted. |
| Phase 7 | Paired primary metrics, support/context slices, uncertainty and diagnostic limits; distinguish model-protocol ranking from attention-specific evidence. | No seed-stability, visibility or causal attention claim from unmatched gains. |

Q2 still trains a **new** shared-subset model with the category's transferred
full-task hyperparameters; slicing full-model predictions is a different
evaluation artifact. Only label-indexed queries/classifier rows change with L;
feature taps, D/h/depth and paths do not. Q3 random-vocabulary controls remain
with the independent selector, and Q4 local adaptation remains separate.

Research output is now sufficient to close 4A.4 and parent 4A, subject to their
documentation checks. It does not complete 4B, release Macro-section 3, pass
Data 2.4, or authorize Phase 5 implementation/training in this step.
