# Architecture compatibility and scaling synthesis

**Created:** 2026-09-08
**Last updated:** 2026-09-08
**Scope:** 4A.4.3 theoretical handoff; no selected topology or implemented model

**Subsequent decision:** this dated compatibility handoff is now consumed by
the [three proposals](topology_proposals.md) and the portfolio's
[4A-D2](../../../project_objective/experimental_model_portfolio.md#4a-d2--custom-attention-topology).
References below to provisional routes or the next subphase describe the
4A.4.3 handoff; they do not override the later P2-S adoption.

## Outcome and boundary

Two readout routes can satisfy the [design objective O1](problem_evidence_synthesis.md#bounded-design-objective)
with explicit, compatible interfaces:

- **Residual spatial readout:** project and fuse intermediate/coarse features,
  then combine class-specific spatial weighting with average evidence.
- **Class-query readout:** query the two feature scales and add a separate
  coarse pooled-logit path for dish context.

A bounded late spatial mixer is a compatible extension to the second route,
not a necessary ingredient. These are reusable design routes, **not the three
final proposals**. The next subphase must package three distinct hypotheses,
compare them and adopt one in the [portfolio](../../../project_objective/experimental_model_portfolio.md).
Neither a more complex route nor a larger scale has demonstrated an advantage.
The [negative food-decoder evidence](attention_component_evidence.md#task-relevant-empirical-evidence)
remains part of this handoff: successful generic query-head results do not
establish that Q should beat the simpler R or pooled baseline here.

For a reproducible numerical anchor, the routes below keep an intact
EfficientNetV2-S feature extractor and vary only newly initialized modules.
This is a provisional design choice, not an amendment of the established-model
portfolio. A fixed encoder makes weight reuse honest and helps isolate readout
changes; it also places a floor under total parameter count. MaxViT feature
interfaces remain documented as an alternative, not a second encoder to run
in parallel or an automatic size fallback.

## Evidence, method and provenance

Inputs are the completed [brief](problem_evidence_synthesis.md),
[component record](attention_component_evidence.md), current
[feature plan](../../../plans/custom_attention_model.md), binding portfolio,
and [model contract](../../../implementation_details/models.md) at `0ef124f`.
The brief's audited benchmark inputs remain unchanged. No new discovery,
dataset inspection, validation/test access or candidate execution was needed.

On 2026-09-08, source-only inspection checked the installed TorchVision 0.23.0
EfficientNet, MaxViT and FPN definitions against the versioned upstream source.
The following evidence supports the synthesis:

| ID | Source and checked fact | Evidence boundary |
| --- | --- | --- |
| K1 | [EfficientNet v0.23.0 source](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/efficientnet.py): stage configuration, strides, expansion/SE rules, classifier and parameter metadata | Static source facts; spatial sizes and MACs below are deductions, not forward-pass measurements. |
| K2 | [MaxViT v0.23.0 source](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/maxvit.py): stage widths, 224 input, classifier including its bias-free final linear layer | Static interface and parameter-subtraction anchor; no local MaxViT execution. |
| K3 | [FPN v0.23.0 source](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/ops/feature_pyramid_network.py): lateral projection, nearest upsampling to an explicit target size and addition | Supports the fusion operator; our two-level, one-output route is an adaptation, not the whole library FPN. |
| K4 | [PyTorch 2.8 MHA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MultiheadAttention.html) and [SDPA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html) | Layout, head divisibility, projection and mask/dropout contracts; optimized training backend remains unverified. |
| K5 | [PyTorch initialization primitives](https://docs.pytorch.org/docs/2.8/nn.init.html) and the [supporting-layer evidence](attention_component_evidence.md#c5--residual-paths-normalization-and-supporting-layers) | Available mechanisms, not evidence that one initialization is optimal for Yummly. |
| K6 | [BaseModel](../../../../src/models/commons.py), [visualization helpers](../../../../src/commons/visualizations.py), [dashboard consumer](../../../../src/dashboards/dash/pages/model_visualization.py) | Current configuration/hook behavior; the proposed custom interface has not been integrated. |

The scalar-only [calculator](../../../../src_scratches/custom_attention_design/estimate_design_envelope.py)
reproduces parameter, MAC and storage arithmetic without importing PyTorch,
constructing tensors/models, loading weights or reading data. Its assertions
match the EfficientNetV2-S count to upstream metadata and check stage sizes.
The [calculator notes](../../../../src_scratches/custom_attention_design/README.md)
describe its execution and exclusions. Reported numbers are planning evidence.

## Shared tensor contract

Use `B` for physical batch, `L` for ordered labels, `D` for new-module width,
`h` for attention heads and `Tq` for query-block depth. `L=165` is the current
full task; `L=50` below is only an illustrative projection size, not a selected
vocabulary. All tensors in the diagrams are per-image/batch features, not
recipe text or label statistics.

| Boundary | Prospective contract | Reason or limitation |
| --- | --- | --- |
| Input | Floating RGB `(B,3,224,224)`, common fit-and-center-pad and ImageNet normalization | Inherited portfolio policy; exact transform rounding/fill remains shared Phase 5 work. |
| Backbone | Original feature stages run once in their original order; intermediate maps are side outputs | Reading a map does not change pretrained parameter shapes. New modules must not be inserted inside a reused stage silently. |
| New spatial projection | Bias-free `1x1 Conv(C,D)` on NCHW maps | Resolves channel mismatch without changing the backbone. |
| Flattening | `(B,D,H,W) -> (B,H*W,D)` with row-major spatial order | Preserve `(H,W)`, scale and token ranges for diagnostics. |
| Normalization | New token `LayerNorm(D)`, affine, `eps=1e-5` in the reference specification | Never apply it to an NCHW channel dimension by accident; original backbone normalization is unchanged. |
| Output | `(B,L)` raw logits in the experiment encoder's saved order | No sigmoid, threshold, top-k or label-axis softmax inside the model. |

**Padding resolution for these routes:** process the whole padded canvas, with
no valid-content mask. This matches the existing single-tensor model interface
and common baseline treatment. It does not make padding invisible: convolutions,
normalization, pooling and attention can all respond to it. No pixel-color
mask inference or decoder-only claim of padding removal is permitted.

A masked variant would be a separate documented intervention needing geometry
transport, partial-token rules and matched controls. It is not a pending
ambiguity in the routes specified here. Reference mask bugs recorded under
[I2](attention_component_evidence.md#implementation-register) are not imported:
use public attention primitives, noncausal attention and no mask. Ordinary
multi-head attention here also does not enable SDPA's grouped-query-attention
option; that option is different from grouping output labels in ML-Decoder.

## Source-derived feature interfaces

The following shapes follow from K1's stride/padding configuration at 224.
`features[i]` means the output of the entire sequential stage, not its first
block. No forward pass has verified them yet.

| EfficientNetV2-S tap | Channels | Grid | Tokens | Use in the envelope |
| --- | ---: | --- | ---: | --- |
| `features[0]`, stem | 24 | 112×112 | 12544 | Inherited early processing only |
| `features[1]` | 24 | 112×112 | 12544 | Inherited early processing only |
| `features[2]` | 48 | 56×56 | 3136 | No new dense attention here |
| `features[3]` | 64 | 28×28 | 784 | Conditional finer-evidence alternative |
| `features[4]` | 128 | 14×14 | 196 | Intermediate stage; not the selected 14-grid tap |
| `features[5]` = `F16` | 160 | 14×14 | 196 | Main intermediate readout tap |
| `features[6]` | 256 | 7×7 | 49 | Retained inside the backbone |
| `features[7]` = `F32` | 1280 | 7×7 | 49 | Final projected map; context/coarse tap |

The stride labels `16` and `32` identify nominal sampling intervals, not
receptive-field sizes or guaranteed visible-ingredient resolution. `F16` can
already encode broad context. Using it does not establish that small objects
survive the common resize.

K2 gives an alternative MaxViT-T interface: `stem` produces `(B,64,112,112)`;
`blocks[0:4]` produce grids 56, 28, 14 and 7 with channels 64, 128, 256 and
512 respectively. Its corresponding two taps would be `blocks[2]` and
`blocks[3]`, with new projections `256 -> D` and `512 -> D`. Keep its block/grid
operators, relative biases, normalization and checkpoint topology intact.
This changes the representation protocol and pretraining context; it is not
a width scale of an EfficientNet-based route.

## Compatible route R: fused residual spatial readout

This route instantiates the [C3 fusion](attention_component_evidence.md#c3--feature-scales-fusion-position-and-padding)
and [C4 residual-readout](attention_component_evidence.md#gap-and-class-specific-residual-attention)
mechanisms. A complete reference flow is:

```text
image -> intact feature extractor -> F16 (B,160,14,14)
                                 -> F32 (B,1280,7,7)
F16 -> 1x1 projection -> P16 (B,D,14,14) --------------------+
F32 -> 1x1 projection -> P32 (B,D,7,7) -> nearest to (14,14) -+-> add
add -> bias-free 3x3 Conv(D,D,padding=1) -> flatten -> LN(D) -> GELU
    -> U (B,196,D) -> class-specific residual aggregation -> logits (B,L)
```

The fusion has explicit target size and matching channel width; no cropping,
implicit broadcasting or extra pyramid levels. The `3x3` layer smooths/mixes
the fused map and is newly initialized. Normalization is after the fusion,
not inside the pretrained stages. Coarse features arrive at each aligned
intermediate location; upsampling does not recreate lost image detail.
No extra coordinate or scale embedding is used for this aligned single grid.

For image `b`, class `l`, location `n`, define

\[
s_{bln}=\bar w_l^T U_{bn},\qquad
a_{bln}=\operatorname{softmax}_{n}(\tau s_{bln}),\qquad
z_{bl}=b_l+\frac1{196}\sum_n s_{bln}
             +\lambda\sum_n a_{bln}s_{bln}.
\]

`w_bar` is the per-class weight divided by its L2 norm clamped below at
`1e-12`. `tau` is positive and multiplies scores (inverse-temperature
convention). `lambda` is a fixed nonnegative residual coefficient, positive
for the active attention route and zero for its residual-off control. Both are
serialized nontrainable scalars; their initial numerical values belong in the
4A.4.4 proposal. Their values do not change these parameter counts. There is
one residual-attention branch, not CSRA's multi-head ensemble. The added
per-class bias is applied **once after aggregation**; together with the fusion
and normalization this makes the route a CSRA-inspired adaptation, not an
exact reproduction or a claimed new attention mechanism.

The mean term retains dish context while the weighted term changes where each
label draws evidence. Both use the same features/class weights. Turning off
the residual (`lambda=0`) gives this route's own normalized pooled control;
it does not reconstruct the established raw-`F32` GAP classifier. Class-weight
normalization and the new fusion can themselves change score scale and capacity.
Matched controls must distinguish the full route from its attention component.

## Compatible route Q: class queries with a pooled context path

This is a Query2Label-inspired adaptation of
[C4](attention_component_evidence.md#query2label-style-class-queries), without
label self-attention. It is not a reproduction of standard ML-Decoder.

```text
F16 -> 1x1 projection -> flatten -> LN(D) -> add scale embedding e16 -> U16
F32 -> 1x1 projection -> flatten -> LN(D) -> add scale embedding e32 -> U32
concat along token axis: [U16; U32] -> U (B,245,D)
learned class queries E (L,D), expanded over B
    -> Tq independent cross-attention/FFN blocks attending to U
    -> LN(D) -> per-class linear readout -> local logits (B,L)
F32 -> GAP -> Linear(1280,L) ---------------------> context logits (B,L)
local logits + context logits -------------------> final logits (B,L)
```

Each block has distinct weights, biased `D -> D` query/key/value/output
projections and the following residual/normalization order:

```text
A = MultiHeadCrossAttention(query=LNq(Q), key=U, value=U)
Q1 = Q + A
Qnext = Q1 + Linear(4D,D)(GELU(Linear(D,4D)(LNff(Q1))))
```

The memory already has per-scale normalization before its scale embedding;
there is no extra per-block memory LayerNorm in this specification. There
are no shared weights between blocks and no label-to-label attention or graph.
Projection to heads gives queries `(B,h,L,D/h)` and keys/values
`(B,h,245,D/h)`; merging attention heads returns `(B,L,D)` before the residual.
Final readout is `z_local[b,l] = Wout[l] dot LN(Qfinal[b,l]) + bout[l]`,
not a dense `L*D -> L` projection mixing labels.

The reference equations have zero added dropout. If a later training protocol
adds dropout, its locations and rates must be explicit and shared across the
declared scale family; evaluation uses zero attention dropout. The inherited
backbone retains its own regularization, rather than being globally rewritten.

Two learned scale embeddings identify intermediate versus coarse tokens.
Spatial coordinate embeddings are deliberately absent in this pure readout:
aggregation is invariant to token order when features and scale IDs travel
together. CNN features still contain location/context information inherited
from convolutions and boundaries. Joint attention across 196 versus 49 tokens
does **not** give equal prior mass to each scale; the larger token set has more
entries in the same softmax. No unsupported scale-balancing benefit is claimed.

The separate coarse linear path is available even when a class has no visible
region. Its logits are summed with the local branch at coefficient one; the
sum is a discriminative score, not a calibrated mixture of probabilities.
Removing the query branch leaves the exact *form* of the common pooled
baseline. Independently trained weights need not be identical. The local
branch can also exploit context: the two paths do not identify direct versus
contextual recognition causally.

The same-trunk GAP comparison controls encoder identity, not the added head's
capacity, fusion or optimization. It tests the complete readout protocol. A
claim specifically about learned spatial weighting would need a separately
defined operator-level control; this synthesis does not schedule one or claim
that branch removal already isolates that narrower effect.

### Optional late spatial interaction

A proposal may insert **one** pre-norm self-attention/ratio-4 FFN block on the
245 memory tokens before class-query decoding. Its exact reference order is
`V0 = LNs(U)`, `U1 = U + MHA(V0+P, V0+P, V0)`, then
`Uout = U1 + Linear(4D,D)(GELU(Linear(D,4D)(LNff(U1))))`.
Here `P` is added to query/key inputs **before** their learned projections;
`Uout` becomes the memory for the unchanged class-query stack. The mixer
has its own two affine norms and parameters. It remains present at every
S/M/L size of a topology that adopts it; adding it only in L would change the
mechanism, not merely size. No early dense self-attention is introduced.

For this option, include a fixed 2D sine/cosine encoding in spatial Q/K only:
use normalized feature-cell centers `x=(col+0.5)/W`, `y=(row+0.5)/H`, concatenate
sin/cos channels for x and y, and use frequencies
`10000**(-i/(D/4))`, `i=0,...,D/4-1`, with arguments `2*pi*x*frequency` and
`2*pi*y*frequency`. `D` is divisible by four. Position is generated separately
for the two grids in their saved token order, has no learned parameters and
is not injected into V. Existing scale embeddings remain part of U.
This is an explicit project convention, not a claim to reproduce a source
model's exact position scheme or pretrained weights.

The mixer provides a different possible hypothesis: image features exchange
information *before* label-dependent readout. Whether this adds anything over
the intact CNN's receptive fields is for a later control, not settled here.

### Conditional finer-scale route

Replacing the 14-grid tap by `features[3]` gives projected 28-grid tokens and
`N=784+49=833`. This is shape-compatible, with `64 -> D` in the fine branch,
but must remain a fixed scale choice within a proposal's S/M/L family. The
calculator records it as a conditional alternative: the large query stack
costs about 1.827 G head MACs per image versus 1.076 G at `N=245`.
It is not ruled out as impossible on 8 GB, nor promoted merely because it
contains finer locations. Early features have different semantics; their
usefulness at 224 remains unknown. No additional mixer is assumed for this
conditional envelope.

## Label identity and vocabulary projection

For route Q, specify **`G(L)=L`**, with one independently initialized **learned**
query per output label at every vocabulary size. Grouped decoding and its
`min(100,L)` regime are not part of the retained route. This avoids silently
changing from grouped to full-query decoding when moving from 165 to around
50 labels. The chosen head counts below refer to attention subspaces, not
ingredient groups. Route R likewise has one classifier row per label.

Save query/classifier row order with the experiment's encoder identity. In
evaluation, jointly permuting label queries, classifier rows, biases and
context-head rows must permute outputs only; spatial/FFN layers do not depend
on label index. With shared weights and no query self-attention, selecting
query rows alone cannot alter another retained query's deterministic output.
These are algebraic invariants for tests, not claims about retrained subsets.

Under the [comparison methodology](../../../project_objective/model_comparison_methodology.md),
the selected-vocabulary run is a **new training run with transferred
hyperparameters**, not a sliced trained checkpoint. Full-model predictions
restricted for paired evaluation and a newly trained subset model must remain
different artifacts. Rebuilding `L` changes only label-indexed tensors; width,
feature taps, number of blocks and preprocessing remain the same. A shared
seed does not automatically yield matching random rows for differently shaped
initialization calls; record the initialization policy rather than claiming
paired initialization by seed alone.

## Initialization, trainability and reproducibility

The reference envelope separates architecture definition from initialization:

| Path | What can be reused | Interpretation and disposition |
| --- | --- | --- |
| Intact pretrained EfficientNetV2-S trunk plus new modules | All original `features.*` weights and normalization buffers from `EfficientNet_V2_S_Weights.IMAGENET1K_V1`; no stock classifier | Preferred planning route. New projections, fusion, queries and heads are not pretrained. Pin artifact URL/hash and notices in Phase 5. |
| Same topology, random trunk | No external weights; original constructor initialization for the unchanged trunk, explicitly specified new-module initialization | Structurally compatible, but tests training from scratch as well as readout. No expectation of successful optimization under the same budget. Not a free substitute for pretrained results. |
| Intact MaxViT-T trunk plus the specified channel adapters | Its own `MaxVit_T_Weights.IMAGENET1K_V1` feature weights; none of the complete stock classifier | Compatible alternative representation protocol. Recompute counts and revise the hypothesis in 4A.4.4 before choosing it. |
| Arbitrarily narrower/wider or interleaved pretrained stages | At most explicitly shape- and operation-matched submodules, not a full pretrained encoder | Not retained in this envelope. A new small trunk needs its own source-to-tensor specification; matching final width alone is insufficient. |

For the reference new modules, specify Xavier-uniform weights with gain one
for convolutions/linear projections, zero biases, LayerNorm weight one/bias
zero, and independent normal `std=0.02` query/scale embeddings. Class-weight
normalization in R occurs in its forward definition, not by permanently
replacing the trainable weights. These are auditable initialization
conventions, **not empirically selected hyperparameters**. A proposal may
revise them explicitly with a reason; no global initialization pass may run
over an already loaded pretrained trunk. Save the seed, operation order and
exact initialization configuration/checkpoint.

The first feasibility mode is full fine-tuning with all new modules trainable,
preserving the trunk's original normalization modules and settings. For small
physical batches, BatchNorm behavior is an explicit Phase 5 stability risk;
gradient accumulation does not resolve it. A frozen-trunk fallback disables
parameter gradients and holds trunk normalization/stochastic layers in eval
mode while the new modules train. Do not silently call that full fine-tuning
or freeze the newly initialized readout. This is an alternative feasibility
mode, not another required campaign or a solution for a random trunk.

TorchVision source reuse follows its BSD-3-Clause source licence; checkpoint
provenance remains separate. The [CSRA code-provenance caveat](attention_component_evidence.md#implementation-register)
is still open. R specifies the published mathematics, not permission to copy
AGPL source into another codebase. Phase 5 must resolve its chosen implementation
route and preserve attribution. Q uses standard public operators; the old
Query2Label/ML-Decoder launchers, private helpers and mask behavior need not be
imported. No external repository or package was installed in this stage.

## Topology-preserving S/M/L envelope

The provisional axes are new-module width and, for Q, repetitions of the same
query block. **Keep the original backbone, tap identities, fusion/readout type,
context paths, label semantics and position policy fixed within one topology.**

| Size | New width D | Attention heads h | Head width D/h | Query blocks Tq | R residual branches | Optional spatial-mixer blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S | 128 | 4 | 32 | 1 | 1 | 1 if adopted |
| M | 256 | 8 | 32 | 2 | 1 | 1 if adopted |
| L | 384 | 12 | 32 | 3 | 1 | 1 if adopted |

R uses D but has no multi-head query decoder; its attention-head/depth columns
are inapplicable. Its fixed single residual branch and fusion structure remain
unchanged. FFN ratio four is fixed where an FFN exists. Width/head changes do
not alter backbone parameters or imply new pretrained custom blocks.

S/M/L are sizes **within a route**, not globally matched parameter budgets.
In particular, the fixed roughly 20.18 M trunk dominates R. Artificially
inflating its classifier to match Q's L count would add capacity without a
design reason. Changing to EfficientNetV2-M/L, changing feature stride, adding
a label graph, or dropping a context path is not a fallback under these axes.

### Parameter arithmetic

The scalar count reconstructs 20,177,488 parameters in the EfficientNetV2-S
feature extractor. Adding its original `1280 -> 1000` biased classifier gives
the source total 21,458,488. Under the common 165-label linear head it becomes
**20,388,853**. MaxViT-T's source total is 30,919,624; subtracting the complete
stock LayerNorm/linear/Tanh/bias-free output classifier leaves 30,143,944,
or **30,228,589** with the common `512 -> 165` biased linear head.
Buffers are not trainable parameters and are excluded from these counts.

Let `P0=20,177,488`. For the specified 14+7 routes:

\[
P_R-P_0=9D^2+(1442+L)D+L,
\]
\[
P_Q-P_0=(1448+2L)D+1282L+T_q(12D^2+13D).
\]

R includes both projections, the `3x3` fusion, LayerNorm and class weights/bias.
Q includes both projections and their norms, scale/query embeddings, final
query norm, query classifier, context classifier and all decoder blocks.
Each attention/ratio-4-FFN block contributes `12D²+13D`, including biases and
two affine norms. The optional spatial mixer adds one such block; fixed
position encodings add no parameters. These exact equations concern the
specified reference construction, not all CSRA/Query2Label implementations.

| Route, L=165 | S total M (new M) | M total M (new M) | L total M (new M) |
| --- | ---: | ---: | ---: |
| R, fused residual | 20.531 (0.353) | 21.179 (1.001) | 22.122 (1.944) |
| Q, class-query/context | 20.815 (0.637) | 22.424 (2.246) | 26.395 (6.218) |
| Q plus one late spatial mixer | 21.013 (0.836) | 23.213 (3.036) | 28.170 (7.992) |

This places the retained envelope near the convolutional baseline and below
the selected hybrid baseline in total parameters. That is not a memory or
throughput ranking. In frozen mode, only the parenthesized new-module count
is trainable, while the whole model still occupies weight storage.

At illustrative `L=50`, Q totals become 20.638/22.217/26.159 M. Label reduction
reduces query length and label-indexed parameters but does not shrink the
trunk; it cannot be assumed to halve training cost. `L=50` is not a Phase 3
decision and does not select ingredients.

### Dominant compute and activation pressure

MACs mean multiply-accumulate operations, not an unspecified published FLOP
unit. One multiply and one addition would commonly be counted as two FLOPs.
The calculator counts dense convolutions, depthwise/SE linear operations,
projections and attention matrix products. It omits normalization, activation,
softmax, pooling/interpolation, elementwise gates and backward; there is no
throughput estimate. Static EfficientNetV2-S trunk arithmetic at 224 gives
about **2.849 G convolution/SE MACs per image**, not its 384-input model-card
operation figure.

For Q, the dominant per-block terms are
`(2N + 10L)*D² + 2*L*N*D`: K/V projection, query/output projection, ratio-4
query FFN and the two attention products. A spatial mixer adds
`12*N*D² + 2*N²*D`. These are per-image arithmetic terms. Batch scales their
work and storage; FFNs and projections can dominate at modest N.

| New modules only, L=165 | S G MAC/image | M G MAC/image | L G MAC/image |
| --- | ---: | ---: | ---: |
| R | 0.045 | 0.148 | 0.309 |
| Q | 0.058 | 0.346 | 1.076 |
| Q plus one late spatial mixer | 0.121 | 0.570 | 1.556 |

For illustration, set physical `B=4` and two bytes per stored activation.
Neither is a frozen training setting. One materialized score tensor contains
`B*h*L*N` elements for query attention or `B*h*N²` for self-attention.

| Individual tensor | S MiB | M MiB | L MiB |
| --- | ---: | ---: | ---: |
| Q memory `(B,245,D)` | 0.239 | 0.479 | 0.718 |
| Q score matrix, one block | 1.234 | 2.467 | 3.701 |
| Expanded query FFN `(B,165,4D)` | 0.645 | 1.289 | 1.934 |
| Optional mixer score matrix | 1.832 | 3.664 | 5.495 |
| Optional spatial FFN `(B,245,4D)` | 0.957 | 1.914 | 2.871 |

R's class-score tensor `(4,165,196)` is 0.247 MiB independently of D.
The largest individual convolution output in the scalar trunk traversal is
`(B,192,56,56)`: 4.594 MiB at these assumptions. Summing every convolution
output gives about 93.03 MiB, but this is **neither total saved activations nor
an upper bound on training memory**: it omits separate normalization/nonlinearity
saves, residual lifetimes, backward intermediates and workspaces. Multiple
attention blocks can retain multiple tensors; not just the largest one.

For FP32 parameters, gradients and momentum, an SGD-like persistent-state
estimate is `12*P` bytes; two-moment Adam-like state gives `16*P` bytes.
This is roughly 238/318 MiB for Q-S, and 322/430 MiB for Q-L plus the mixer.
It excludes buffers, optimizer temporaries, master copies if used, allocator
reservation and activations. Mixed precision does not guarantee half-sized
optimizer state. SDPA can avoid a stored score matrix, while its mathematical
backend can keep FP32 intermediates. None of these figures certifies an
8 GB training step.

### Initial feasibility order and exclusions

The provisional engineering order is S at a small physical batch, then M only
if the selected topology's measured headroom justifies it; L is a documented
capacity option, not another mandatory run. A plausible 8 GB route exists
through an intact pretrained trunk, bounded spatial tensors and, if needed,
frozen-trunk training of the modest new modules. Only Phase 5 can verify it.

| Combination | Disposition and reason |
| --- | --- |
| R or Q with the 14+7 interface and S/M envelope | Retain: explicit shapes, compatible pretrained trunk and bounded new-state arithmetic. Effectiveness/actual memory remain open. |
| Q with one late mixer | Retain conditionally: affordable symbolic token count and distinct interaction point; must earn its cost in proposal reasoning. |
| 28+7 query memory | Conditional alternative, not rejected as infeasible; greater token/compute pressure and weaker semantics need a specific hypothesis. |
| Dense self-attention on 56-grid early features, especially repeatedly | Exclude from this envelope: one B4/h12/two-byte score tensor is already about 900.38 MiB, before other state. Fused kernels can reduce storage but not quadratic arithmetic; early depth has no sufficient brief-specific justification. This is not a proof that every such model fails on 8 GB. |
| Full detector pyramid, high-resolution decoder, and several context/fusion stacks together | No bounded useful-resource justification in the retained evidence; require a separate synthesis before re-entry. |
| Two complete encoders running together | Not retained: changes pretraining/capacity and loses the economical same-trunk control without resolving an unmet design function. |
| Pool to one token before claiming spatial selection | Incompatible with the claimed mechanism; no spatial alternatives remain. |
| Zero/random frozen new trunk as the memory fallback | Not a useful pretrained-feature fallback; no evidence-backed route to the intended experiment. |

## Integration obligations discovered, not implemented

K6 establishes a mostly compatible image/logit boundary, with specific work
remaining for Phase 5 if a route is selected:

- Extend subclass construction, `to_config()` and `_load_config()` together:
  BaseModel's default loader filters to its known common keys. Persist route,
  backbone/weight identity, taps, widths, heads, depth, normalization,
  initialization, padding/position policy, trainability and output-order
  association. No reconstruction should fetch unspecified `DEFAULT` weights.
- Use registered submodules for side outputs and readout. Provide a
  `conv_target_layer` actually traversed by the forward and a classifier target
  with declared tensor semantics. Standard inference still returns only logits;
  optional diagnostic attention outputs must not change that contract.
- Preserve input-gradient diagnostics in frozen mode. The existing Grad-CAM
  helper makes the input require gradients; an unconditional `no_grad` block
  inside a new backbone wrapper would defeat that existing path.
- **Do not promise drop-in feature factorization.** The current helper feeds
  individual concept vectors directly to `factorization_classifier_layer`,
  and the dashboard invokes it unconditionally. Q's classifier expects one
  vector per class after image-conditioned decoding; applying it to a generic
  concept is not the same operation. R's normalized/fused attentive score also
  differs from a generic linear concept score. Add an explicit supported
  diagnostic adapter with a qualified interpretation, or a capability-aware
  dashboard path that omits unsupported factorization. Returning a dummy
  linear head or silently labelling a proxy as the model output is not valid.

Phase 5 verification must include odd vocabulary sizes (including 1, 50 and
165), batch-one shape/gradient behavior, label-row permutation, transform
consistency, no unwanted classifier parameters, finite signed logits,
serialization and offline checkpoint reconstruction, frozen/eval behavior,
diagnostic compatibility, and full-step allocated/reserved memory. Reference
attention and any optimized backend must agree within a declared numerical
tolerance in evaluation. These are a handoff checklist, not tests already run.

## Handoff to 4A.4.4

| Brief question | 4A.4.3 resolution | Remaining owner |
| --- | --- | --- |
| Q1: simple label-dependent aggregation | Complete R and Q equations with separate simpler controls | 4A.4.4 chooses the hypothesis/topology; Phases 6--7 budget controls. |
| Q2: feature scales and cost | Verified-source taps; 14+7 primary envelope, 28+7 conditional; scalar counts | 4A.4.4 fixes taps per proposal; Phase 5 measures their runtime behavior. |
| Q3: context without visible localization | R mean/coarse fusion; Q explicit pooled-logit residual; no heatmap-as-presence claim | Proposal-specific failure criteria and later diagnostics. |
| Q4: grouping and vocabulary size | Q uses learned `G(L)=L`, no query self-attention, ordered label-indexed parameters | Serialize and test the chosen contract in Phase 5. |
| Q5: weights, normalization and scale | Intact source-compatible trunk; explicit new initialization and S/M/L arithmetic | 4A.4.4 adopts exact initialization/initial scale; Phase 5 resolves artifact/licence and stability gates. |
| Q6: padding and position | All-canvas processing; aligned-grid/no-position readout; explicit coordinates only for the optional mixer | Shared transform details and diagnostic geometry in Phase 5. |
| Q7: separating mechanism from capacity | Same-trunk controls, R residual-off and Q branch-removal boundaries, remaining confounders | Three distinct falsifiable proposals, then one binding decision. |

The next phase should use this bounded envelope rather than restart a broad
component search. In particular, it must explain whether each proposal is an
adaptation of a published head, why its additional structure is warranted, and
which claims survive if a control cannot be afforded. A different trunk,
masking policy or new dependency may require revisiting the affected synthesis,
not concealing the change in an S/M/L label. No custom winner, initial training
configuration or new benchmark policy has been adopted here.
