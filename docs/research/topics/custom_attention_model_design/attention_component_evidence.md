# Attention components for weakly supervised multi-label images

**Created:** 2026-09-08
**Last updated:** 2026-09-08
**Scope:** 4A.4.2 evidence and provisional component handoff
**Source cutoff and access date:** 2026-09-08

## Research question and outcome

Which established components could serve the [design brief](problem_evidence_synthesis.md)
without presuming that a deeper attention network is more useful?

The evidence supports carrying **spatial features, a compact class-dependent
readout, and access to context** into compatibility synthesis. A convolutional
or hybrid representation with residual spatial attention or image-query
cross-attention supplies a plausible functional route. Full early global
attention, elaborate label graphs and several stacked decoders are not needed
to make that route complete. This is a provisional research recommendation;
4A.4.3 must still resolve interfaces, initialization and scale, and 4A.4.4 must
compare three topologies before a binding choice.

The direct food evidence is mixed. It supplies a reason to test the hypothesis
and a reason to keep simple alternatives, not a reason to assume its success.

## Search method and evidence strength

The review starts from the existing C1--C5 [dossiers](../experimental_model_candidates/README.md)
and follows their original papers to tokenization, spatial interaction, readout,
fusion and optimization components. Search families included food ingredient
GAP/attention comparisons; Query2Label and ML-Decoder; class-specific residual
attention; convolutional stems; window/grid/global attention; feature pyramids;
normalization; efficient attention; and attention interpretation. A bounded
recent-work check used 2025 multi-label/class-specific attention queries.

Core methods/results were inspected in the original Nutrition5K, Query2Label,
ML-Decoder, CSRA, convolutional-stem and FPN papers; EfficientViT's mechanism
was read in its full HTML. Official head code and relevant TorchVision/PyTorch
contracts were inspected as source, without imports or model construction.
Supporting and exclusion sources were reviewed at their primary abstract or
specified mechanism level; their numerical results are not imported. Search
snippets and secondary summaries are discovery aids only.

The review is not systematic or exhaustive. It stops when each brief function
has an evidenced route, a simpler comparator and a recorded material alternative.
No local training, weight download, dataset rescan, candidate validation or
test evaluation was performed. A source's experiment is not a local result.

Evidence labels used below are **mechanism** (paper/equation), **empirical**
(source-reported comparison), **implementation** (inspected code/API), and
**interpretation** (our transfer argument). Bibliographic IDs S01--S32 and
implementation IDs I1--I8 are defined in the source registers below.

## Task-relevant empirical evidence

| Evidence | Source conditions and result used | Transfer boundary |
| --- | --- | --- |
| S01: Ismail and Yuan, food ingredient prediction | Nutrition5K side-camera first frames; roughly 15K train/2.5K test images, 448-square input, ImageNet-initialized encoders, asymmetric loss and a common training configuration. Section IV reports a small ML-Decoder benefit for EfficientNetB0 but deterioration for other tested pairings. | Direct food counterevidence to a universal decoder advantage. The paper calls its metric mAP but Section III-D describes micro aggregation with 500 thresholds; it is not the project's label-macro AP. No local effect size is inferred. |
| S02: Query2Label | Generic multi-label datasets including MS-COCO and VOC; class queries, spatial cross-attention, asymmetric loss and source-specific pretrained backbones. Reported mAP improvements include high-resolution 448/576/640 configurations. | Supports adaptive readout as a mechanism; high resolution, pretraining and training recipe prevent score transfer to padded 224-square recipe images. |
| S03: ML-Decoder | MS-COCO/TResNet-M head ablation reports similar mAP for the standard decoder and full-query ML-Decoder, with a small trade-off for grouped queries. Its COCO recipe uses 448 input and Open Images backbone pretraining; throughput experiments use different 224-input/V100 conditions. | Supports removing unnecessary query interaction in that setting. It does not certify that grouping helps at 165 labels, or that the same result holds for ingredient labels. |
| S04: CSRA | VOC/COCO use 448-square inputs, pretrained backbones and BCE; WIDER-Attribute uses 224. The trained head uses a higher learning rate than the baseline classifier. Reported benefits motivate a small residual class-specific readout. | A simpler comparator is credible, but the trained comparison includes head-specific optimization. No direct Yummly or matched Nutrition5K CSRA result was established here. |
| S05 and S06: convolutional stems and MetaFormer | ImageNet experiments study optimization under stem changes and competitive pooling-based token mixing, respectively. | Independent mechanism evidence for local inductive bias and simple controls, not proof that attention is unnecessary in this task. |

The previous C4 dossier retained S01 as attention-decoder precedent without
highlighting its negative outcome. This review corrects that interpretation.
The study is not a universal rejection either: its fixed recipe and limited
decoder analysis leave adaptation sensitivity unresolved. The earlier
Recipe1M family comparison remains in the portfolio as backbone evidence; it
does not settle the custom readout question.

## Notation and functional boundary

Use `B` for batch size, `L` for labels, `N=Hf*Wf` for spatial tokens, `C` for
backbone channels, `D` for attention width, `h` for attention heads and `G` for
query groups. `F` is `(B,C,Hf,Wf)` or its explicitly flattened `(B,N,C)` form.
These are interface symbols, not chosen model dimensions.

A standard attention operation is

\[
\operatorname{Attn}(Q,K,V)=
\operatorname{softmax}_{\text{keys}}(QK^T/\sqrt{d_h}+M)V.
\]

S07 defines the scaled dot-product mechanism. For image self-attention,
queries/keys/values come from spatial features; for a label readout, queries
come from class/group embeddings and keys/values from image features. The
softmax distributes weight over evidence locations. It is **not** a softmax
over mutually exclusive ingredient classes. Each final class still receives
its own raw logit, with sigmoid/loss/calibration outside the model.

**Mathematical limitation:** spatial softmax sums to one even when an ingredient
is absent. A visually appealing attention peak is therefore not a presence
score. The final classifier must learn negative outputs; adding a null token
or forcing sparse maps would be an additional design choice, not an automatic
property of attention.

## C1 — Spatial representation and tokenization

**Role:** serve Q2/Q5 and R2/R5/R8 by keeping image features spatially available
until the chosen aggregation point.

S08 establishes patch-token visual transformers. S05 compares a coarse
patchifying stem with stacked overlapping convolutions followed by a channel
projection, finding improved optimization behavior under its ImageNet
conditions. S09 supplies the efficient Fused-MBConv/MBConv precedent already
present in the selected portfolio. These are compatible alternatives, not
three modules to stack together.

**Provisional route:** carry a convolutional/hybrid feature extractor and its
unpooled intermediate/final maps. Preserve a plain patch route when reusing an
intact pretrained transformer is the scientific intent. Prefer source-backed
standard convolution blocks over inventing early attention without a specific
need. Exact feature taps remain open.

**Interface and cost:** a convolutional map must be projected from `C` to `D`
before token interaction when widths differ. Dense stride-4 features contain
far more tokens than stride-16/32 features at the same input. Overlapping
convolution retains a locality bias but downsampling can still discard cues.

**Initialization limitation:** replacing a pretrained patch stem or changing
stage widths does not preserve checkpoint compatibility merely because its
output has the same shape. An intact TorchVision feature extractor is a clearer
reuse path ([I1]). A new stem requires an explicit initialization and training
interpretation in 4A.4.3. S05 does not establish that this dataset can train a
large transformer from scratch.

## C2 — Spatial interaction and attention alternatives

**Role:** Q2/Q3, R2/R5/R8. Context may already be available through a backbone's
receptive field; an additional spatial-attention stack is optional.

| Mechanism and source | Interface/resource implication | Provisional disposition and limitation |
| --- | --- | --- |
| Local convolution or simple token mixing, S05/S06/S09 | Spatial map in/out; bounded neighborhood, no dense `N*N` attention scores. | Retain as representation and comparator. Local mixing alone needs depth/hierarchy or a contextual readout to aggregate across the image. |
| Late global self-attention, S07/S08 | `(B,N,D)` in/out; all-token interaction, dense score size `B*h*N*N`. | Retain as a conditional context operator at modest token counts. It can attend to source/plating cues as well as ingredients. |
| Window/shifted-window attention, S10/S11 | Partition, mask shifted boundaries, attend locally, undo partition; width/head and window geometry must agree. | Retain when finer spatial processing warrants grouping. Window masking and positional assumptions are part of the operator, not optional implementation details. |
| Block plus grid attention, S12 | Two different partitions provide local and dispersed interactions at fixed partition sizes. | Retain as inherited MaxViT evidence; a wholesale copy followed by GAP would repeat the selected family. |
| Axial attention, S13 | Attend along height and width separately; factorized interaction replaces full spatial all-pairs computation. | Defer: its factorization adds another axis of comparison without a currently unresolved function. Original generative/image evidence is not ingredient recognition evidence. |
| SE, S14 | Global channel summary gates a spatial feature map; no class-specific query. | Retain when already part of a reused convolutional block. Channel selection alone does not implement ingredient-specific spatial aggregation. |
| CBAM, S15 | Sequential channel and spatial gates preserve map shape. | Keep as a lightweight alternative, not an automatic addition to SE/query attention. Its spatial mask is not a distinct ingredient readout. |

The project-level preference is a small number of spatial interactions only
where needed by O1. Adding window, grid, axial and global modules to one network
would add cost and confounding without separate evidence for each contribution.
The original studies primarily measure classification/detection performance,
not visibility under recipe-level weak supervision.

The inspected TorchVision [Swin/MaxViT implementations](#implementation-register)
provide concrete partition and positional-bias references. The MaxViT weight
constructor's 224 input requirement remains the portfolio's implementation
boundary; its paper's complexity claim is not arbitrary-shape compatibility.

## C3 — Feature scales, fusion, position and padding

**Role:** Q2/Q3/Q6, R2/R5/R8. Allow aggregation to use local detail and context
without silently changing the image policy.

S16's feature pyramid uses lateral projections and top-down upsampling to
combine semantically different resolutions. Detection evidence motivates
multi-scale features, but does not demonstrate a benefit for this recipe task.
The TorchVision implementation uses explicit target sizes for upsampling and
lateral addition ([I1]).

**Candidates:** retain a single-scale spatial readout as the simplest route;
carry either a small FPN-style fusion or separately projected scale tokens as
conditional alternatives. For addition, channels and spatial alignment must
match. For concatenation, align feature width and preserve scale identity;
more tokens raise attention cost. Upsampling a coarse map does not restore
image detail that the representation discarded. A full detector pyramid and
its high-resolution levels are not automatically needed.

S07 supports explicit positional encodings; S10/S11 support relative spatial
biases. A readout over CNN features can also be deliberately order-invariant:
the inspected ML-Decoder has no added positional encoding. Its keys/values
can still encode context inherited from the CNN. Choose positional information
according to the intended spatial operation, not by stacking absolute,
relative and scale embeddings by default. Exact use remains Q6 for 4A.4.3.

**Padding is unresolved implementation work, not free metadata.** The common
transform's image geometry can generate a valid-content mask without using
recipe labels. The current model contract does not provide that mask as a
separate input. 4A.4.3 must explicitly choose either all-canvas processing or a
supported mask flow and record the additional intervention. Do not infer the
mask by matching pixel colors: real image pixels can equal the pad value.

A feature token near a border mixes valid pixels and padding through its
receptive field; a center-point mask does not undo that mixing. Masking only the
decoder does not remove padding influence introduced earlier. If masks are
used, define partially valid tokens, pooling denominators and all-masked
behavior. SDPA boolean masks use `True` for included entries, while
`MultiheadAttention` masks use `True` for excluded entries ([I5/I6]). The
opposite conventions need explicit handling.

## C4 — Readout: simple residual attention and query decoding

**Role:** Q1/Q3/Q4/Q7, R1--R4/R7/R9/R11. This is the most direct component
family for the primary objective.

### GAP and class-specific residual attention

GAP plus a linear classifier is the common control. CSRA (S04) adds a
class-dependent spatial contribution to a global average contribution. A
single-head algebraic view is

\[
s_{ln}=\bar w_l^T f_n,\quad
a_{ln}=\operatorname{softmax}_n(Ts_{ln}),\quad
z_l=\frac{1}{N}\sum_n s_{ln}+\lambda\sum_n a_{ln}s_{ln}.
\]

Here `w_bar` denotes the normalized classifier weight in the inspected
implementation ([I4]); `T` multiplies scores and therefore acts as inverse
temperature. Published multi-head settings combine several such branches.

**Interpretation:** this supplies the brief's local-plus-context function with
few extra operations and without label-to-label interaction. The attention
score and classifier are coupled, which limits flexibility but gives a useful
low-complexity comparator for learned query/key projections. Class maps cost
`B*L*N`; they need not create an `N*N` matrix. Setting the residual coefficient
to zero recovers that head's own normalized pooled score, not necessarily an
arbitrary existing unnormalized GAP head.

**Disposition:** retain as a simple candidate/reference. Its code-reuse terms
need resolution ([I4]); this review does not approve copying the reference
implementation. No claim of ingredient localization follows from its maps.

### Query2Label-style class queries

S02 learns one query per label, queries spatial image features, and produces
class-specific logits. The original formulation includes query self-attention;
some high-resolution experiments omit it. Thus self-attention is a protocol
choice, not an inseparable requirement for every Query2Label experiment.
The official implementation exposes feature projection, query embeddings and
class-wise linear output ([I3]).

**Interpretation:** carry image-query cross-attention as the flexible candidate
when simple residual spatial weighting is insufficient. A minimal variant
without label self-attention must be named as an adaptation, not an exact
reproduction. Labels share features and optimization even without explicit
query mixing; separate sigmoid outputs do not imply statistical independence.

### ML-Decoder and grouped queries

S03 removes decoder self-attention and supports grouped readout. Its standard
closed-set reference uses **fixed random query embeddings**, with learned
projections and output weights ([I2]). Query2Label's learned queries and
ML-Decoder's fixed queries must not be conflated.

**Interpretation:** retain the cross-attention mechanism; treat grouping as a
conditional efficiency option. At 165 labels a query per class is not an
extreme-classification problem. Grouping sacrifices a one-to-one query/label
map and couples several logits through a group representation; its cost saving
must be meaningful for the chosen feature scales. A group map cannot be
displayed as a uniquely ingredient-specific explanation.

S03's removal of a fixed query transformation does not prove that repeated
self-attention among already image-conditioned queries is mathematically
redundant. That stronger claim is not adopted. S01 supplies the direct negative
food result described above.

### Readout disposition

| Option | What would justify it | Required simpler comparison |
| --- | --- | --- |
| GAP | Contextual recipe cues already suffice; efficient stable baseline. | Existing common pooled protocol. |
| CSRA-style residual weighting | Label-specific aggregation adds value with small overhead. | Same features and explicitly matched classifier policy with residual weighting removed. |
| Class-query cross-attention | Learned query/key/value projections add useful evidence access beyond simple weighting. | Same trunk plus GAP; CSRA is an additional informative reference if budget permits. |
| Grouped ML-Decoder | Query count meaningfully limits the selected route's cost. | Full-query or simpler pooled route, as the later budget allows. |
| Query self-attention / explicit label graph | A distinct, justified dependency hypothesis beyond spatial aggregation. | Deferred from the initial component route; source and control must be separately declared. |

These are component alternatives, not the three final topology proposals. No
decoder width, query count, residual coefficient, head count or training loss
is selected here.

## C5 — Residual paths, normalization and supporting layers

**Role:** Q5, R8/R10. A working attention block requires explicit residual,
normalization and feed-forward behavior; selecting an attention formula alone
does not define a network.

| Component | Evidence and mechanism | Provisional use and limitation |
| --- | --- | --- |
| Residual addition, S19 | ResNet supports learning a correction to an identity/projection path. | Retain. Shapes must match; a projection is needed when width or resolution changes. Residual paths do not make arbitrary component combinations stable. |
| LayerNorm placement, S17 | Transformer theory and NLP experiments relate pre-normalization to better-behaved initialization gradients. | Pre-norm is a reasonable candidate for newly built attention blocks. It is not proof of warmup-free or stable food training. Preserve a reused block's own normalization unless a change is explicitly part of the design. |
| Channel/token normalization, S18 and I7 | GroupNorm avoids batch statistics; LayerNorm normalizes the specified trailing dimensions. | Carry LayerNorm for token features and the pretrained backbone's normalization for reused stages. GroupNorm is an alternative for newly initialized convolutional stages, not an automatic replacement for pretrained BatchNorm. |
| Feed-forward sublayer, S07/S22 | Per-token linear expansion, nonlinearity and projection provide transformations beyond token mixing; GELU is an established activation. | Retain a conventional small FFN for a query/transformer block. ReLU versus GELU and expansion ratio remain explicit choices; no source establishes a task-specific winner. |
| Dropout / stochastic depth, S07/S20 | Activation/branch regularization is supported in the original source settings. | Available supporting components, not new mandatory training axes. Preserve inherited behavior; extra rates belong in the later declared protocol. |
| Residual scaling, S21 | CaiT studies learned small residual scales for deeper image transformers. | Conditional on depth/optimization need. Shallow custom readout does not inherit a requirement for deep-transformer stabilization machinery. |

For a newly composed pre-norm block, a useful schematic is
`x <- x + attention(norm(x)); x <- x + FFN(norm(x))`. Cross-attention also
requires an explicit policy for the memory features' normalization. This
schematic is a candidate interface, not an instruction to rewrite the
post-normalized ML-Decoder or Swin V2 blocks. Mixing their normalization
placement and weight scales silently would create a different operator.

Gradient accumulation changes the optimizer's effective batch, not the
physical batch from which BatchNorm obtains statistics. Freezing an encoder
requires declaring normalization and stochastic-layer behavior as well as
parameter gradients. The established portfolio already identifies this issue;
custom initialization and trainability must carry it forward.

Initialization belongs to the component specification: new linear/query/fusion
parameters need a saved initialization policy, while reused weights need an
exact source and compatible parameter mapping. Zeroing all class queries under
fully shared operations can preserve an unwanted symmetry; fixed *random*
queries are a distinct construction. No pretrained status is claimed for newly
inserted components.

## C6 — Efficient execution and honest memory estimates

**Role:** Q2/Q4/Q5, R8/R10. Reduce unnecessary intermediate storage before
changing the scientific mechanism solely for an advertised complexity bound.

FlashAttention (S23) changes memory access and avoids materializing the full
attention matrix while computing exact softmax attention up to numerical
differences. It does not turn the attention arithmetic into linear complexity.
PyTorch 2.8 SDPA provides optimized and mathematical backends, selected under
input-dependent constraints ([I5]). A usable backend and its speed remain
Phase 5 measurements.

**Mathematical planning estimates:** ignoring projections, self-attention
mixing costs `O(B*N*N*D)` and query cross-attention costs `O(B*G*N*D)`.
Materialized score storage is `B*h*N*N` or `B*h*G*N` elements. Query/key/value
and output projections add roughly `O(B*(N+G)*D*D)` work; FFNs and the trunk
can dominate at small token counts. Fewer heads at fixed width do not remove
all of these costs. Group decoding also retains a label-dependent output cost.

For illustration only, assume `B=8`, `h=8`, two bytes per score and a single
materialized score matrix. These are arithmetic examples, not selected batch
sizes, precision, feature taps or memory measurements:

| Interaction | Token/query counts | One score tensor, MiB |
| --- | --- | ---: |
| Global attention on a 56×56 feature grid | `N=3136` | 1200.50 |
| Global attention on a 14×14 feature grid | `N=196` | 4.69 |
| Class-query readout on a 14×14 grid | `G=L=165`, `N=196` | 3.95 |
| Class-query readout on 14×14 plus 7×7 tokens | `G=165`, `N=245` | 4.93 |
| Class-query readout on a 28×28 grid | `G=165`, `N=784` | 15.79 |

Each value is `B*h*queries*keys*2 / 2^20`. It excludes gradients, softmax
workspace, saved activations, FFNs, weights, optimizer states, allocator
overhead and the backbone. The SDPA mathematical backend may retain float32
intermediates even for half inputs. Fused kernels may avoid storing this
matrix. None of these entries is a claim that an entire training step fits
8 GB. They explain why feature scale matters more than calling a layer
"lightweight".

| Alternative | Evidence and disposition |
| --- | --- |
| SDPA / optimized MHA | Preferred execution candidate for ordinary softmax attention. `need_weights=False` enables the optimized MHA path where supported; diagnostic weight extraction is a separate path ([I5/I6]). |
| Activation checkpointing | Recomputes forward segments to reduce retained activations ([I8]); conditional engineering option with extra time cost, not evidence that an oversized design is feasible. |
| EfficientViT linear attention, S24 | Changes the attention kernel and adds multi-scale local aggregation. The paper itself identifies a local-concentration limitation of plain ReLU linear attention. Defer unless long spatial sequences actually require it; it is not interchangeable with exact softmax SDPA. |
| LoRA, S25 | Low-rank trainable updates reduce parameter/optimizer state for a frozen pretrained model. Original evidence is language-model adaptation; visual effectiveness and activation savings are unresolved here. Keep conditional for a reused backbone, not as a way to pretrain a new trunk. |
| Frozen pretrained features | A plausible resource fallback when weights remain meaningful. It changes adaptation and must be reported separately from full fine-tuning; freezing a randomly initialized new trunk is not the same fallback. |

## C7 — Label dependence, calibration and interpretation

**Role:** Q3/Q4/Q7, R2--R4/R6/R7/R9/R11.

The ML-GCN mechanism (S30) explicitly constructs label classifiers from a
label graph and word embeddings. It is evidence that dependency modeling is a
real intervention, not merely a synonym for spatial attention. The present
brief does not require a graph, recipe text or ingredient-name initialization.
Train-only structures could be studied later with a declared source/control,
but are provisionally deferred because they add a second explanation for gains.

Similarly, learned class queries encode training-distribution information;
fixed random queries acquire useful behavior through learned projections.
Neither construction guarantees freedom from priors. Independent logits mean
separate output scores, not independent labels or causal evidence.

S26 establishes that accurate neural classification need not be calibrated;
its temperature-scaling evidence is primarily multiclass and does not select
the project's multi-label calibrator. A change of head, score normalization
or residual coefficient can change score scale. Preserve the existing
validation-only calibration/threshold policy and report AP with fixed-policy
F1; no softmax over labels, hard top-k decoder or architecture-specific F-score
loss is introduced by this component review.

S27 and its response S28 debate when NLP attention supports explanation;
neither alone proves a universal statement about vision. S29 supplies a vision
example of why relevance propagation must consider the network beyond raw
attention. **Project interpretation:** expose attention as diagnostic evidence,
preserve the path to logits, and assess image dependence with declared controls.
A heatmap cannot be the annotation that establishes the same model's ingredient
visibility. Occlusion can alter the input distribution, while full-image
shuffling removes both local and contextual correspondence; neither isolates
direct ingredient recognition by itself.

The sparse/noisy-target problem remains a data/loss/evaluation issue. The
asymmetric losses in S01--S03 are part of their source experiments, not a
required companion to the proposed architecture. Changing the backbone, head,
loss, augmentation and label graph together would make O1 difficult to test.

## Implementation register

These are inspected source facts, not successful local integrations. GitHub
default-branch revisions for I2/I3 and the CSRA reference were resolved through
the public commits API on 2026-09-08. A source licence does not identify the
terms or provenance of every external checkpoint.

| ID | Exact reference and inspected behavior | Reuse consequence |
| --- | --- | --- |
| I1 | TorchVision `v0.23.0`: [EfficientNet](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/efficientnet.py), [MaxViT](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/maxvit.py), [Swin](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/swin_transformer.py), [FPN](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/ops/feature_pyramid_network.py); [BSD-3-Clause source licence](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE). | Concrete feature-map, partition and fusion references on the project's library generation. Reuse compatible components; pin weight artifacts separately in Phase 5. |
| I2 | [ML-Decoder source](https://github.com/Alibaba-MIIL/ML_Decoder/blob/8a9e984f671c9c30c98d2c45dfcaf4383381c254/src_files/ml_decoder/ml_decoder.py), revision `8a9e984` (2022-04-05), [MIT licence](https://github.com/Alibaba-MIIL/ML_Decoder/blob/8a9e984f671c9c30c98d2c45dfcaf4383381c254/LICENSE). | Standard non-ZSL queries are frozen random embeddings; one decoder layer, no added positions. Default groups are capped at `min(100,L)`; grouped outputs are flattened and truncated to `L`. The custom decoder accepts mask arguments but does not forward them to MHA. It uses a private activation helper and a JIT group loop. Direct compatibility with PyTorch 2.8 is unverified. |
| I3 | [Query2Label head](https://github.com/SlongLiu/query2labels/blob/55eb05064f4badbe03423b79e5c9d143da2dff2e/lib/models/query2label.py) and [transformer](https://github.com/SlongLiu/query2labels/blob/55eb05064f4badbe03423b79e5c9d143da2dff2e/lib/models/transformer.py), revision `55eb050` (2022-03-18), [MIT licence](https://github.com/SlongLiu/query2labels/blob/55eb05064f4badbe03423b79e5c9d143da2dff2e/LICENSE). | Learned class embedding, channel projection and group-wise linear logits. Omitted self-attention is handled by a flag in the inspected post-norm path; do not assume every path treats it identically. Old helper/dependency conventions require adaptation rather than importing its launchers. |
| I4 | [Official CSRA](https://github.com/Kevinz-code/CSRA/tree/c8480d12742459809179eb0fc4ee0a88b6b98bfa), revision `c8480d1` (2023-03-19), whose repository identifies AGPL-3.0; [MMPreTrain v1.2.0 head](https://github.com/open-mmlab/mmpretrain/blob/v1.2.0/mmpretrain/models/heads/multi_label_csra_head.py), whose header states it was modified from CSRA, under a repository [Apache-2.0 licence](https://github.com/open-mmlab/mmpretrain/blob/v1.2.0/LICENSE). | The library head verifies normalized class weights, spatial softmax and residual pooled logits. Upstream and library licence/provenance signals differ; do not infer permissive copying from the library badge alone. Keep code reuse unresolved and preserve the published mechanism as research evidence. |
| I5 | [PyTorch 2.8 SDPA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html). | Shape/mask/backend contract verified. Boolean `True` includes a position; evaluation must explicitly pass zero dropout. Kernel eligibility and numerical behavior remain environment-dependent. |
| I6 | [PyTorch 2.8 MHA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MultiheadAttention.html). | Batch-first is configurable; width must divide across heads. Boolean masks exclude `True` entries. Requesting attention weights affects the optimized execution path; averaging heads can obscure distinct patterns. |
| I7 | [PyTorch 2.8 LayerNorm](https://docs.pytorch.org/docs/2.8/generated/torch.nn.LayerNorm.html). | `LayerNorm(D)` applies to the last dimension: `(B,N,D)` and channel-first maps cannot be interchanged without a deliberate layout/normalization policy. |
| I8 | [PyTorch 2.8 checkpointing](https://docs.pytorch.org/docs/2.8/checkpoint.html). | Activation recomputation trades time for memory; stochastic behavior and checkpoint API choices belong to the saved implementation protocol. |

The source inspection establishes reusable algorithms and integration risks.
It does not establish that these old research repositories are currently
maintained or executable with the project's stack. No research launcher,
external model package or pretrained head was installed.

## Provisional compatibility and exclusions

| Combination or alternative | Disposition and reason |
| --- | --- |
| Spatial CNN/hybrid features -> compact residual or query readout -> logits | Compatible in principle; width projection, context access and normalization remain to specify. This covers every brief function without a new large trunk. |
| One feature scale versus a small projected multi-scale set | Both retained; choose using tensor/cost reasoning before topology proposals. More scales are not assumed better. |
| GAP before a spatial decoder | Reject as a route to spatial selection: after reducing to one token there are no spatial alternatives to query. |
| Class-query cross-attention plus explicit graph and deep query self-attention | Defer the additional dependency mechanisms; the primary objective does not currently justify all three. |
| Arbitrary pretrained pieces with matching output widths | Reject the compatibility inference. Operation order, tensor semantics, normalization and parameter shapes must match the actual checkpoint. |
| Dense global attention at all early image scales | Defer: score/activation growth lacks a brief-specific justification. Late attention and query readout already provide context routes. |
| FPN plus full-resolution decoder plus several fusion blocks | Defer the expansion. Dense detection evidence does not establish a need at 224 input with weak recipe labels. |
| Special-purpose linear/deformable attention | Keep S24 as an efficiency reference; no current token budget requires importing a different attention kernel or additional sampling machinery. |
| External label text, recipe-conditioned queries, autoregressive sets | Outside the initial component route; they change the information or output contract. |
| Recent medical class-localization systems | S31 uses anatomical/segmentation priors; S32 adds a causal/information-bottleneck objective. Abstract-level adjacent leads only, not retained component candidates or evidence of causal ingredient recognition. |

This is a bounded exclusion record, not a judgment that these alternatives are
inferior on their source tasks. Re-entry requires a concrete unanswered design
function, compatible information sources and a simpler comparator.

## Handoff to 4A.4.3

| Brief question | Answer supported at component level | Still to resolve |
| --- | --- | --- |
| Q1 | C4 supplies a complexity ladder: GAP, CSRA-style weighting, class-query cross-attention, optional grouping. | Exact readout semantics and fair control; no default winner. |
| Q2 | C1--C3 support spatial convolution/hybrid features with optional small multi-scale fusion. C6 bounds attention growth. | Exact feature taps, layouts, token counts and scale-alignment policy. |
| Q3 | Context can remain in feature receptive fields, a pooled residual path or image-attending queries. Presence is decided by logits, not attention peaks. | Which context path each candidate route uses and how to test its contribution. |
| Q4 | Fixed versus learned queries and optional label mixing are distinct choices. Grouping is not compelled by 165 labels. | Choose and serialize `G(L)` and query/output ordering for full and projected vocabularies; avoid silently inheriting ML-Decoder's changing group regime. |
| Q5 | C5/I1--I3 identify standard supporting blocks and honest initialization alternatives. | Exact trainable modules, reusable weights, normalization and initialization; S/M/L parameter bands. |
| Q6 | C3/I2/I5--I7 expose padding, position and layout obligations. | Valid-token semantics and mask transport, or explicit all-canvas processing. Repairing reference mask behavior is Phase 5 work if needed. |
| Q7 | Negative food evidence and simple heads make the claim falsifiable. | Mechanism-specific control and failure criterion for each later topology; Phases 6--7 own execution budgets. |

The compatible functional route is sufficient to close component research.
4A.4.3 must turn the retained options into explicit interfaces and estimates,
not reopen an unrestricted layer search. Research cannot close local
optimization, calibration, GPU feasibility or ingredient-visibility questions.

### Component reuse register

The [compatibility synthesis](architecture_compatibility_synthesis.md) and
completed [three-proposal comparison](topology_proposals.md) consume these
entries. The links below preserve the source-to-proposal trail without
changing the earlier component evidence. The portfolio, not this research
register, owns the adopted P2-S decision.

| Entry | Compatibility consumer | Proposal link |
| --- | --- | --- |
| C1 representation; C2 spatial interaction | [Source-derived taps](architecture_compatibility_synthesis.md#source-derived-feature-interfaces), [optional late mixer](architecture_compatibility_synthesis.md#optional-late-spatial-interaction) | [Common encoder](topology_proposals.md#common-specification-for-all-three-proposals); [P3 retains late mixing](topology_proposals.md#p3--late-spatial-interaction-before-ingredient-query-readout), P1/P2 omit it. |
| C3 fusion/position/padding | [All-canvas contract](architecture_compatibility_synthesis.md#shared-tensor-contract), [aligned fusion](architecture_compatibility_synthesis.md#compatible-route-r-fused-residual-spatial-readout) | [P1 aligned fusion](topology_proposals.md#p1--fused-residual-spatial-readout); [P2 scale-separated memory](topology_proposals.md#p2--dual-scale-ingredient-query-readout-with-pooled-context); P3 adds specified positions. |
| C4 readout | [Residual route](architecture_compatibility_synthesis.md#compatible-route-r-fused-residual-spatial-readout), [query route](architecture_compatibility_synthesis.md#compatible-route-q-class-queries-with-a-pooled-context-path), [label identity](architecture_compatibility_synthesis.md#label-identity-and-vocabulary-projection) | [P1 residual](topology_proposals.md#p1--fused-residual-spatial-readout), [P2/P3 learned queries](topology_proposals.md#p2--dual-scale-ingredient-query-readout-with-pooled-context); grouping/text/query self-attention excluded. |
| C5 supporting layers; C6 cost | [Initialization](architecture_compatibility_synthesis.md#initialization-trainability-and-reproducibility), [scaling estimates](architecture_compatibility_synthesis.md#topology-preserving-sml-envelope) | [Shared conventions](topology_proposals.md#common-specification-for-all-three-proposals), [nine size estimates](topology_proposals.md#scale-and-resource-comparison); only P2-S adopted, no size campaign. |
| C7 interpretation | [Interface/diagnostic limits](architecture_compatibility_synthesis.md#integration-obligations-discovered-not-implemented), [proposal handoff](architecture_compatibility_synthesis.md#handoff-to-4a44) | [Qualitative gates](topology_proposals.md#qualitative-gates-and-selection-rationale) and [implementation/comparison obligations](topology_proposals.md#implementation-and-comparison-handoff); no visibility or isolated attention-effect claim. |

## Primary-source register

The record cites primary papers below. Reading depth is **targeted method and
result passages** for core evidence, **mechanism/abstract** for supporting context,
and **abstract lead only** for excluded recent directions. Bibliographic
recency does not substitute for direct task evidence or independent replication.

| ID | Primary source | Evidence used and reading depth |
| --- | --- | --- |
| S01 | Ismail and Yuan, *Food Ingredients Recognition through Multi-label Learning*, ESSCIRC EAI 2022, [paper][S01] | Direct ingredient evidence; Sections III--IV and appendix; negative transfer retained. |
| S02 | Liu et al., *Query2Label: A Simple Transformer Way to Multi-Label Classification*, 2021, [paper][S02] | Generic multi-label mechanism and experiments, Sections 3--4; official implementation separately inspected. |
| S03 | Ridnik et al., *ML-Decoder: Scalable and Versatile Classification Head*, WACV 2023, [author manuscript][S03] | Sections 2--3 and COCO/throughput appendices; fixed/grouped-query and head-ablation evidence. |
| S04 | Zhu and Wu, *Residual Attention: A Simple but Effective Method for Multi-Label Recognition*, ICCV 2021, [paper][S04] | Class-specific residual formulation and multi-label comparison context. |
| S05 | Xiao et al., *Early Convolutions Help Transformers See Better*, NeurIPS 2021, [paper][S05] | Stem construction, optimization conclusion and stem-ablation appendix passages. |
| S06 | Yu et al., *MetaFormer Is Actually What You Need for Vision*, CVPR 2022, [paper][S06] | Mechanism/abstract: pooling token mixers motivate simpler representation controls. |
| S07 | Vaswani et al., *Attention Is All You Need*, NeurIPS 2017, [paper][S07] | Transformer attention, FFN, residual and position mechanism; original task is translation. |
| S08 | Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021, [paper][S08] | Patch-token mechanism and large-pretraining transfer boundary. |
| S09 | Tan and Le, *EfficientNetV2: Smaller Models and Faster Training*, ICML 2021, [paper][S09] | Architecture summary and inherited C1 dossier; efficient convolutional precedent. |
| S10 | Liu et al., *Swin Transformer*, ICCV 2021, [paper][S10] | Shifted-window hierarchy; mechanism plus inspected library partitions. |
| S11 | Liu et al., *Swin Transformer V2*, CVPR 2022, [paper][S11] | Normalization/relative-position mechanism; preserve original block semantics. |
| S12 | Tu et al., *MaxViT: Multi-Axis Vision Transformer*, ECCV 2022, [paper][S12] | Block/grid mechanism plus inherited dossier and library source. |
| S13 | Ho et al., *Axial Attention in Multidimensional Transformers*, 2019, [paper][S13] | Mechanism/abstract; excluded factorization alternative. |
| S14 | Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018, [paper][S14] | Mechanism/abstract; channel recalibration, not label-specific spatial readout. |
| S15 | Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV 2018, [paper][S15] | Mechanism/abstract; channel/spatial gating alternative. |
| S16 | Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017, [paper][S16] | Top-down/lateral mechanism and source detection comparison; exact library fusion inspected. |
| S17 | Xiong et al., *On Layer Normalization in the Transformer Architecture*, ICML 2020, [paper][S17] | Theory/abstract; pre/post-norm initialization behavior, with NLP transfer boundary. |
| S18 | Wu and He, *Group Normalization*, ECCV 2018, [paper][S18] | Mechanism/abstract; normalization independent of batch statistics. |
| S19 | He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016, [paper][S19] | Mechanism/abstract; residual-path rationale. |
| S20 | Huang et al., *Deep Networks with Stochastic Depth*, ECCV 2016, [paper][S20] | Mechanism/abstract; optional branch regularization. |
| S21 | Touvron et al., *Going Deeper with Image Transformers*, ICCV 2021, [paper][S21] | Mechanism/abstract; LayerScale as conditional deep-model support. |
| S22 | Hendrycks and Gimpel, *Gaussian Error Linear Units (GELUs)*, 2016, [paper][S22] | Mechanism/abstract; established FFN activation alternative. |
| S23 | Dao et al., *FlashAttention*, NeurIPS 2022, [paper][S23] | Exact attention with different IO/storage behavior; no local speed claim. |
| S24 | Cai et al., *EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction*, ICCV 2023, [paper][S24] | Full HTML mechanism, including the stated local-concentration limitation of ReLU linear attention. |
| S25 | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022, [paper][S25] | Mechanism/abstract; parameter-efficient alternative, original evidence is NLP. |
| S26 | Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017, [paper][S26] | Abstract-level calibration context; does not select a multi-label calibrator. |
| S27 | Jain and Wallace, *Attention is not Explanation*, NAACL 2019, [paper][S27] | Abstract-level NLP interpretation warning. |
| S28 | Wiegreffe and Pinter, *Attention is not not Explanation*, EMNLP 2019, [paper][S28] | Abstract-level counterargument; interpretation requires an explicit diagnostic definition. |
| S29 | Chefer et al., *Transformer Interpretability Beyond Attention Visualization*, CVPR 2021, [paper][S29] | Primary vision motivation/method summary; not a validated diagnostic for this custom network. |
| S30 | Chen et al., *Multi-Label Image Recognition With Graph Convolutional Networks*, CVPR 2019, [paper][S30] | Label graph/word-embedding mechanism; a distinct deferred intervention. |
| S31 | *CLARiTy*, 2025 preprint, [record][S31] | Abstract lead only: medical class tokens plus anatomical/segmentation priors; no component promoted. |
| S32 | *Information Bottleneck-based Causal Attention for Multi-label Medical Image Recognition*, MICCAI 2025 (author manuscript), [record][S32] | Abstract lead only: medical causal/objective changes; no causal ingredient claim or component promoted. |

### Retrieval and verification limitations

Older arXiv HTML routes for S01/S02 returned 404; their PDFs were available and
used. The WACV HTML route for S03 returned 403; the author manuscript and
official repository supplied the inspected evidence. The source has later
manuscript revisions, so its preprint date is not confused with the WACV 2023
publication. The brief's existing sources remain dated historical evidence.

The source-specific numerical experiments are not normalized into a league
table. No independent replication of the exact proposed Yummly mechanism was
found. Strong generic multi-label results and negative direct food evidence
therefore coexist in the handoff. The recent-work check is a bounded update,
not a claim to cover all research available by the cutoff.

[S01]: https://arxiv.org/pdf/2210.14147
[S02]: https://arxiv.org/pdf/2107.10834
[S03]: https://arxiv.org/pdf/2111.12933
[S04]: https://arxiv.org/pdf/2108.02456
[S05]: https://arxiv.org/pdf/2106.14881
[S06]: https://arxiv.org/abs/2111.11418
[S07]: https://arxiv.org/pdf/1706.03762
[S08]: https://arxiv.org/abs/2010.11929
[S09]: https://proceedings.mlr.press/v139/tan21a.html
[S10]: https://arxiv.org/abs/2103.14030
[S11]: https://arxiv.org/abs/2111.09883
[S12]: https://arxiv.org/abs/2204.01697
[S13]: https://arxiv.org/abs/1912.12180
[S14]: https://arxiv.org/abs/1709.01507
[S15]: https://arxiv.org/abs/1807.06521
[S16]: https://arxiv.org/pdf/1612.03144
[S17]: https://arxiv.org/abs/2002.04745
[S18]: https://arxiv.org/abs/1803.08494
[S19]: https://arxiv.org/abs/1512.03385
[S20]: https://arxiv.org/abs/1603.09382
[S21]: https://arxiv.org/abs/2103.17239
[S22]: https://arxiv.org/abs/1606.08415
[S23]: https://arxiv.org/abs/2205.14135
[S24]: https://arxiv.org/html/2205.14756v6
[S25]: https://arxiv.org/abs/2106.09685
[S26]: https://arxiv.org/abs/1706.04599
[S27]: https://arxiv.org/abs/1902.10186
[S28]: https://arxiv.org/abs/1908.04626
[S29]: https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf
[S30]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Multi-Label_Image_Recognition_With_Graph_Convolutional_Networks_CVPR_2019_paper.pdf
[S31]: https://arxiv.org/abs/2512.16700
[S32]: https://arxiv.org/abs/2508.08069

[I1]: #implementation-register
[I2]: #implementation-register
[I3]: #implementation-register
[I4]: #implementation-register
[I5]: #implementation-register
[I5/I6]: #implementation-register
