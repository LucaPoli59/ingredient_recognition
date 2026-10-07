# DICA-Net-S: ingredient queries at two scales

**Created:** 2026-10-07
**Last updated:** 2026-10-07

## Variant and scope

**Dual-scale Ingredient-query and Context Attention Network**, scale S, is the implemented historical P2-S proposal. Its serialized identity remains `p2_s`. The [binding portfolio](../project_objective/experimental_model_portfolio.md#4a-d2--custom-attention-topology) owns the scientific design; the [runtime owner](../implementation_details/experimental_model_contract.md) owns verified configuration, state and qualification. This is a custom readout over an existing pretrained encoder, not a new pretrained backbone or a demonstrated novelty claim.

## Tensor flow

The intact [EfficientNetV2-S representation](efficientnet_v2_s.md) runs once. Stage five provides F16 `[B,160,14,14]`; stage seven provides F32 `[B,1280,7,7]`. Separate bias-free 1×1 projections convert both to width 128. Row-major spatial flattening and affine token LayerNorm (`eps=1e-5`) produce 196 and 49 tokens. Distinct learned scale embeddings distinguish the two ranges; concatenation gives memory `[B,245,128]`.

One learned 128-dimensional query corresponds to each **saved output row**. A single pre-norm block applies cross-attention from these L queries to the shared image memory, then an independent query-wise feed-forward network with ratio four and GELU. Attention has four heads of width 32, separate biased Q/K/V/output projections and residual connections. Final LayerNorm and class-wise biased dot products produce query logits `[B,L]`.

In parallel, the original F32 grid is averaged and passed through a new biased `Linear(1280,L)`. Final logits are the unit-weight sum of query and context logits. Sigmoid is outside the model. These branches are not a probability mixture or a direct-versus-contextual visibility decomposition.

There is no query self-attention, label graph/text, label grouping, new positional encoding, content mask, spatial mixer or custom dropout. Queries interact with the same image memory but do not explicitly exchange representations with each other. The fixed token ordering preserves scale boundaries. The graph uses public PyTorch scaled-dot-product attention and the attributed [Query2Label-inspired](https://arxiv.org/abs/2107.10834) readout route; it does not reproduce the entire original Query2Label network.

## State and output semantics

Original ImageNet feature weights/buffers survive new-module initialization. New conv/linear weights use Xavier gain one, biases zero, LayerNorm one/zero and independent normal query/scale embeddings with standard deviation 0.02. Initialization order and seed are saved; the initializer never visits the encoder.

The encoder has 20,177,488 parameters. New modules add `383616 + 1538L`; at L=165 the total is 20,814,874, including 637,386 new parameters. The increment over the pooled EfficientNet experiment is 426,021. Only label-indexed tensors vary with vocabulary size; S remains width128/four-head/one-block. M/L are retained design alternatives, not implemented training options here.

Shared-row permutation/subsetting of a deterministic saved state only permutes/selects outputs when the shared weights stay identical. A freshly initialized 59-label training model is a separate artifact with new label-indexed parameters; it is not a resumed slice of a 165-label run. Its class order and base-column mapping come from P7, not an independently fitted split vocabulary.

## Adaptation, diagnostics and hypothesis

Full adaptation trains encoder and both readout paths. The same-S frozen alternative fixes the whole encoder in eval and trains all new modules. Native encoder BatchNorm remains epsilon 1e-3/momentum 0.1 with preserved initialization statistics. Accumulation does not enlarge its normalization batch.

Grad-CAM differentiates the actual final 7×7 feature stage through the complete readout, retaining input gradients in frozen mode. It is not a separate attribution of both scales. Standalone-concept factorization cannot supply the complete two-scale query input, so the dashboard explains its unavailability while preserving predictions/Grad-CAM. A context-only linear is not the complete classifier. Attention or sensitivity is not localization/visibility evidence.

The hypothesis is a useful ranking/resource trade-off over the same pooled encoder. Testing it requires matched adaptation and downstream policy; independent HPO does not guarantee a matched mechanism control. Added capacity, multiple scales and the context branch mean a gain would not isolate attention alone.

## Evidence and sources

[`dica_net.py`](../../src/models/dica_net.py) owns the tensor graph; [the adapter](../../src/models/experimental_dica_net.py) owns original initialization and configuration. [Tensor tests](../../tests/test_experimental_dica_net.py) independently verify attention outputs/gradients, both-path gradients and label-row invariants. [Diagnostic tests](../../tests/test_experimental_dica_net_diagnostics.py) verify real callback behavior. Equations, evidence and counterevidence remain in the [compatibility specification](../research/topics/custom_attention_model_design/architecture_compatibility_synthesis.md). No effectiveness or learned-localization claim follows from those tests.
