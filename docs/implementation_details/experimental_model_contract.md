# Experimental models and runtime

**Created:** 2026-10-06
**Last updated:** 2026-10-07

## Purpose and verified boundary

This document owns the implemented adapters, shared foundations and runtime for the [Phase 5 portfolio](../project_objective/experimental_model_portfolio.md). The [feature plan](../plans/additional_model_implementation.md) owns execution; the portfolio owns the binding architectures and 4A-D1/4A-D2 decisions. Measured resource qualification is a separate gate. No scientific selection rule, vocabulary membership, comparative augmentation policy or benchmark hyperparameter has changed.

Verified code consists of [`ExperimentalFitPad224`](../../src/data_processing/experimental_transforms.py), [`ExperimentalModelContract`](../../src/models/experimental_contract.py), deterministic head/offline helpers, [`ExactBatchPlan`](../../src/training/batching.py), the [experimental EfficientNet](../../src/models/experimental_efficientnet.py) and [MaxViT](../../src/models/experimental_maxvit.py) adapters, [`ExperimentalLGNM`](../../src/lightning/experimental_lgn.py), its [canonical construction boundary](../../src/training/experimental_runtime.py), the [DICA-Net-S tensor core](../../src/models/dica_net.py) and its [experiment adapter](../../src/models/experimental_dica_net.py). These are opt-in: full/default vocabulary and legacy/selector behavior remain unchanged. DICA-Net's capability-aware diagnostics and production inference callback are verified on synthetic CPU fixtures; measured CUDA caps and real-image/browser acceptance remain pending. The BaseLGNM guard prevents a 4A adapter from silently taking the legacy approximate-batch path.

The Phase 3 selector remains the separate 384-pixel, stock-dropout-retaining instrument in [`efficientnet.py`](../../src/models/efficientnet.py). Its trained state, source inventory and measured cap eight must not be reused as experimental initialization, modified, or interpreted as a cap for the new 224 protocols.

## Model and configuration identity

`ExperimentalModelContract` is immutable and emits only primitive configuration values. Schema version 1 records role `4a_experiment`, model and architecture identity, output count, raw-logit semantics, exact original weight enum, adaptation, complete transform specification and new-head initialization metadata. Supported identities and reserved adapter names are:

| Contract `model_id` | Original weight identity | Adapter and availability |
| --- | --- | --- |
| `efficientnet_v2_s` | `EfficientNet_V2_S_Weights.IMAGENET1K_V1` | Implemented `EfficientNetV2SExperiment`, `src/models/experimental_efficientnet.py`; CUDA qualification pending |
| `maxvit_t` | `MaxVit_T_Weights.IMAGENET1K_V1` | Implemented `MaxViTTExperiment`, `src/models/experimental_maxvit.py`; CUDA qualification pending |
| `p2_s` | `EfficientNet_V2_S_Weights.IMAGENET1K_V1` | Implemented `DICANetSExperiment`, reusing `DICANetSCore`/`DICAReadoutS`; capability-aware CPU diagnostics verified, CUDA/real-consumer qualification pending |

The user-approved DICA-Net naming amendment supersedes the unused reserved class/path `IngredientQueryP2S`/`ingredient_query.py`. No implementation or checkpoint ever used that class. The existing schema-1 `p2_s` identifier and architecture payload are retained exactly; it denotes DICA-Net-S, historically P2-S, rather than a second model.

Only `full` and `frozen_encoder` are accepted. Positive integer `num_classes` is not a hardcoded vocabulary: actual class order and the selected projection remain owned by the [P7 runtime contract](ingredient_selection.md#p7-runtime-projection). A model-width match alone is insufficient to accept an encoder or checkpoint. Canonical consumers validate saved P7 hashes, base indices and exact ordered classes.

`from_config()` reconstructs the contract and rejects missing, unknown, altered or inconsistent fields rather than silently accepting a different version, transform, weight enum or topology. The retained `p2_s` payload fixes S, taps 5/7, row-major 196+49 tokens, width 128, four heads, one block, ratio-four GELU FFN, normalization epsilon, bias policies, no added dropout/self-attention/positions/per-block memory normalization and fixed unit fusion coefficients. The tensor graph realizes this specification; the DICA-Net adapter validates it during construction and full/light checkpoint reconstruction. Its additional `encoder_provenance` pins the artifact and normalization policy without changing the existing shared schema.

The existing `ExpConfig` and `encode_config()`/`decode_config()` preserve the primitive dictionary under `hyper_parameters.torch_model.experimental_contract` (`tm_experimental_contract=contract.to_config()`). This round trip is tested. Do not serialize transform instances/closures: the generic encoder can omit callable objects. The existing `BaseModel._load_config()` whitelist does not restore this extra payload automatically; new adapters must implement both configuration directions and validate duplicated common fields rather than ignoring disagreements.

## Common full-frame 224 preprocessing

`ExperimentalFitPad224` subclasses `torchvision.transforms.v2.Transform`, so the existing DataModule accepts it as model-owned preprocessing and does not append dataset-statistics normalization. Its versioned identity is `4a-rgb-fit-pad-224-v1`:

1. Accept one PIL image or one CHW image tensor; convert PIL modes to RGB. One-channel tensors are repeated; four-channel tensors discard alpha, matching RGB conversion without alpha compositing.
2. Convert uint8 to FP32 by division by 255; floating inputs must be finite and within [0,1]. Batched images, unsupported channel counts/integer dtypes and empty spatial dimensions are rejected.
3. Resize the long side to 224 and the short side with integer round-half-up, clamped to at least one pixel. Use bilinear interpolation with antialiasing. There is no crop.
4. Center the resized image in a 224×224 canvas filled with RGB ImageNet mean `(0.485,0.456,0.406)`. An odd extra padding pixel goes right/bottom.
5. Optionally flip horizontally **after padding**. The probability is serialized; zero consumes no RNG and supplies deterministic validation/inference. Only horizontal flip is supported by this foundation, not an implicit arbitrary augmentation pipeline.
6. Normalize with the same mean and standard deviation `(0.229,0.224,0.225)`. Padding is exactly zero in normalized coordinates.

This order includes float conversion before resize. Geometry, interpolation, fill, normalization and operation order are all checked during configuration restoration. Training augmentation remains a declared configuration option; Phase 6 must freeze the comparative policy. Plain validation is always probability zero. The engineering smoke uses no augmentation and does not adopt that choice for the benchmark.

## New-head initialization

`make_experimental_linear(in_features,out_features,seed)` creates a biased CPU FP32 linear head, Xavier-uniform gain one and zero bias. A private CPU generator initializes the weights, and constructor RNG use is restored, so the helper leaves the global CPU RNG unchanged. Both established adapters must call this same policy after loading the intact pretrained encoder and replacing the entire stock readout; no stock classifier weight or dropout is retained in the new common readout.

DICA-Net's retained `p2_s` contract additionally records LayerNorm one/zero, independently drawn query/scale embeddings from `Normal(0,0.02)`, separate Q/K/V Xavier initialization and exact new-module/block initialization order. `DICAReadoutS` implements this new-module initializer and never visits the supplied encoder. Independent draw-order tests and actual pretrained-state preservation pass in 5.4.2; the established-model linear helper is not a complete custom initializer.

Seed defaults to 42 and is persisted. The same seed at different output counts does not prove paired or identical head rows: Xavier bounds and tensor shapes differ. Fresh selected-task training is not restoration of a sliced full-task model.

## Fresh construction versus offline restoration

`construct_experimental_model(factory,contract,state_dict=...)` separates original initialization identity from the operational construction flag. A fresh factory call receives `initialize_pretrained=True`; a nonempty restoration receives `False`, then loads the **complete** state with `strict=True`. The constructed model must expose the exact `experimental_contract`; missing/mismatched state or identity is rejected. A failed restoration never returns an apparently valid random model.

The original weight enum remains in the contract when operational download is disabled. Random tiny fixtures establish this helper behavior only; they are neither approved benchmark initializations nor the frozen-pretrained fallback. Exact original artifact URL, complete hash, package/source versions and notices must be verified with the actual adapters in 5.2/5.3/5.4/5.5.

`validate_checkpoint_contract()` checks a top-level `experimental_model_contract` payload against the requested contract. `ExperimentalLGNM` now writes and checks this key; full/light callback behavior is verified. Canonical training, dashboard checkpoint selection and best-trial restoration use `load_model_for_experiment()` for the new path. Legacy reconstruction retains its original initialization/loading behavior. Generic inherited `LightningModule.load_from_checkpoint()` is explicitly rejected for experimental models: use the complete saved experiment configuration and the maintained helper instead.

## Exact effective batching and tail weighting

The existing `BaseLGNM` rounding can exceed the request (for example, request 128 with cap 50 yields 43×3=129). It remains unchanged for serialized legacy runs and the frozen selector. New experimental adapters must use `ExactBatchPlan.resolve()` as an explicit opt-in instead:

- Choose the largest divisor of the requested size within the cap, or validate an explicitly supplied divisor. Request 128/cap 50 becomes 32×4=128; cap 24 becomes 16×8=128; physical 8 becomes 8×16=128.
- Save requested size, physical size, accumulation and cap in schema version 1. Restoration preserves the saved plan exactly; it does not silently recompute it against a new cap. Physical×accumulation must equal requested size.
- `finite_loader_consumed_records()` counts the actual finite loader horizon, including `limit_train_batches` (integer microbatch count versus floating fraction). `planned_optimizer_updates()` includes the last partial group. A runtime adapter must verify the actual loader/horizon assumptions, not blindly use dataset length.
- For mean-reduced loss and Lightning's fixed division by accumulation K, multiply the backward loss by `K*n_i/N_group`: microbatch records divided by records actually consumed in that accumulation group. Keep the logged per-sample loss unscaled. This corrects short microbatches, the final group and deliberately truncated smoke horizons without dropping records.

CPU analytic gradient/update tests cover 221 records with effective 128/physical 8/accumulation 16 (groups 128 and 93), a 16-microbatch horizon (128 records), and a 22-microbatch horizon (176 records, groups 128 and 48). The helper assumes a finite conventional single-device, fixed-size, non-dropping loader without custom sampling; distributed/sampled/early-stop horizons must not reuse that arithmetic without a verified extension. Accumulation reproduces sample-mean gradient weighting, **not** large-physical-batch BatchNorm statistics or identical stochastic model trajectories.

`ExperimentalLGNM` now consumes these helpers; its supported finite-loader horizon, update/record checks and CPU runtime evidence are described below. This does not change legacy `BaseLGNM` arithmetic or the selector's separate batching.

## Predeclared engineering smoke policy

`experimental_smoke_policy()` returns an independent primitive copy of `phase5-engineering-smoke-v1`. It is a resource-check policy, not a default benchmark/HPO configuration:

| Setting | Engineering declaration |
| --- | --- |
| Seed / requested effective batch | 42 / 128 |
| Physical probes / useful minimum | 128,64,32,16,8 / eight, fresh disposable state per probe |
| Numeric policy | `32-true`, matmul `highest`, CUDA matmul and cuDNN TF32 disabled |
| Memory allowance | 85% of device total; record both allocated and reserved peaks, device total/free memory and other use |
| Capacity / subsequent smoke | Two optimizer updates per probe; four for the smoke, two validation batches, at most one epoch |
| Loss / optimizer | Unweighted mean `BCEWithLogitsLoss`, no `pos_weight`; AdamW LR 1e-4, weight decay zero |
| Data/runtime | No augmentation; `drop_last=False`, two workers, portable `pin_memory=None`, one device |

The future launcher must set the actual microbatch horizon to the declared update budget, assert validation device before Lightning teardown, isolate output/scratch/cache, refuse overwrite and block test/predict image inference. A successful forward alone is not a capacity result. Full adaptation is tried first; only the adopted same-topology frozen/eval encoder fallback is permitted. The minimum batch is an engineering usefulness gate, not evidence that eight is universally adequate for fine-tuning; adaptation/normalization comparability still belongs to the later protocol decision.

**No CUDA probe, experimental pretrained construction or new training campaign has been run for 5.1. No cap is established for these models.** Phase 6 still owns comparative augmentation, loss, LR/decay, HPO, stopping and training horizon; the selector's settings do not transfer automatically.

## Verification and remaining integration gates

At Git base `d45b243` plus the new foundation sources on 2026-10-06:

```bash
python -m unittest discover -s tests -p 'test_experimental_*.py'
python -m unittest discover -s tests
```

Use the project's ML environment. **28 focused tests and all 177 repository tests pass.** Sources are [`test_experimental_model_contract.py`](../../tests/test_experimental_model_contract.py) and [`test_experimental_batching.py`](../../tests/test_experimental_batching.py). Focused checks are synthetic CPU/no-network; repository regressions include existing tiny CPU Lightning fits and one existing metadata-only real test-split vocabulary check, not predictive test evaluation or a CUDA qualification. No selector evidence, data/metadata, published projection or user-owned launcher/configuration was changed.

This is the retained 5.1 completion evidence. Subsequent 5.2 implementation evidence follows; actual approved-artifact verification, real dashboard diagnostics and measured CUDA execution remain 5.5 gates.

## Experimental EfficientNet and canonical runtime — 5.2

At base `4b00726` plus the 5.2 source changes on 2026-10-06, `EfficientNetV2SExperiment` constructs the intact 1000-way TorchVision model using the explicit approved enum before replacing its **whole** stock classifier with the seeded biased `Linear(1280,L)`. The original `features` and adaptive pool are retained, without a stock dropout or classifier weight. Input must be `[B,3,224,224]`; outputs are ordered raw `[B,L]` logits. Real TorchVision weights-none fixtures verify 20,177,488 feature parameters plus 1,281L head parameters: **20,388,853 at L=165**, and output widths 1/50/59/165. This proves structure/count, not the downloaded ImageNet artifact's provenance.

Full adaptation trains the features and head. Frozen mode fixes every feature parameter and holds the complete feature stack in eval across parent `.train()` calls; the head remains trainable. No unconditional `no_grad()` disables input-gradient diagnostics. The visual hook is the traversed last feature stage and the complete classifier hook is the new linear head. Hook/input-gradient tests pass; actual dashboard Grad-CAM/factorization execution is still a 5.5 acceptance gate.

Use the explicit `ExperimentalLGNM` class rather than the default legacy Lightning type. A configuration example, **not a launch or qualified batch recommendation**, is:

```python
contract = ExperimentalModelContract("efficientnet_v2_s", 165)
config = ExpConfig(hp_lgn_model_type=ExperimentalLGNM,
                   tm_type=EfficientNetV2SExperiment, tm_num_classes=165,
                   tm_experimental_contract=contract.to_config(), hp_batch_size=128)
```

For the selected task, construct the 59-output contract and explicitly add the published P7 projection; do not mutate/slice a full-task checkpoint. Numeric optimizer/loss/augmentation values in a new ExpConfig remain its existing defaults unless explicitly declared; this example does not freeze Phase 6 choices. Mean `BCEWithLogitsLoss`, weighted or unweighted, is the supported experimental accumulation loss. No protocol-specific physical cap exists yet; explicit `physical_batch_size`/`max_physical_batch_size` are configuration inputs, not automatically measured facts.

`prepare_experimental_config()` verifies a fitted strict `MultiLabelBinarizer`, class count, exact name-to-column mapping and projection identity before persisting encoder, `output_class_order` and `exact_batch_plan`. Explicit mismatches are rejected, not overwritten. Canonical training uses it; HPO's external trial configuration is enriched **before** saving, without running an HPO study. Variable-only HPO logging filters cannot remove the experimental restoration metadata. Default full output remains unchanged; selected order must match the published projection exactly and no row is dropped.

Experimental checkpoints retain four top-level identities even when light callbacks prune generic hyperparameters: `experimental_model_contract`, `experimental_batch_plan`, `experimental_output_class_order`, and encoded `experimental_training_config`. Full checkpoints also cross-check duplicate generic model/batch/loss/projection fields. Changes to output order, adaptation, transforms, batch plan, optimizer/LR/scheduler or other saved training settings are rejected. A light checkpoint needs its original external experiment configuration; its retained training identity is not a complete replacement DataModule configuration.

`load_model_for_experiment(config,checkpoint_path=...)` verifies protocol/nonempty saved state, constructs operationally offline and strictly restores **all** module state, including a coherent weighted-BCE `pos_weight` buffer. Arbitrary dropped fields, missing model parameters or incompatible loss buffers are rejected. Fresh construction still requests the original ImageNet enum; the operational false flag is never saved as scientific initialization. Saved positive weights are checked against the supplied DataModule at startup, including repeated startup. Optimizer/scheduler construction preserves their configured classes and operates on trainable parameters only.

Training validates one finite ordinary single-device DataLoader, fixed matching physical batch/accumulation, no custom/replacement sampler, no dropped rows and a verified integer/fractional batch limit. Returned backward loss is corrected; logged mean loss is not scaled. Actual optimizer updates **and consumed records** must match the planned horizon. Unsupported dynamic cuts/fast-dev-run or custom sampling fail closed. Resume is supported at epoch boundaries only: Lightning 2.6.1 may skip epoch-start hooks for a partial-epoch restore, so saved batch progress is checked at train start before any batch. Partial-epoch checkpoints remain readable for weight-only inference, not continuation of this training protocol.

Verification commands remain the two above. **57 focused tests and 206 repository tests pass.** Added evidence is [`test_experimental_efficientnet.py`](../../tests/test_experimental_efficientnet.py) and [`test_experimental_lightning.py`](../../tests/test_experimental_lightning.py): real weights-none architecture checks, synthetic full/frozen state and complete offline restoration, actual tiny CPU Lightning fits/checkpoint saves, full/light weighted/unweighted restores and epoch-boundary resume. Effective128/physical8/accumulation16 SGD matches direct sample-mean updates for 221 records (128+93) and a 22-microbatch horizon (128+48), with unscaled logged losses. Published 59-label order is checked without reading recipe images.

Repository regressions retain the existing metadata-only real test-split compatibility check; this is not predictive test evaluation. No pretrained experimental artifact was downloaded or qualified, no CUDA experimental step/physical-cap probe or benchmark/HPO campaign ran, and selector/data/projection artifacts plus the user's launcher/configuration remain unchanged. Real artifact hashes/notices, actual canonical GPU and dashboard/analysis acceptance, resource qualification and the later model adapters remain open; 5.2 completion must not be presented as Phase 5 readiness.

## Experimental MaxViT and artifact provenance — 5.3

At Git base `0aec11b` plus the 5.3 changes on 2026-10-06, `MaxViTTExperiment` builds the intact 1000-way TorchVision `maxvit_t` with `MaxVit_T_Weights.IMAGENET1K_V1`, then replaces the whole stock classifier: GAP, flatten, LayerNorm, biased 512-to-512 projection, Tanh and biasless 1000-class projection. The common readout is **GAP → flatten → seeded biased `Linear(512,L)`**. The stem, four blocks, internal attention/stochastic operations and original buffers are preserved. Actual module counts are **30,143,944 backbone parameters + 513L new parameters**, yielding **30,228,589 at L=165**.

Input is strictly `[B,3,224,224]`, B>0, with the same shared transform and head initialization as EfficientNet. The MaxViT-specific contract records input size 224, partition size seven, block grids 56/28/14/7 and the normalization policy. Construction validates those grids and the 22 partition-attention layers. A pre-adapter MaxViT payload without these fields is incomplete and rejected; the already implemented EfficientNet and planned P2 payloads are unchanged.

Full adaptation trains stem, blocks and the new readout. Frozen mode keeps both stem and blocks in eval across parent `.train()` calls, fixes their parameters and retains input-gradient diagnostics; the head trains. The final traversed block is the visual target `[B,512,7,7]`. The classifier/factorization target is the final linear layer, which scores both pooled image representations and two-dimensional concept vectors. The inherited factorization interface therefore exposes the complete scoring operation.

### Explicit normalization policy

Use the installed TorchVision constructor's **BatchNorm epsilon 1e-3 and momentum 0.01** during full adaptation. Preserve all pretrained running means, variances and counters when replacing the readout. Frozen adaptation uses their saved eval state. Constructor validation rejects a library realization with a different policy instead of resetting statistics or silently accepting drift.

The [versioned model documentation](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.maxvit_t.html) records **0.99 for historical pretraining**. The [pinned constructor source](https://github.com/pytorch/vision/blob/824e8c8726b65fd9d5abdc9702f81c2b0c4c0dc8/torchvision/models/maxvit.py) sets runtime momentum 0.01; loading the weights changes classes/input size and state, not momentum. The chosen policy retains this maintained runtime realization without outcome-based selection. Momentum is configuration rather than a state-dict buffer, so it is part of the validated experimental architecture payload. Accumulation preserves effective gradient weighting but does not reproduce BatchNorm statistics from a larger physical batch.

### Original artifact and notices

The official artifact was absent from cache, downloaded through the explicit enum URL and inspected on CPU using `weights_only=True`. Library hash-prefix verification was supplemented with a complete SHA-256 calculation:

| Provenance field | Verified value |
| --- | --- |
| Original enum/file | `MaxVit_T_Weights.IMAGENET1K_V1` / `maxvit_t-bc5ab103.pth` |
| Official URL | [TorchVision MaxViT-T artifact](https://download.pytorch.org/models/maxvit_t-bc5ab103.pth) |
| File size / SHA-256 | 124,538,661 bytes / `bc5ab103d47a7c6c02dc35bf65796b3a6cccf3d51bce330326dbdc634a3ac0e1` |
| Torch version / source revision | `2.8.0+cu129` / `a1cb3cc05d46d198467bebbb6e8fba50a325d4e7` |
| TorchVision version / source revision | `0.23.0+cu129` / `824e8c8726b65fd9d5abdc9702f81c2b0c4c0dc8` |
| Installed `torchvision/models/maxvit.py` SHA-256 | `73abc5ac42dd021446d573ab5e8c13249a76e4589f76abba4de2a02133b83ea6`; byte-matches the pinned upstream file |
| Installed TorchVision LICENSE SHA-256 | `6502f676851cfe25f8af75531dfb32375b7325b73c37e7b43741fa422893e71d` |

TorchVision's [BSD-3-Clause notice](https://github.com/pytorch/vision/blob/824e8c8726b65fd9d5abdc9702f81c2b0c4c0dc8/LICENSE) names Soumith Chintala, 2016, and requires preserving copyright, conditions and disclaimer on redistribution, including non-endorsement. The installed package retains that notice; this adapter imports library operators and copies no upstream model implementation. The [TorchVision README](https://github.com/pytorch/vision/blob/v0.23.0/README.md) and [model documentation](https://docs.pytorch.org/vision/0.23/models.html) distinguish source licensing from potentially dataset-derived pretrained-weight terms. The source license is not independent clearance of ImageNet-derived weights. No Google TensorFlow implementation or converted Google weights are used.

### Restoration and verification boundary

MaxViT uses the existing experimental Lightning and canonical helper without another training implementation. Full/light checkpoint identities include the explicit geometry and normalization policy, output order and projection, exact batch plan and encoded training configuration. Strict complete restoration retains every relative-position index buffer. A recursive adapter load hook also checks their **exact canonical values, dtype and shape**: ordinary `strict=True` alone would accept same-shaped corrupt indices. Missing state, malformed geometry, altered normalization or incompatible saved identities fail closed.

The original artifact contains 582 entries, including 22 relative-position buffers. A separate CPU check with the actual approved initialization compares all 577 non-classifier entries against the downloaded artifact after readout replacement; synthetic-image logits are finite and complete offline model restoration reproduces all state and logits exactly. This verifies original initialization and reconstruction. It establishes neither a physical GPU batch cap nor benchmark performance.

The verification commands above now pass **73 focused experimental tests and all 222 repository tests**. The 16 new tests are [`test_experimental_maxvit.py`](../../tests/test_experimental_maxvit.py) and [`test_experimental_maxvit_runtime.py`](../../tests/test_experimental_maxvit_runtime.py). Real TorchVision weights-none fixtures cover widths 1/50/59/165, exact counts, backbone-state retention, shared-head/transform identity, full/frozen gradients and normalization/stochastic buffers, actual forward hooks, strict primitive configurations and complete canonical offline full/light restoration. The production Grad-CAM and feature-factorization helpers execute on synthetic CPU images with a frozen real MaxViT graph. Full/light save-hook payloads are serialized and restored; these are not actual MaxViT `Trainer.fit` runs. Existing tiny CPU Lightning fits separately verify the shared runtime mechanics.

Repository regressions include the existing metadata-only real test-split vocabulary check, not predictive evaluation. No selector/data/projection evidence or user-owned training files changed. No MaxViT CUDA optimizer step, physical-cap probe, real-food dashboard acceptance, HPO or benchmark campaign ran. Those measured resource and real-consumer gates remain 5.5; the P2 adapter remains future 5.4 work. Phase 5.3 is complete at its implementation gate, not a declaration that Phase 5 is ready for comparative training.

## DICA-Net-S tensor core — 5.4.1

At base `557c3d9` plus the 5.4.1 changes on 2026-10-06, [`dica_net.py`](../../src/models/dica_net.py) implements the selected tensor graph. **DICA-Net** expands to *Dual-scale Ingredient-query and Context Attention Network*, as adopted in [4A-D2's naming amendment](../project_objective/experimental_model_portfolio.md#rationale-and-falsifiable-claim). P2-S is its historical research identifier. This implementation imports public PyTorch operators and follows the attributed [Query2Label-inspired route-Q equations](../research/topics/custom_attention_model_design/architecture_compatibility_synthesis.md#compatible-route-q-class-queries-with-a-pooled-context-path); it copies no research launcher or private helper.

`DICANetSCore(features, num_classes, head_seed=42)` accepts the intact eight-stage TorchVision EfficientNetV2-S `features` object, retains its parameters/buffers/mode and executes each stage exactly once. Input is floating `[B,3,224,224]`, B>0. The side outputs are `features[5]` F16 `[B,160,14,14]` and `features[7]` F32 `[B,1280,7,7]`. The core registers no stock pooling/classifier/dropout. It is an `nn.Module`, **not yet a canonical `BaseModel` experiment adapter**: it neither downloads/identifies encoder weights, freezes the encoder, supplies transforms nor persists an experiment configuration. The caller supplies the feature stack; no eight-stage shape check establishes its provenance.

`DICAReadoutS` applies independent bias-free 1×1 projections to width 128, row-major flattening, affine token LayerNorm (`eps=1e-5`) and distinct learned scale embeddings. Memory is `[B,245,128]`, with token ranges `[0,196)` for F16 and `[196,245)` for F32. One learned query per output row passes through one four-head block with width 32 per head, separate biased Q/K/V/output projections, pre-query LayerNorm, attention residual, pre-FFN LayerNorm and ratio-four exact GELU FFN residual. A final LayerNorm and independent biased class-wise dot product produce query logits. There is no normalization between labels or label-to-label attention.

The parallel context path averages the original F32 spatial grid and applies biased `Linear(1280,L)`. `forward()` returns only the fixed unit-weight sum of raw query and context logits. `forward_branches()` exposes the two raw tensors for engineering checks; they are not probabilities or a direct/contextual visibility decomposition. No new positions, spatial mixer, content mask, per-block memory normalization, label grouping/graph/text or added dropout is introduced. Original backbone operations remain intact.

The backend is public [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html), with `attn_mask=None`, `dropout_p=0.0` and `is_causal=False` in both train and eval. CPU FP32/FP64 outputs **and gradients** agree with an independent explicit softmax/einsum reference including projections, normalization and both residuals. Declared `rtol/atol` are `2e-4/3e-6` for FP32 and `1e-9/1e-11` for FP64. This does not verify a particular CUDA fused kernel or throughput.

New modules are initialized on CPU FP32 in the existing serialized order with a private seeded generator. Conv/linear/class-wise weights use Xavier gain one; biases zero; LayerNorm one/zero; query and scale embeddings independent normal with standard deviation 0.02. Q/K/V are distinct parameters initialized separately. No recursive initialization reaches the supplied backbone. Broader initializer, actual ImageNet artifact, frozen-state and complete checkpoint acceptance remain the next checkpoint.

Actual counts match the research arithmetic: **20,177,488 encoder + (383,616 + 1,538L) new parameters**. At 165 labels this is **20,814,874 total**, **637,386 new**, and **426,021 more** than the pooled experimental EfficientNet. Only label-indexed tensors change with L. The S core exposes no M/L or vocabulary-dependent width/depth option.

The 12 CPU/no-network tests in [`test_experimental_dica_net.py`](../../tests/test_experimental_dica_net.py) verify real TorchVision weights-none topology with B1 and L=1/50/59/165, stage traversal, feature shapes/order, affine normalization/scale ranges, raw fusion, finite forward/backward through both paths and the real encoder, reference attention, label-row permutation/subsetting, absent cross-label gradients, batch independence, and retained supplied state. The 59-row algebraic test uses the published P7 order/indices, reading only its registered resource. It jointly restricts learned queries, class-wise weights/biases and context rows while sharing other parameters, and matches projected full outputs at `rtol=2e-5, atol=3e-6`. This test-only state copy is not a subset-training or checkpoint-resume API; later selected-task training still starts a separate model.

Reproduce with `python -m unittest discover -s tests -p 'test_experimental_dica_net.py'` in the ML environment. All **234 repository tests pass**, including these 12. The full suite includes an existing metadata-only real test-split vocabulary check, not predictive evaluation. The new tests read no recipe metadata/images, download no pretrained artifact and perform no CUDA or optimizer step. 5.4.2 owns the approved initialization/adaptation/persistence adapter; 5.4.3 owns diagnostic capabilities and consumer integration; 5.5 owns measured qualification. The established adapters and serialized experimental contract are unchanged.

## DICA-Net-S initialization and persistence — 5.4.2

At base `9c76f86` plus the 5.4.2 changes on 2026-10-07, [`DICANetSExperiment`](../../src/models/experimental_dica_net.py) implements the `BaseModel` boundary around the unchanged S core. It constructs the intact original 1000-class `torchvision.models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)`, verifies the cached artifact's **complete SHA-256 and size**, then passes only its original `features` to the custom core. Stock classifier weights and dropout are discarded. A URL change, failed artifact hash or incompatible library readout/normalization causes an error; the adapter never substitutes a selector checkpoint or random initialization on failure.

The new-module initializer is verified against an independent sequence of draws for every parameter, including separate Q/K/V, scale/query normals, class-wise and context weights, zero biases and LayerNorm one/zero. Repeating a seed repeats the new state and leaves the caller's CPU RNG unchanged. Changing seed changes the new state; equal seeds at different L do not produce paired class rows. The actual approved encoder's **780 state entries** remain exactly equal to the original artifact after initialization, in both full and frozen adaptation.

`full` trains all **20,814,874** parameters at L=165. `frozen_encoder` fixes every encoder parameter and keeps its entire feature stack in eval across parent `.train()` calls, including all **110 BatchNorm modules** and stochastic depth; all **637,386** new parameters remain trainable. The native [TorchVision 0.23 source](https://github.com/pytorch/vision/blob/824e8c8726b65fd9d5abdc9702f81c2b0c4c0dc8/torchvision/models/efficientnet.py) uses BatchNorm **epsilon 1e-3/momentum 0.1**. This policy is validated and saved; loaded running statistics/counters are preserved, not reset. Gradient accumulation is not a change to physical-batch normalization. No encoder `no_grad()` blocks input gradients.

### Saved identity and offline restoration

The model has a strict nine-key configuration: the seven common `BaseModel` fields plus `experimental_contract` and `encoder_provenance`. The retained schema-1 `p2_s` contract owns topology/version, taps and token order, dimensions, zero-dropout/fusion boundaries, initialization seed/order, adaptation and exact common 224 transform. The additional provenance dictionary owns constructor, enum, URL, complete artifact hash/size and encoder-normalization policy. Missing, unknown, altered or inconsistent fields are rejected. The operational `initialize_pretrained` flag is never serialized as a new scientific initialization.

Fresh canonical construction uses the approved artifact; complete restoration constructs with `initialize_pretrained=False`, needs neither network nor original weight cache, then loads the entire saved state strictly. [`ExperimentalLGNM`](../../src/lightning/experimental_lgn.py) and [`load_model_for_experiment`](../../src/training/experimental_runtime.py) work unchanged. The encoded training identity retains the complete adapter configuration even in light checkpoints; full checkpoints also validate their duplicate model fields. Saved loss buffers, exact batching, ordered classes and the P7 projection remain mandatory. A full-task checkpoint cannot be resumed as a newly selected-task model, and missing query/scale/context weights or feature buffers are rejected.

The common model-owned transforms are preserved across configuration reconstruction. `conv_target_layer` is the traversed final feature stage; `classifier_target_layer` is the **complete two-input custom readout**, never just the context linear. Standalone-concept `factorization_classifier_layer` raises `NotImplementedError` because that operation is not valid for this classifier. At the 5.4.2 checkpoint this was a model-side boundary only; the [5.4.3 consumer integration](#dica-net-s-diagnostics-and-consumers--543) below closes the capability-aware dashboard gate.

### Verified original artifact and source

The original TorchVision artifact was already cached; no new download or Phase 3 trained state was used. The [official model entry](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.efficientnet_v2_s.html) and pinned source identify the enum and original ImageNet route. The local complete-file verification is:

| Field | Verified value |
| --- | --- |
| Original file / URL | `efficientnet_v2_s-dd5fe13b.pth` / [official artifact](https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth) |
| File size / complete SHA-256 | 86,721,253 bytes / `dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74` |
| State entries | 782 original entries, of which all 780 feature entries are retained exactly |
| Torch version / revision | `2.8.0+cu129` / `a1cb3cc05d46d198467bebbb6e8fba50a325d4e7` |
| TorchVision version / revision | `0.23.0+cu129` / `824e8c8726b65fd9d5abdc9702f81c2b0c4c0dc8` |
| Installed `torchvision/models/efficientnet.py` SHA-256 | `82d058e1989ef84e4ef7ce43e4a49dc10f1b95dcd003303c2c052438de823b17` |
| Installed TorchVision LICENSE SHA-256 | `6502f676851cfe25f8af75531dfb32375b7325b73c37e7b43741fa422893e71d` |

The installed package retains the [BSD-3-Clause notices](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE), including copyright, conditions, disclaimer and non-endorsement. The wrapper imports library operators and copies no research implementation. Source licensing does not independently establish permissions for ImageNet-derived weights; the [artifact/notice distinction already recorded for TorchVision](#original-artifact-and-notices) also applies here.

To reproduce the bounded original-state check, construct `DICANetSExperiment(adaptation=mode)` for each declared mode; the constructor verifies the original cache entry. Load that file with `torch.load(..., map_location="cpu", weights_only=True)`, compare every `features.*` key against `model.model.features.state_dict()` after stripping the original prefix, then reconstruct with `load_from_config(model.to_config(), initialize_pretrained=False)` and strict complete state loading. The recorded run uses one synthetic non-square image through `transform_plain`, verifies finite `[1,165]` outputs and **exact** offline CPU logits for both modes. This establishes initialization and restoration, not training quality or GPU capacity.

### Verification boundary

The final regression snapshot includes the independently committed launcher changes at `71b7bdc`; its ten launcher tests are separate from this checkpoint's fourteen additions. The combined DICA-Net core/adapter/runtime suite passes **26 tests**.

The new [adapter tests](../../tests/test_experimental_dica_net_adapter.py) and [canonical persistence tests](../../tests/test_experimental_dica_net_runtime.py) supply **14 passing tests**. They use real TorchVision weights-none graphs with synthetic inputs, plus explicit mocked fresh-factory routing; original pretrained evidence is the separate artifact check above. Tests verify initialization order/RNG, feature-state retention, full/frozen buffers and gradients, a bounded synthetic frozen-head SGD update, widths 1/50/59/165, strict configuration/provenance, full/light save callbacks, disk round trips and complete offline reconstruction. The matrix covers full/selected vocabularies, full/frozen adaptation and weighted/unweighted loss with exact restored state/logits. Network/cache access is forbidden in restoration tests. Complete save-hook payloads are exercised; these are not actual DICA-Net `Trainer.fit` runs. Existing tiny CPU Lightning tests verify the shared trainer mechanics.

Run `python -m unittest discover -s tests -p 'test_experimental_dica_net*.py'` for the custom model and `python -m unittest discover -s tests` for regressions. The recorded full-suite run passes **258 tests**. No new recipe metadata/images, CUDA optimizer step, physical-cap probe, HPO, benchmark campaign or predictive test evaluation is part of this checkpoint. The generic suite's existing real test-split vocabulary check is metadata-only. Actual DICA-Net diagnostic consumers remain 5.4.3 and measured useful execution remains 5.5; 5.4 is not yet complete.

## DICA-Net-S diagnostics and consumers — 5.4.3

At base `b29e4c5` plus the 5.4.3 changes on 2026-10-07, the [shared model interface](../../src/models/commons.py) exposes `supports_feature_factorization` and `feature_factorization_unavailable_reason`. The default retains existing models' factorization behavior; it does not newly qualify every legacy architecture. DICA-Net overrides the capability to false and explains why standalone concepts cannot be scored by its two-scale ingredient-query/context classifier. Its complete readout and explicit unsupported-operation property remain unchanged. These are implementation capabilities, not new trainable state or scientific configuration fields.

The [production dashboard callback](../../src/dashboards/dash/pages/model_visualization.py) checks that capability before accessing the factorization classifier. It still computes Grad-CAM and the ordered sigmoid prediction table, replaces the unsupported panel with an explanation and returns an informational completion notification. It does not display the context linear as the whole classifier or leave a stale previous model's factorization plot. Supported models still run factorization; unexpected failures retain the existing error/logging path rather than being mislabeled as unsupported.

[`prepare_visualization_image`](../../src/dashboards/runtime.py) executes the saved transform **once** and derives the display image by undoing its declared RGB normalization. It supports the versioned experimental fit/pad transform and legacy TorchVision v1/v2 compositions ending in `Normalize`. This fixes the old assumption that every transform exposes a sliceable `.transforms` list and prevents a second stochastic preprocessing draw from misaligning the overlay. Unknown normalization layouts or invalid/nonfinite RGB tensors fail explicitly. The input tensor is unchanged; only the display copy is clamped to [0,1]. No transform, metadata or vocabulary policy changed.

The [shared Grad-CAM helper](../../src/commons/visualizations.py) enables the required gradient path even inside an enclosing `no_grad` context and uses input gradients for a frozen encoder. Both diagnostic helpers release their hooks in `finally`, including on errors; they do not suppress unrelated `IndexError`s. This does not add support for PyTorch `inference_mode`. DICA-Net's hook remains the real final 7×7 feature stage, differentiated through the complete forward/readout. Its Grad-CAM is a target-logit sensitivity summary at that stage, not a separate attribution of both spatial scales, an attention map, ingredient visibility evidence or verified localization.

### Verification and remaining boundary

[`test_experimental_dica_net_diagnostics.py`](../../tests/test_experimental_dica_net_diagnostics.py) adds eight CPU/no-network tests. Real weights-none graphs verify full/frozen Grad-CAM, automatic/explicit targets, nonzero finite input/feature/readout gradients, preserved encoder buffers and exact forward logits. Production `make_inference` is exercised through its actual on-disk adapter cache after canonical offline light-checkpoint restoration for full165/full-adaptation and selected59/frozen-adaptation fixtures. It verifies class order, unchanged plain Lightning prediction tuples, complete prediction confidences for the displayed top20, informative factorization omission, and retained Grad-CAM. A real ResNet18 fixture verifies supported factorization and that unrelated failures still report errors. Further checks cover single-pass display geometry, RGB/grayscale/RGBA, v1/v2 legacy normalization and hook cleanup on failures.

The page's registration and experiment discovery are isolated in these tests: this is execution of the real callback functions, **not** HTTP/browser or real-food acceptance. The fixtures make no claims about learned attention/localization quality. Existing MaxViT factorization and all prior state/persistence tests remain regression gates. Measured GPU execution/capacity, actual canonical DICA-Net training and real consumer acceptance remain 5.5. No retraining, HPO, predictive test evaluation, selector evidence change or benchmark campaign is part of 5.4.3.

Using `/root/miniconda3/envs/wsl_image_pytorch/bin/python`, `-m unittest discover -s tests -p 'test_experimental_*.py'` passes **107 tests** and `-m unittest discover -s tests` passes **266**. Regressions include existing tiny synthetic CPU Lightning fits and the metadata-only real test-split vocabulary compatibility check, not predictive evaluation on test images. This closes 5.4's implementation/diagnostic gate; Phase 5 remains open for 5.5.
