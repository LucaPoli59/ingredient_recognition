# Experimental models and runtime

**Created:** 2026-10-06
**Last updated:** 2026-10-06

## Purpose and verified boundary

This document owns the implemented adapters, shared foundations and runtime for the [Phase 5 portfolio](../project_objective/experimental_model_portfolio.md). The [feature plan](../plans/additional_model_implementation.md) owns execution; the portfolio owns the binding architectures and 4A-D1/4A-D2 decisions. Measured resource qualification is a separate gate. No scientific selection rule, vocabulary membership, comparative augmentation policy or benchmark hyperparameter has changed.

Verified code consists of [`ExperimentalFitPad224`](../../src/data_processing/experimental_transforms.py), [`ExperimentalModelContract`](../../src/models/experimental_contract.py), deterministic head/offline helpers, [`ExactBatchPlan`](../../src/training/batching.py), the [experimental EfficientNet](../../src/models/experimental_efficientnet.py) and [MaxViT](../../src/models/experimental_maxvit.py) adapters, [`ExperimentalLGNM`](../../src/lightning/experimental_lgn.py) and its [canonical construction boundary](../../src/training/experimental_runtime.py). These are opt-in: full/default vocabulary and legacy/selector behavior remain unchanged. P2-S and measured CUDA caps remain pending. The BaseLGNM guard prevents a 4A adapter from silently taking the legacy approximate-batch path.

The Phase 3 selector remains the separate 384-pixel, stock-dropout-retaining instrument in [`efficientnet.py`](../../src/models/efficientnet.py). Its trained state, source inventory and measured cap eight must not be reused as experimental initialization, modified, or interpreted as a cap for the new 224 protocols.

## Model and configuration identity

`ExperimentalModelContract` is immutable and emits only primitive configuration values. Schema version 1 records role `4a_experiment`, model and architecture identity, output count, raw-logit semantics, exact original weight enum, adaptation, complete transform specification and new-head initialization metadata. Supported identities and reserved adapter names are:

| Contract `model_id` | Original weight identity | Adapter and availability |
| --- | --- | --- |
| `efficientnet_v2_s` | `EfficientNet_V2_S_Weights.IMAGENET1K_V1` | Implemented `EfficientNetV2SExperiment`, `src/models/experimental_efficientnet.py`; CUDA qualification pending |
| `maxvit_t` | `MaxVit_T_Weights.IMAGENET1K_V1` | Implemented `MaxViTTExperiment`, `src/models/experimental_maxvit.py`; CUDA qualification pending |
| `p2_s` | `EfficientNet_V2_S_Weights.IMAGENET1K_V1` | Planned `IngredientQueryP2S`, `src/models/ingredient_query.py` |

Only `full` and `frozen_encoder` are accepted. Positive integer `num_classes` is not a hardcoded vocabulary: actual class order and the selected projection remain owned by the [P7 runtime contract](ingredient_selection.md#p7-runtime-projection). A model-width match alone is insufficient to accept an encoder or checkpoint. Canonical consumers validate saved P7 hashes, base indices and exact ordered classes.

`from_config()` reconstructs the contract and rejects missing, unknown, altered or inconsistent fields rather than silently accepting a different version, transform, weight enum or topology. The P2 payload fixes S, taps 5/7, row-major 196+49 tokens, width 128, four heads, one block, ratio-four GELU FFN, normalization epsilon, bias policies, no added dropout/self-attention/positions/per-block memory normalization and fixed unit fusion coefficients. This is a serialized specification, not evidence of an implemented attention graph; 5.4 must verify its realization against 4A-D2.

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

P2's contract additionally records LayerNorm one/zero, independently drawn query/scale embeddings from `Normal(0,0.02)`, separate Q/K/V Xavier initialization and exact new-module/block initialization order. Those modules and their initializer are still future 5.4 work; the current linear helper is not a complete P2 initializer. Initializers must affect only newly added modules, never recursively reset the loaded trunk or its buffers.

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
