# Experimental model foundations

**Created:** 2026-10-06
**Last updated:** 2026-10-06

## Purpose and verified boundary

This document owns the implemented shared foundations for the [Phase 5 portfolio](../project_objective/experimental_model_portfolio.md), not the architectures or their resource qualification. The [feature plan](../plans/additional_model_implementation.md) owns execution; the portfolio continues to own 4A-D1/4A-D2. No scientific selection rule, vocabulary membership, comparative augmentation policy or benchmark hyperparameter has changed.

Verified code consists of [`ExperimentalFitPad224`](../../src/data_processing/experimental_transforms.py), [`ExperimentalModelContract`](../../src/models/experimental_contract.py), a deterministic linear-head initializer, a construction/identity-check helper and [`ExactBatchPlan`](../../src/training/batching.py). These are opt-in foundations. The experimental adapters, their canonical Lightning/checkpoint/dashboard wiring and measured CUDA caps do **not** yet exist. Existing `BaseModel`, `BaseLGNM`, default configurations and selector sources are unchanged.

The Phase 3 selector remains the separate 384-pixel, stock-dropout-retaining instrument in [`efficientnet.py`](../../src/models/efficientnet.py). Its trained state, source inventory and measured cap eight must not be reused as experimental initialization, modified, or interpreted as a cap for the new 224 protocols.

## Model and configuration identity

`ExperimentalModelContract` is immutable and emits only primitive configuration values. Schema version 1 records role `4a_experiment`, model and architecture identity, output count, raw-logit semantics, exact original weight enum, adaptation, complete transform specification and new-head initialization metadata. Supported identities and reserved adapter names are:

| Contract `model_id` | Original weight identity | Planned adapter and module, not available yet |
| --- | --- | --- |
| `efficientnet_v2_s` | `EfficientNet_V2_S_Weights.IMAGENET1K_V1` | `EfficientNetV2SExperiment`, `src/models/experimental_efficientnet.py` |
| `maxvit_t` | `MaxVit_T_Weights.IMAGENET1K_V1` | `MaxViTTExperiment`, `src/models/experimental_maxvit.py` |
| `p2_s` | `EfficientNet_V2_S_Weights.IMAGENET1K_V1` | `IngredientQueryP2S`, `src/models/ingredient_query.py` |

Only `full` and `frozen_encoder` are accepted. Positive integer `num_classes` is not a hardcoded vocabulary: actual class order and the selected projection remain owned by the [P7 runtime contract](ingredient_selection.md#p7-runtime-projection). A model-width match alone is insufficient to accept an encoder or checkpoint. Saved P7 hashes, base indices and exact ordered classes must still be checked by the eventual consumers.

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

`validate_checkpoint_contract()` checks a top-level `experimental_model_contract` payload against the requested contract. The existing light-checkpoint callback leaves this top-level field intact while pruning hyperparameter sections; a unit test verifies that property. The future experimental checkpoint hooks must actually write/check the key, including before shape-compatible state restoration. This helper does not change legacy checkpoint acceptance or currently repair canonical offline reconstruction: `BaseLGNM.load_from_config()` and dashboard reconstruction still need the new adapters' operational restore path.

## Exact effective batching and tail weighting

The existing `BaseLGNM` rounding can exceed the request (for example, request 128 with cap 50 yields 43×3=129). It remains unchanged for serialized legacy runs and the frozen selector. New experimental adapters must use `ExactBatchPlan.resolve()` as an explicit opt-in instead:

- Choose the largest divisor of the requested size within the cap, or validate an explicitly supplied divisor. Request 128/cap 50 becomes 32×4=128; cap 24 becomes 16×8=128; physical 8 becomes 8×16=128.
- Save requested size, physical size, accumulation and cap in schema version 1. Restoration preserves the saved plan exactly; it does not silently recompute it against a new cap. Physical×accumulation must equal requested size.
- `finite_loader_consumed_records()` counts the actual finite loader horizon, including `limit_train_batches` (integer microbatch count versus floating fraction). `planned_optimizer_updates()` includes the last partial group. A runtime adapter must verify the actual loader/horizon assumptions, not blindly use dataset length.
- For mean-reduced loss and Lightning's fixed division by accumulation K, multiply the backward loss by `K*n_i/N_group`: microbatch records divided by records actually consumed in that accumulation group. Keep the logged per-sample loss unscaled. This corrects short microbatches, the final group and deliberately truncated smoke horizons without dropping records.

CPU analytic gradient/update tests cover 221 records with effective 128/physical 8/accumulation 16 (groups 128 and 93), a 16-microbatch horizon (128 records), and a 22-microbatch horizon (176 records, groups 128 and 48). The helper assumes a finite conventional single-device, fixed-size, non-dropping loader without custom sampling; distributed/sampled/early-stop horizons must not reuse that arithmetic without a verified extension. Accumulation reproduces sample-mean gradient weighting, **not** large-physical-batch BatchNorm statistics or identical stochastic model trajectories.

No current training path consumes these helpers yet. Experimental Lightning wiring must verify effective-plan persistence, actual optimizer update counts, supported horizons and unchanged logged-loss semantics in 5.2/5.5.

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

5.1 is complete at the foundation boundary. Next, 5.2 must implement `EfficientNetV2SExperiment`, extend its actual model configuration and experimental Lightning/checkpoint/restore path, and consume the exact-batch helper. Later steps must verify real full/frozen encoder state, intact weights, output/projection identity, actual dashboard/analysis restoration, diagnostics and measured CUDA execution. Those gates remain open and cannot be inferred from these fixture tests.
