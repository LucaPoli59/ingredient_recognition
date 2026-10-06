# Additional-model implementation feature plan

**Created:** 2026-10-06
**Last updated:** 2026-10-06
**Linked macro-section and work packages:** [Phase 5](../general_plan.md#5-additional-model-implementation), 5.1–5.5

## Objective and boundary

Turn the adopted [4A-D1/4A-D2 portfolio](../project_objective/experimental_model_portfolio.md) into three reproducible implementations: **EfficientNetV2-S**, **MaxViT-T**, and **P2-S dual-scale ingredient-query readout with pooled context**. Each must work through the canonical data, Lightning, configuration, checkpoint, analysis and dashboard paths, and pass a bounded engineering run on the development GPU.

This is an implementation plan, not another model-selection study or a training campaign. The [completed 4A plan](experimental_model_research.md) and [custom research plan](custom_attention_model.md) own the retained research sequence; the portfolio owns architecture decisions. This plan owns execution and acceptance evidence. Planned behavior becomes a current implementation contract only after code and tests exist.

## Progress tracker

**Overall status:** In progress
**Current task:** 5.3 complete: experimental MaxViT, explicit normalization, artifact and offline restoration verified
**Next action:** Execute 5.4.1's P2-S tensor/attention implementation. Actual CUDA and real-consumer acceptance remain 5.5.

| # | Task | Status | Required result |
| --- | --- | --- | --- |
| 5.1 | Shared implementation contract and reusable foundations | **Done** | [Implemented contract](../implementation_details/experimental_model_contract.md), exact preprocessing/head/batch helpers; 28 focused and 177 repository tests pass. Adapter/consumer/CUDA gates remain in subsequent steps. |
| 5.2 | Experimental EfficientNetV2-S adapter | **Done** | Distinct 224-pixel pooled head, full/frozen state, exact Lightning and strict full/light offline restoration; 57 focused/206 repository tests. Actual qualification remains 5.5. |
| 5.3 | Experimental MaxViT-T adapter | **Done** | Intact backbone/common readout, explicit normalization/geometry, approved artifact and strict offline full/light persistence; CPU diagnostics verified, 73 focused/222 repository tests pass. Actual qualification remains 5.5. |
| 5.4.1 | P2-S tensor flow and attention readout | **Pending** | Once-only feature extraction, both logit paths and numerical/label-row invariants. |
| 5.4.2 | P2-S initialization, adaptation and persistence | **Pending** | Intact pretrained state, full/frozen behavior, complete config and checkpoint round trips. |
| 5.4.3 | Capability-aware diagnostics and runtime integration | **Pending** | Predictions and valid Grad-CAM; unsupported feature factorization handled explicitly without breaking legacy models. |
| 5.5 | Measured qualification and Phase 6 handoff | **Pending** | Per-model bounded CUDA/resource/restore/dashboard evidence, regression checks and maintained documentation. |

5.2 and 5.3 depend on 5.1; 5.4 reuses the EfficientNet foundations from 5.2. A model's 5.5 checks can run as soon as its implementation is ready; there is no need to postpone early failures until all three models exist. Only S is a required custom implementation/resource qualification. The larger custom presets remain retained design options, not mandatory campaigns.

## Inputs and authority

| Input | Role in this plan |
| --- | --- |
| [Problem definition](../project_objective/problem_definition.md) | Image-only, weakly supervised recipe-label prediction; no detection or direct-visibility claim. |
| [Benchmark decisions](../project_objective/benchmark_decisions.md) and [comparative methodology](../project_objective/model_comparison_methodology.md) | Frozen data/vocabulary, test isolation, Q1–Q4 boundaries and one declared seed per configuration. Later amendments and completed runtime records govern current execution rather than earlier historical pending statements. |
| [Experimental portfolio](../project_objective/experimental_model_portfolio.md) | Binding constructors, readout, input, custom topology, initialization, fallbacks and interpretation limits. |
| [Custom compatibility specification](../research/topics/custom_attention_model_design/architecture_compatibility_synthesis.md) and [topology handoff](../research/topics/custom_attention_model_design/topology_proposals.md#implementation-and-comparison-handoff) | Exact route-Q equations and implementation risks; retain their source/counterevidence links instead of repeating the literature review. |
| [Current models](../implementation_details/models.md), [data runtime](../implementation_details/image_data_loading.md), [P7 projection](../implementation_details/ingredient_selection.md#p7-runtime-projection) | Verified reusable APIs and compatibility requirements. Data 2.4 and P7 are complete; their evidence does not qualify a new architecture automatically. |
| [Artifact contract](../implementation_details/experiment_artifacts.md) and [comparison tooling](../implementation_details/experiment_comparison.md) | Saved-state and offline-analysis integration; historical comparison is not final benchmark evidence. |

The planning survey used Git base `5cdb080` and the current source on 2026-10-06. The user-owned one-shot training script and PyCharm configuration were left untouched. No model was constructed, weight downloaded, test run or training started for this planning step. The previously recorded 149-test result belongs to Data 2.4, not to these future implementations.

## Scope and non-goals

Included: model adapters, minimal reusable operators, exact preprocessing/configuration, safe full/frozen adaptation, existing vocabulary projection, persistence, diagnostic capability handling, thin rerunnable engineering launchers and bounded acceptance checks.

Excluded:

- Reopening the research shortlist, selecting architectures from validation scores, implementing reserve Swin or custom P1/P3, or promoting M/L because there is spare memory.
- Selector retraining, changes to the D4/D6 rules or published membership, or reuse of the trained selector as benchmark initialization.
- Full training, Optuna studies, repeated-seed campaigns, selected-versus-random controls, local hyperparameter adaptation, calibration or final test evaluation. These remain Phases 6–7.
- A generic model registry/framework rewrite, unrelated legacy repairs, dataset regeneration, manual image/ingredient review or historical artifact deletion.
- Implementing the entire final metric suite here. Existing `val_loss` monitoring can support a labelled engineering smoke; it does not settle the Phase 6 HPO objective.

Preparation may read canonical split metadata, which currently includes test/predict metadata. Engineering checks use synthetic fixtures and real **train/validation images only**, with explicit guards against test/predict loaders and image inference. Metadata-only access must be reported separately from predictive evaluation.

## 5.1 — Shared contract and foundations

Read the binding inputs and trace actual construction through [`BaseModel`](../../src/models/commons.py), [`ExpConfig`](../../src/commons/exp_config.py), [`BaseLGNM`](../../src/lightning/lgn_models.py), [`training/commons.py`](../../src/training/commons.py), checkpoint restoration and [`dashboards/runtime.py`](../../src/dashboards/runtime.py).

Resolve the small remaining implementation choices before building adapters:

1. Give the three experimental models stable, distinct identities and schema versions. Preserve `EfficientNetV2SSelector` as the separate 384-pixel, dropout-retaining Phase 3 instrument. Do not repurpose its class, cap, transform or hash-bound source inventory.
2. Define one model-owned **RGB → aspect-preserving fit → centered 224×224 padding → ImageNet normalization** realization. Record rounding, odd-padding placement, fill, interpolation/antialiasing, tensor conversion and ordering. Training augmentation stays configurable and serialized; Phase 6 freezes its comparative policy. A smoke augmentation is explicitly engineering-only. Test portrait, landscape, square, odd dimensions, RGB conversion and checkpoint/dashboard parity.
3. Define one identical new-linear-head initialization policy for the established pair. The custom initialization is already binding under 4A-D2 and cannot be replaced by a library default. Save seed and construction order; a shared seed alone does not imply paired head rows at different `L`.
4. Persist constructor/weight identity, architecture/transform versions, adaptation mode and all non-common model fields through both configuration directions. `BaseModel._load_config()` currently whitelists common keys; subclasses must explicitly reconstruct their own fields. Reject inconsistent or unsupported states rather than silently substituting defaults. Maintain serialized legacy behavior.
5. Specify offline reconstruction: saved model state must reload without downloading pretrained weights. Fresh benchmark initialization still uses the exact approved ImageNet artifact. A random-weight fixture is only an engineering test and is never a frozen-pretrained fallback or a benchmark configuration.
6. Declare the bounded smoke's requested batch, precision, optimizer/loss, update limit, memory allowance and useful execution criterion before outcomes. These are resource-check settings, not final HPO choices. Inspect the existing physical-batch/accumulation calculation and reject or explicitly resolve non-exact effective-batch requests; record actual optimizer updates and partial-batch handling.

Keep changes narrow. Reuse existing transformation and training facilities where compatible, but do not edit frozen selector sources merely to share a helper. New wrappers may reuse the same public library operators without sharing the selector's protocol.

**Completion:** The shared choices are recorded in the implementation owner with tests for introduced behavior; configuration identity and backward compatibility are explicit. No comparative augmentation/loss/HPO decision is smuggled into constructor defaults.

### 5.1 completion checkpoint — 2026-10-06

At base `d45b243`, added `src/data_processing/experimental_transforms.py`, `src/models/experimental_contract.py` and `src/training/batching.py`, with two focused test files. The [implementation owner](../implementation_details/experimental_model_contract.md) records exact transform geometry/order/fill, primitive identity/schema, stable adapter names, private FP32 Xavier head initialization, P2-only initialization metadata, operational fresh/offline factory semantics, strict identity guards and the predeclared engineering policy. Existing model/training interfaces and selector sources are unchanged.

Resolved exact effective batching using the largest fitting divisor, preserving the saved plan on restore. The tail-loss helper corrects sample-mean weighting against the actual consumed loader horizon, including integer/fractional limits; analytic tests cover full and final/truncated accumulation groups. This supplies a reusable foundation, **not** a claim that current `BaseLGNM` rounding or checkpoint/dashboard construction has changed. No model registry or architecture was implemented.

Verification: `python -m unittest discover -s tests -p 'test_experimental_*.py'` passes **28 tests**; `python -m unittest discover -s tests` passes **177** in the project ML environment. Focused tests are CPU/no-network and verify primitive `ExpConfig` persistence, strict synthetic offline restoration, top-level identity survival through the existing light callback, transform compatibility/no extra DataModule normalization, and batching arithmetic. Generic regressions include existing CPU miniature Lightning fits and metadata-only test-split vocabulary checking; no predictive test access, real pretrained experiment or CUDA qualification occurred.

5.1's completion gate is satisfied. The remaining canonical integration is explicitly assigned to adapters and 5.5: consume the extra config rather than the base whitelist, write/validate top-level checkpoint identity, provide actual offline construction in model/Lightning/dashboard paths, preserve P7 ordered output identity, integrate exact batching and verify optimizer updates. Actual weights, parameter counts, frozen state, diagnostic capabilities and useful GPU caps remain unmeasured. Next is 5.2; Phase 5 remains **In progress**, not implemented-ready.

## 5.2 — Experimental EfficientNetV2-S

Implement a distinct `BaseModel` adapter using `efficientnet_v2_s` and explicit `EfficientNet_V2_S_Weights.IMAGENET1K_V1`. Retain intact `features` and buffers; use **GAP → flatten → newly initialized biased `Linear(1280,L)`**, without stock classifier dropout or stock classifier weights. The input is the shared 224 policy, not the selector's 384 transform.

Support full adaptation and the same-checkpoint frozen-encoder alternative. Frozen mode holds the encoder in evaluation mode across parent `.train()` calls, including normalization buffers and stochastic layers, while the head trains. Ensure Grad-CAM can obtain input gradients: an unconditional encoder `no_grad()` shortcut must not disable diagnostic execution.

Verify raw `(B,L)` logits, expected trainable parameters, complete readout replacement, transform parity, full/frozen state behavior, standalone config reconstruction and canonical checkpoint restoration. At `L=165`, the research-derived parameter expectation is **20,388,853**; compare it with the actual module count and investigate differences, not adjust the topology to hide them.

**Completion:** A tested experimental wrapper exists without changing selector semantics or artifacts. Its remaining real resource/consumer checks are tracked in 5.5.

### 5.2 completion checkpoint — 2026-10-06

At base `4b00726`, implemented `EfficientNetV2SExperiment`, `ExperimentalLGNM` and `training/experimental_runtime.py`, with narrow canonical training, dashboard and best-trial construction/restore wiring. The [implementation owner](../implementation_details/experimental_model_contract.md#experimental-efficientnet-and-canonical-runtime--52) records exact fields, APIs and limits. Default full vocabulary and selector semantics remain unchanged; a 4A adapter cannot silently enter legacy approximate batching.

Real no-network TorchVision fixtures verify intact feature state, widths 1/50/59/165 and the expected **20,388,853 parameters at 165**, with the stock readout/dropout replaced completely. Full/frozen state, persistent feature eval, active hooks/input gradients and strict model configuration pass. Full/light checkpoints retain protocol, exact batch, ordered classes and encoded training identity; complete offline restoration includes coherent positive-loss buffers. Fitted full/default encoder column mapping and selected P7 order are checked explicitly. The HPO config-save/filter path preserves mandatory restoration fields; no study ran.

Actual tiny CPU Lightning fits verify sample-mean SGD parity for 221 records and a truncated 176-record horizon at effective 128/physical 8/accumulation 16, two updates per epoch and unscaled logging. Actual full/light checkpoint saves restore weighted/unweighted state and support epoch-boundary resume to step four. Partial-epoch training resume, unsupported loaders/cuts, changed mappings/configurations and incomplete states fail closed. This is engineering evidence, not convergence or ranking.

`python -m unittest discover -s tests -p 'test_experimental_*.py'` passes **57 tests**; `python -m unittest discover -s tests` passes **206**. Generic regressions include the existing metadata-only real test-split compatibility check, not predictive test evaluation. No experimental weight download, CUDA step/cap probe, real dashboard diagnostic qualification or benchmark campaign occurred. Approved artifact hashes/notices, measured capacity and actual real GPU/consumer acceptance remain 5.5. 5.2 is **Done** at its implementation gate; 5.3 is next and Phase 5 stays **In progress**.

## 5.3 — Experimental MaxViT-T

Implement the explicit TorchVision `maxvit_t` / `MaxVit_T_Weights.IMAGENET1K_V1` route. Retain the stem/blocks and their original buffers. Replace the **whole stock classifier**, including its intermediate normalization/projection/Tanh, with the common GAP/flatten/biased linear readout. Assert the approved 224 input and partition geometry rather than pretending arbitrary resolution is supported.

Use the same transform and head initialization policy as 5.2. Verify full and frozen state behavior. Inspect and record the pinned implementation's BatchNorm momentum behavior and choose an explicit fine-tuning normalization policy; do not silently reset pretrained running statistics. A physical-batch reduction is not equivalent to larger-batch BatchNorm via gradient accumulation.

Record the exact weight URL, full file hash, package/source version and applicable notices, and prove offline checkpoint reconstruction. The research-derived `L=165` count is **30,228,589**, subject to actual verification. Use hooks on modules traversed by the real forward path and verify established-model Grad-CAM/factorization compatibility rather than assuming it from the CNN interface.

**Completion:** The intended adapter and interface/persistence/state tests pass. A failure after the declared resource fallback reopens 4A-D1 with evidence; Swin is not substituted automatically.

### 5.3 completion checkpoint — 2026-10-06

At base `0aec11b`, implemented `MaxViTTExperiment` and the MaxViT-specific geometry/normalization payload in the existing experimental contract. The [implementation owner](../implementation_details/experimental_model_contract.md#experimental-maxvit-and-artifact-provenance--53) records APIs, provenance and limits. The complete stock readout is replaced by the shared GAP/flatten/new biased linear; actual L=165 count is **30,228,589**. Full/frozen behavior preserves both stem and blocks, with persistent eval and input gradients in frozen mode. Shared runtime is reused.

Adopted the native TorchVision runtime **BatchNorm epsilon 1e-3/momentum 0.01**, preserving loaded statistics; the documented historical pretraining momentum 0.99 stays explicit provenance. Geometry and normalization are saved and validated. Missing, wrong-shaped, wrong-dtype or same-shaped corrupt relative-position indices are rejected through recursive restoration. Existing EfficientNet/P2 payloads and selector behavior are unchanged.

The exact approved artifact was downloaded and its complete SHA-256, file size, package/source revisions and notices recorded. The actual pretrained adapter retains **all 577 original non-classifier entries**, including 22 relative-position buffers and 34 BatchNorm modules; synthetic CPU outputs are finite and complete offline reconstruction preserves state/logits exactly. No-network fixtures verify interface, state, saved class/projection identity, full/light canonical restoration and production Grad-CAM/factorization helpers. These engineering results do not establish a measured physical GPU cap or model effectiveness.

The new source/tests are linked in the implementation owner. `python -m unittest discover -s tests -p 'test_experimental_*.py'` passes **73 tests**; `python -m unittest discover -s tests` passes **222**. Existing regressions include the metadata-only real test-split vocabulary check, not predictive test evaluation. Actual CUDA/resource and real dashboard acceptance remain 5.5; P2 remains pending. 5.3 is **Done** at its implementation gate, Phase 5 stays **In progress**, and 5.4.1 is next.

## 5.4 — P2-S custom implementation

The [binding construction](../project_objective/experimental_model_portfolio.md#4a-d2--custom-attention-topology) and [route-Q equations](../research/topics/custom_attention_model_design/architecture_compatibility_synthesis.md#compatible-route-q-class-queries-with-a-pooled-context-path) remain authoritative. The following checkpoints translate them into code/tests rather than redesigning the network.

### 5.4.1 — Tensor flow and numerical attention

Execute the intact EfficientNetV2-S features **once**, capturing `features[5]` as `(B,160,14,14)` and `features[7]` as `(B,1280,7,7)` at 224 input. Project separately with bias-free 1×1 convolutions, flatten row-major, apply affine token LayerNorm (`eps=1e-5`) and learned scale embeddings, then concatenate **196+49=245** tokens.

Implement S with `D=128`, four heads and one pre-norm cross-attention/ratio-four GELU FFN block. Use one learned query per ordered class, distinct biased Q/K/V/output projections, the specified residual order, final LayerNorm and class-wise biased readout. Add the parallel F32 GAP/biased `Linear(1280,L)` logits with fixed coefficients one. Return raw logits only.

Do not add query self-attention, per-block memory normalization, positions, a spatial mixer, content mask, label grouping/graph/text, sigmoid fusion or custom-module dropout. Preserve original backbone operations. S/M/L tuples remain the recorded scaling rule, not a tuning dimension; larger-size runtime qualification is outside this required work.

Test B1 and `L=1/50/165`, plus the actual 59-output projection. `L=50` is a shape fixture, not a replacement vocabulary. Cover once-only traversal, both-branch gradients, finite forward/backward, token ordering, reference softmax attention versus the actual backend within declared tolerances, and label-row permutation/subsetting with shared weights. The latter must only permute/restrict deterministic outputs; it is not selected-task training.

### 5.4.2 — Initialization, state and persistence

Initialize only new modules: Xavier-uniform gain one for conv/linear weights, zero biases; LayerNorm one/zero; independent normal query/scale embeddings with `std=0.02`. Initialize Q/K/V separately even if the backend later packs them. Prove pretrained feature weights/buffers are unchanged by head initialization and pin the original ImageNet artifact, not a Phase 3 trained checkpoint.

Verify full adaptation and same-S frozen/eval encoder with all new modules trainable. Save topology version, taps/token order, scale and dimensions, initialization, zero-dropout boundary, transform, weight provenance, adaptation and ordered output identity through configuration and full/light checkpoint paths actually supported by the repository. Reject incompatible custom fields, class counts, projection hashes or restored topology.

The research-derived S count at `L=165` is **20,814,874**, including **637,386 new parameters**; its increment over the pooled EfficientNet model is **426,021**, not 637,386. Measure counts separately from memory/timing.

Full-task predictions projected to 59 labels and a freshly initialized 59-label model are separate artifacts. New selected-task training later uses transferred full-task hyperparameters; it is not a resumed sliced checkpoint. Do not change head width, taps or block depth with vocabulary size.

### 5.4.3 — Diagnostic capabilities and consumer integration

Provide valid forward-path Grad-CAM hooks with input gradients in full/frozen modes. The existing standalone-concept feature factorization is not directly compatible with the query-conditioned complete classifier. The minimal supported path is an explicit capability and a dashboard that skips the unsupported panel with a clear explanation while predictions and supported diagnostics remain usable.

Do not expose the context-only linear layer as the complete model classifier. A qualified alternative diagnostic adapter requires its own documented interpretation and tests; it is not mandatory. Preserve factorization for existing supported models and reject/handle unsupported capabilities rather than crashing the whole prediction page.

Optional attention maps must preserve scale boundaries and carry the warning that attention is not ingredient localization or evidence of visibility. No map/annotation is a vocabulary gate. Verify the actual dashboard callback path, not just model properties.

**5.4 completion:** All three custom checkpoints pass their unit/integration gates; S instantiates the adopted topology and preserves pretrained/label semantics. Actual resource and bounded real-runtime acceptance remains 5.5.

## 5.5 — Measured qualification and handoff

Use one reusable, thin engineering CLI/configuration path for the new portfolio, built on [`src/training`](../../src/training/commons.py). The [Data smoke](../../scripts/validation/data_runtime_smoke.py) supplies reusable isolation and restore/dashboard checks; extend or factor it only when doing so preserves its maintained behavior. Do not duplicate a trainer in each family launcher or edit the user's one-shot script to run acceptance checks.

For each required model:

1. Verify the actual approved weight artifact, source/version/notices and offline model/experiment reconstruction. Record Git revision plus the exact relevant source inventory if local changes exist, model/transform/config identity, data generation, class order and projection identity.
2. Probe a **complete training step** on the declared device, including loss, backward, optimizer-state allocation and any logging/validation workload. Use fresh disposable state, synchronized timing and reset peak counters. Record allocated/reserved peaks, physical batch, precision, trainability, workers/pinned memory and allowance. Assert the device for validation resource checks explicitly: Lightning teardown can move the module to CPU, so a post-fit CPU check cannot certify CUDA inference. A forward-only test or parameter estimate is insufficient.
3. Start from the declared requested batch and reduce physical batch intelligently with explicit accumulation if needed. Validate exact effective size/optimizer updates; do not silently round it up. Record BatchNorm implications and precision/TF32 settings. Establish a protocol-specific measured cap only from evidence: the selector's physical cap eight at 384 does **not** establish any new 224-pixel cap.
4. Run the bounded real train/validation smoke through canonical construction, fit and checkpoint APIs. Assert finite loss/gradients, a trainable-parameter update, output order, transform parity, optimizer updates and exact/allclose restored logits under a declared evaluation tolerance. Verify full/default and selected opt-in configuration/restore/projection paths with synthetic fixtures; include a bounded selected-runtime check, retaining all-zero targets and every original record.
5. Exercise actual dashboard prediction/diagnostic behavior and offline analysis identity. P2 factorization may be explicitly unsupported; predictions and supported Grad-CAM must still work. Keep full/selected outputs and adaptation cohorts identifiable, with no predictive test access.
6. Run focused tests and proportionate repository regressions after shared API changes. Preserve immutable metadata, the published projection and retained historical/config/checkpoint evidence; use existing retention checks when a touched path threatens compatibility. Record reproducible commands and pass/failure evidence, not merely a checked box.

Use unique output directories under `experiments/runtime_smoke/`; refuse accidental overwrite and isolate trainer scratch/dashboard cache. Disposable capacity probes are not benchmark initializations. A useful minimum physical batch and memory allowance are declared in 5.1; a technically successful but scientifically impractical batch is not an automatic full-adaptation pass.

First try full adaptation. If it remains infeasible after the declared batch/resource checks, test the adopted **same checkpoint/topology with a frozen/eval encoder** fallback only. Document its changed adaptation semantics. The established pair must later use a matched adaptation policy or explicitly compare different model protocols. Custom-versus-pooled mechanism claims require matched adaptation too. If the permitted fallback is not useful, stop and reopen the portfolio decision with failure evidence; do not silently shrink/change the network or promote a reserve.

The accepted result is a measured engineering capability, not convergence or predictive superiority. No long overfit campaign, HPO or cross-seed repetition is required to close Phase 5.

## Affected components and durable outputs

| Area | Expected change or artifact |
| --- | --- |
| `src/models/`, model transforms | Shared helpers, `EfficientNetV2SExperiment` and `MaxViTTExperiment` exist. The planned P2 name and implemented adapter/runtime are fixed in the [contract owner](../implementation_details/experimental_model_contract.md); P2/resource checks are pending and selector provenance stays separate. |
| Configuration/checkpoint/training | Minimal extensions for custom fields, offline rebuild, adaptation and measured batch policy; reuse current output/vocabulary guards. |
| Dashboard/analysis consumers | Model-aware transforms, real hooks, explicit unsupported diagnostic handling and preserved output/projection identity. |
| `scripts/validation/`, `tests/` | Rerunnable bounded acceptance command(s), synthetic no-network tests and isolated resource/runtime evidence. No new HPO study. |
| [Model implementation owner](../implementation_details/models.md) | Current verified contracts and per-model acceptance evidence, updated with code; split a focused owner only if the custom scope needs it, updating the index. |
| `docs/models_deepdive/` | Architecture explanations tied to the actual implemented variants, using the retained research sources; not a second research/decision record. |
| Feature/general plan and repository knowledge | Completed-step evidence here; material first-level transitions in the general plan; stable model/API entry points in repository knowledge. |

Raw measurements/configurations/checkpoints stay in isolated experiment outputs, not `docs/`. Any future empirical benchmark result belongs to `experiment_results/`; a smoke must not be presented there as a model ranking.

## Completion gate and Phase 6 boundary

Phase 5 is **Done** only when all three required implementations pass their interface, state, persistence, vocabulary, diagnostic-capability and measured useful-execution gates; the bounded canonical runtime checks are reproducible; shared regressions/affected legacy compatibility pass; and current documentation is synchronized. Any adopted resource fallback is explicit and its comparison consequences are handed off. A model with an unresolved failed gate is not marked implemented-ready simply because another model passed.

Hand Phase 6 a compact readiness matrix for the three protocols: exact initialized artifacts, readouts/transforms, supported full/frozen mode, output/projection contracts, feasible measured physical batch/precision and accumulation limits, source/config/checkpoint provenance, supported diagnostics and open comparison constraints. Keep research and measured facts distinguishable.

Phase 6 still must freeze augmentation, loss/weighting, HPO objective/spaces/budgets, stopping, training horizon, one-seed reporting and adaptation comparability, plus random-control/local-panel decisions. Phase 7 still owns the final metric/calibration/uncertainty suite and test evaluation. Neither the selector's 40 epochs/effective batch 128 nor a short smoke's loss/monitor automatically becomes that benchmark policy. Completing this plan releases model engineering readiness; it does not launch a campaign or bypass remaining evaluation/protocol gates.

## Decision and change log

| Date | Change | Consequence |
| --- | --- | --- |
| 2026-10-06 | Created the Phase 5 plan after Data 2.4/P7 completion, using a read-only code/design survey at base `5cdb080`. | Five first-level work packages, with three bounded custom checkpoints; implementation remains Pending. No research decision, selector artifact, source behavior or training campaign changed. |
| 2026-10-06 | Completed 5.1 at base `d45b243` with opt-in preprocessing/identity/initialization/offline/batch helpers and declared smoke policy. | 28 focused and 177 repository tests pass; Phase 5 In progress, 5.2 next. Actual adapters, canonical integration and CUDA qualification remain future gates; no campaign or scientific policy change. |
| 2026-10-06 | Completed 5.2 at base `4b00726` with the separate EfficientNet adapter, exact experimental Lightning and strict offline/full-light persistence. | 57 focused/206 repository tests pass; 5.3 next. Actual artifact/real-consumer/CUDA gates remain 5.5; no selector/vocabulary change or campaign. |
| 2026-10-06 | Completed 5.3 at base `0aec11b` with MaxViT's whole-readout replacement, explicit native normalization/geometry, approved artifact verification and strict offline persistence. | 73 focused/222 repository tests pass, covering CPU state, class identity and diagnostics; 5.4.1 next. Actual CUDA/real-consumer acceptance remains 5.5. |
