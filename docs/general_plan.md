# General project plan

**Created:** 2026-08-02  
**Last updated:** 2026-10-07
**Overall status:** In progress  
**Current macro-phase:** Additional model implementation — custom DICA-Net integration in progress
**Current focus:** Work packages 5.1–5.3 are complete. Work package 5.4 now implements DICA-Net-S (historical P2-S): its tensor/attention graph is verified; approved initialization, adaptation and persistence are next. Actual CUDA/real-consumer acceptance remains 5.5; no campaign has started. Completed Data/selection, full default, projection and historical evidence stay unchanged.

## Purpose

This document is the permanent progress tracker for the entire thesis project. It covers the complete path from problem definition and data preparation through ingredient selection, model research, implementation, training, hyperparameter tuning, result comparison, and thesis writing.

This tracker must preserve project history. Completed work remains recorded when the project moves to a later phase. Status transitions are appended to the history log rather than reconstructed from memory or removed from the document.

The binding research and benchmark policies remain in [`project_objective/benchmark_decisions.md`](project_objective/benchmark_decisions.md). This tracker records their execution status without redefining them.

## Status definitions

| Status | Meaning |
| --- | --- |
| **Done** | The work package's completion gate has been met and supporting evidence exists. |
| **In progress** | Work has started and at least one required task remains. |
| **Pending** | The work is expected to start, but substantive execution has not begun. |
| **Deferred** | The work is intentionally postponed until a named dependency or gate is satisfied. |
| **Blocked** | Progress currently requires unavailable information or an external change. |
| **Superseded** | The output is retained as project history but has been replaced by a newer approach or artifact. |

A macro-section may remain **In progress** while some of its work packages are **Done**. Mark a macro-section **Done** only when its completion gate is satisfied.

## Project overview

| # | Macro-section | Status | Current outcome or next action |
| --- | --- | --- | --- |
| 1 | Project foundation | **Done** | Maintain the objective and documentation when decisions change. |
| 2 | Data | **Done** | Historical compatibility and all data gates pass, including bounded real CUDA training, exact checkpoint reload and dashboard verification for 2.4. |
| 3 | Ingredient selection | **Done** | D6/P6 shared vocabulary and P7 opt-in runtime integration are complete, with original D4 evidence, default full task and all records preserved. Parity/retention checks pass; historical code is retired from active use but retained. Mandatory P5/3.4 remains superseded. |
| 4 | Model research | **Done** | 4A selected two established families and one custom topology; 4B froze the independent 4B-D1 EfficientNetV2-S reference-selector protocol and released Phase 3. |
| 5 | Additional model implementation | **In progress** | [Foundations, established pair and DICA-Net-S tensor core](implementation_details/experimental_model_contract.md) are implemented. Continue 5.4's custom initialization/persistence/integration; remaining artifact/real-consumer/CUDA qualification stays open. |
| 6 | Training and hyperparameter tuning | **Deferred** | Resume after the benchmark, selected ingredients, and model contracts are frozen. |
| 7 | Results comparison | **Deferred** | Work package 7.1 tooling is done and the historical basic_v5 ResNet/DINOv2 validation comparison is retained; final comparisons resume after comparable benchmark runs are complete. |
| 8 | Thesis writing | **Pending** | Define the thesis outline and map project evidence to chapters. |

## 1. Project foundation

**Status:** Done

### Work packages

| Work package | Status | Evidence |
| --- | --- | --- |
| Repository and pipeline understanding | **Done** | [`../README_PROJECT_KNOWLEDGE.md`](../README_PROJECT_KNOWLEDGE.md) |
| Documentation conventions and structure | **Done** | [`README.md`](README.md), [`README_DOCS_ORGN.md`](README_DOCS_ORGN.md) |
| Research folder structure | **Done** | [`research/README.md`](research/README.md) |
| Problem definition | **Done** | [`project_objective/problem_definition.md`](project_objective/problem_definition.md) |
| Benchmark decisions | **Done** | [`project_objective/benchmark_decisions.md`](project_objective/benchmark_decisions.md) |
| Project progress tracker | **Done** | This document |

### Completed

- [x] Documented the repository architecture, training path, available models, dashboards, and operating environment.
- [x] Defined English documentation conventions, dating rules, and directory responsibilities.
- [x] Created structures for dated discovery and focused topic research.
- [x] Formalized the primary research problem, scope, non-goals, research questions, and success criteria.
- [x] Fixed the policies that govern target generation, exact-duplicate handling, splitting, runtime compatibility, evaluation, calibration, and thresholds.

### Completion gate

The research problem, documentation method, decision authority, and project-tracking mechanism are explicit and linked. This gate is satisfied.

### Maintenance action

Reopen this section only when the thesis objective, scope, or binding methodological decisions change. Record the change in the history log.

## 2. Data

**Status:** Done

The Data macro-section covers source understanding, shared image storage, historical compatibility, deterministic target standardization, split generation, and runtime integration. Persistent outputs are intentionally limited to the common image collection and one selected metadata file in each split.

The entries below are intentionally limited to first-level Data work packages. Lower-level implementation tasks, decisions, and verification evidence are retained in the completed Data implementation plan and linked durable documents.

### First-level work-package status

| Work package | Status | Next action |
| --- | --- | --- |
| 2.1 Yummly data understanding, storage, and compatibility | **Done** | Audit, shared image store, retention manifest, historical reproduction, and read-only anchor smoke checks are complete. |
| 2.2 `ingredients_target` standardization and vocabulary | **Done** | FoodOn-first `v5` generation, exact-plus-fallback association, mapping rules, and support policy are frozen. |
| 2.3 Deterministic metadata generation and split | **Done** | `v4` remains the baseline; `v5` also passed all automatic image, split, leakage, distribution, vocabulary, and cardinality checks. |
| 2.4 Runtime target integration and `<UNK>` removal | **Done** | Full-default and legacy contracts pass; real CUDA training, checkpoint reload and dashboard checks complete, with 149 repository tests passing. |

### 2.1 Yummly data understanding, storage, and compatibility

**Status:** Done

The Yummly audit, lineage reconstruction, quality analysis, shared image store, and read-only field audit are complete. The November 2024 ResNet ingredient-selection process has also been reconstructed. Data Work package 2.1c now has a maintained retention manifest, exact historical reproduction, immutable selected metadata, analysis provenance, and three executable checkpoint anchors; its compatibility gate passed read-only on 2026-08-12. No cleanup has been authorized.

**Evidence:** [`project_objective/yummly_data_audit.md`](project_objective/yummly_data_audit.md), [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md), [`plans/recognizable_ingredient_selection.md`](plans/recognizable_ingredient_selection.md), and [`../src_scratches/data_anlysis/README.md`](../src_scratches/data_anlysis/README.md).

**Completion gate:** The selected retention manifest is verified, maintained code reproduces the historical 40-label intersection, and the retained checkpoint anchors load through the shared image layout without rewriting saved semantics.

**Next action:** Maintain the retention manifest and compatibility validator; any cleanup proposal requires a separate reviewed decision.

### 2.2 Improved `ingredients_target` standardization

**Status:** Done

The target pipeline now uses deterministic normalization, pinned FoodOn exact association, approved local fallback rules, exact retry, train-only support filtering, and retained local concepts. The resulting `v5` vocabulary is frozen for standard experiments; mapping authority and controlled-vocabulary evidence remain in their dedicated durable documents.

**Evidence:** [`project_objective/ingredient_vocabulary_audit.md`](project_objective/ingredient_vocabulary_audit.md), [`project_objective/benchmark_decisions.md`](project_objective/benchmark_decisions.md), [`implementation_details/ingredient_mapping_rules.md`](implementation_details/ingredient_mapping_rules.md), and [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md).

**Completion gate:** Identical inputs and rules produce identical targets and ordering, accepted collision boundaries are covered by tests, and the frozen `v5` generation is reproducible without modifying legacy generations. This gate is satisfied.

**Next action:** Maintain the pinned FoodOn resource, mapping registry, and active implementation plan when the target pipeline changes.

### 2.3 Deterministic metadata generation and split

**Status:** Done

The benchmark builder validates images automatically, groups only byte-identical images by SHA-256, allocates a deterministic 80/10/10 split, checks distribution and referential-integrity constraints, and writes one selected metadata generation per split. The `v4` baseline and `v5` generation remain reproducible and legacy generations are unchanged.

**Evidence:** [`research/topics/dataset_splitting/split_strategy.md`](research/topics/dataset_splitting/split_strategy.md), [`technical_details/data/yummly_benchmark_split/explaination.md`](technical_details/data/yummly_benchmark_split/explaination.md), and [`project_objective/benchmark_decisions.md`](project_objective/benchmark_decisions.md).

**Completion gate:** The selected metadata generations pass image, vocabulary, split, leakage, distribution, and deterministic-rerun checks, with no byte-identical image crossing split boundaries. This gate is satisfied.

**Next action:** Follow the active Data implementation plan before generating a replacement.

### 2.4 Runtime target integration and `<UNK>` removal

**Status:** Done

The runtime defaults new experiments to `ingredients_target`, derives the vocabulary from training metadata, omits `<UNK>` from new multi-label outputs, and preserves the serialized legacy encoder contract. The rerunnable bounded WSL smoke completed four CUDA updates, exact-logit checkpoint reload and actual dashboard checks. Dashboard reconstruction now preserves model-specific preprocessing. All 149 repository tests pass. Metadata and retained experiments are unchanged; no predictive test evaluation occurred.

**Evidence:** [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md), [`implementation_details/image_data_loading.md`](implementation_details/image_data_loading.md), [`../tests/test_multilabel_encoder_contract.py`](../tests/test_multilabel_encoder_contract.py), and [`../tests/test_images_recipes_dataloader.py`](../tests/test_images_recipes_dataloader.py).

**Completion gate:** New experiments use the selected target field and output contract, retained historical experiments preserve their semantics, and real training, checkpoint and dashboard smoke validations pass. Satisfied on 2026-10-06.

**Next action:** Maintain the contract and rerunnable smoke when runtime consumers change; proceed to the separate Phase 5 planning handoff.

### Data macro-section completion gate

Shared image loading, legacy compatibility, deterministic target generation, exact-duplicate-safe splitting, runtime integration, and the `<UNK>` decision all pass their completion gates. **Satisfied on 2026-10-06.** This is engineering readiness, not completed benchmark training or test evaluation.

## 3. Ingredient selection

**Status:** Done

This macro-section selects ingredients from numerical evidence of model learning on the existing recipe-ingredient targets. The adopted profile separates train-AP optimization, validation-AP generalization, temporal checks, support, numerical controls and uncertainty; F1 is a secondary diagnostic. Under [Phase 3-D5](project_objective/model_comparison_methodology.md#phase-3-d5--numerical-selection-and-optional-interpretation-appendix), manual semantic relevance and visual observability are optional interpretation outside selection. Numerical evidence is not reduced to raw frequency or one transient F1 maximum.

The 2024 ResNet selection has been reconstructed as a historical baseline: it intersected four top-quartile sets defined by each label's maximum train F1 and produced 40 legacy `ingredients_ok` labels. Train F1 is accepted as an intentional convergence signal for that narrow question, but the legacy rule is not reused as the final `v5` criterion and the old/new plot is not accepted as comparative evidence. Macro-section 3 retains ownership of the new shared selected vocabulary. Subphase 4B froze the independent EfficientNetV2-S reference-selector protocol, Phase 3 P1 froze its campaign and measurement contract, and P2 implemented and resource-validated the maintained workflow without inspecting outcomes. P3 executes the sealed campaign and exposes only the pilot cohort. The Subphase 4A experiment-model shortlist remains a separate decision.

### Work-package status

| Work package | Status | Next action |
| --- | --- | --- |
| 3.1 Preliminary evidence and historical reconstruction | **Done** | Retain the exact max-Q3 intersection as a read-only baseline and regression fixture. |
| 3.2 Selection criteria and experimental protocol | **Done** | [D6](project_objective/model_comparison_methodology.md#phase-3-d6--held-out-quality-inclusion-policy) adopts the reviewed held-out-quality inclusion policy without a count target, preserving the selector/campaign and original D4 decision. |
| 3.3 Reproducible `v5` learnability study | **Done** | The [D6 result](experiment_results/phase3_d1_v3_d6_profile.md) has matching paired image-cluster uncertainty, independent reasons and fixed sensitivity. Two executions agree exactly; original D4/P4 evidence remains unchanged. |
| 3.4 Former mandatory relevance and visual-observability validation | **Superseded** | Retain the rubric, source and unannotated packet as an [optional interpretation appendix](project_objective/ingredient_observability_protocol.md). No human review is required for vocabulary selection or comparison. |
| 3.5 Final numerical-profile-based vocabulary and integration | **Done** | The shared D6 projection is frozen and integrated explicitly with canonical training/checkpoint/analysis APIs. Real train/validation column parity, 140 tests and 72-artifact retention checks pass. Historical launchers/notebooks stay immutable evidence; physical cleanup requires a separate reviewed decision. |

### Completed evidence

- [x] Quantified legacy label support and long-tail behavior.
- [x] Identified corrupted frequent labels and obvious alias fragmentation.
- [x] Demonstrated cuisine-dependent contextual shortcuts.
- [x] Defined the observability states `direct`, `contextual`, `not_inferable`, and `uncertain`.
- [x] Set preliminary support requirements for reliable headline evaluation.
- [x] Reconstructed the four historical ResNet stages, exact 40-label intersection, experiment groups, metadata projection, and evidence hierarchy.
- [x] Resolved the historical discrepancies: train F1 is an intentional convergence signal; the maximum-only rule requires improvement; the fourth run is unweighted; augmentation reporting is inverted; and the old/new plot is not a valid comparison.
- [x] Opened the dedicated [`recognizable_ingredient_selection.md`](plans/recognizable_ingredient_selection.md) operational plan.
- [x] Adopted a research-informed learnability decision profile: train AP for optimization, validation AP for generalization, and separate stability, support, mechanism, and observability evidence; F1 is fixed-policy diagnostic evidence only.
- [x] Adopted the cross-phase methodology: Subphase 4A chooses the experiment model categories, Subphase 4B chooses the reference selector, Macro-section 3 produces one shared selected vocabulary, and Macro-sections 6–7 separate full-task comparison, transferred vocabulary ablation, support-matched random controls, and optional local adaptation.
- [x] Selected supervised EfficientNetV2-S with full-backbone fine-tuning and an independent 165-logit pooled head as `M_ref` after a current-source audit and bounded 384-pixel output, gradient, provenance-path, and 8 GB smoke; no candidate training or predictive comparison informed the choice.
- [x] Froze 4B-D1: exact EfficientNetV2-S weights, full end-to-end trainability, RGB 384-pixel full-frame fit/pad preprocessing, independent pooled head, train-only positive-weighted BCE, FP32 batch-8 resource target, interpretation limits, and the Phase 3 provenance handoff.
- [x] Froze Phase 3-D1: one seed-42 AdamW/warm-up/cosine run for 20 epochs, deterministic train/validation audits every two epochs, robust AP windows, fixed-0.5 F1 diagnostics, final validation bootstrap intervals, low-cost controls, a blind support-stratified pilot cohort, and one-run P3/P4 reuse.
- [x] Implemented P2 through canonical source modules and thin CLIs: exact selector/transform/loss/schedule, train/validation-only data access, fixed-state AP/F1 audits, bootstrap/controls, blind rule gate, manifest schemas, and deterministic historical reproduction. All 64 tests pass and the real RTX 4060 batch-8 FP32 gate fits the 8 GB boundary.
- [x] Amended execution before per-label inspection under Phase 3-D2/D3: main-workspace launcher and source snapshots, effective batch 128, and a 40-epoch horizon; the extended 72-test suite verifies the current batch, schedule, final windows and analysis contract.
- [x] Completed the fresh 40-epoch v3 selector campaign, exposed only the sealed 24-label pilot, and froze [Phase 3-D4](project_objective/model_comparison_methodology.md#phase-3-d4--pilot-frozen-numerical-profile-rule) with absolute AP/support/stability/advantage gates and a conservative bootstrap overlap band. The [reviewed pilot record](experiment_results/phase3_d1_v3_pilot.md) is provisional, not a selected vocabulary.
- [x] Applied the unchanged D4 rule to all 165 labels in P4, confirmed all 24 pilot decisions exactly, and produced deterministic named provisional groups and scientific figures. The [reviewed full-profile record](experiment_results/phase3_d1_v3_full_profile.md) does not certify direct visibility or a final vocabulary.
- [x] Adopted D6 after the objective-alignment review and completed corrected saved-score analysis: 59 eligible, 60 uncertain and 46 below the operational quality floor, with paired image-cluster intervals, deterministic artifacts and an explicit outcome-informed boundary. The [reviewed result](experiment_results/phase3_d1_v3_d6_profile.md) records verification and limitations.

### Active after the Subphase 4B protocol freeze and handoff

- [x] Freeze the remaining global `M_ref` campaign and measurement contract without inspecting new selector outcomes.
- [x] Implement the 4B-D1 wrapper/transform/loss and Phase 3-D1 optimizer, scheduler, deterministic audit, AP/F1, bootstrap, pilot-isolation, provenance, and output-schema paths through canonical APIs.
- [x] Piloted robust train-AP learning dynamics and validation-AP statistics on the sealed cohort, including initialization/early-to-late change, late-window levels, temporal sensitivity, final validation bootstrap uncertainty, frozen profile gates and fixed-0.5 F1 diagnostics. Configuration sensitivity remains unavailable under the single-configuration contract.
- [x] Add Phase 3 support/prevalence and non-visual controls, and preserve the matching fields needed for Macro-section 6's later matched-size vocabulary controls. No reduced-vocabulary or random-control training claim is made in P4.
- [x] Implement deterministic historical reproduction and `v5` analysis in a dedicated source package, with plot-ready evidence generated from validated manifests rather than notebook state.
- [x] Superseded mandatory manual relevance and observability review under D5; retain the prepared material as an optional future-results appendix without selection effects.
- [x] Audited original inclusion gates against recipe-ingredient prediction, reproduced D4 outcomes and recorded bounded counterfactual sensitivity without a retained-count target.
- [x] Recorded D6 and matching validation uncertainty separately, retaining D4 as the original pilot-frozen result.
- [x] Freeze the named shared `v5` projection and retain the uncertain/below-floor groups without using test outcomes; runtime integration remains separate.
- [x] Integrate the frozen projection as an explicit opt-in without changing the 165-label default or removing records; verify saved class identity, full/light checkpoint reload/resume, analysis projection and historical compatibility. P7 closes with documented retain/deprecate/defer dispositions and no deletion.

### Completion gate

The historical rule is reproduced by maintained read-only code; the `v5` campaign and analysis are deterministic with complete provenance; the final shared projection is justified by an explicit versioned inclusion rule and matching uncertainty, with original D4 evidence and the post-outcome amendment boundary preserved; integration and retention gates pass; and no test outcome or manual review influences selection. Support and single-run limitations remain explicit. An annotation report is not required.

### Next action

The [P7 completion checkpoint](plans/recognizable_ingredient_selection.md#p7-integration-and-retirement-checkpoint--2026-10-06) closes this macro-section. Later training/comparison consumes the shared projection under the frozen methodology; this does not itself authorize a new campaign. The independent Data 2.4 checks are now also complete. Manual review remains optional, physical cleanup requires a separate reviewed decision, and retained interrupted runs plus the documented gate-device limitation remain unchanged.

## 4. Model research

**Status:** Done

This macro-section has two coordinated but decision-independent subphases. Subphase 4A is the primary model-research stream and chooses the model categories to implement, tune, and compare in the thesis experiments. Subphase 4B chooses only the reference selector `M_ref` used by Macro-section 3 to generate the shared selected ingredient vocabulary.

Broad discoveries, primary-source catalogs, implementation audits, and resource evidence may support both subphases. Their decisions remain separate: selection or exclusion as `M_ref` does not include or exclude a model from the experiment shortlist, and inclusion in the experiment shortlist does not make a model the selector. Each subphase records its own criteria, rationale, status, and output.

### Subphase status

| Subphase | Status | Owned outcome | Next action |
| --- | --- | --- | --- |
| 4A Experimental-model research | **Done** | EfficientNetV2-S and MaxViT-T plus P2-S dual-scale ingredient-query readout with pooled context, with source-linked hypotheses and implementation gates. | Use [4A-D1/4A-D2](project_objective/experimental_model_portfolio.md) and the [completed plans](plans/README.md#completed-plans) for the Phase 5 handoff; data readiness is satisfied. |
| 4B Reference-selector research | **Done** | The binding 4B-D1 EfficientNetV2-S selector protocol and Phase 3 reproducibility/instrumentation handoff. | Maintain the decision and reopen it only through a versioned methodology revision if its implementation gate fails before outcome inspection. |

### 4A. Experimental-model research

**Status:** Done

This is the principal stream of Macro-section 4. It asks which representation, architecture, head, and adaptation hypotheses deserve controlled comparison on the frozen benchmark. It owns the experiment-model shortlist and the research questions that later drive implementation, tuning, and final comparison.

#### Completed evidence

- [x] Documented existing ResNet, DenseNet, DINOv2, and dummy-model implementations.
- [x] Produced a technical deep dive for DINOv2 ViT-B/14.
- [x] Defined the repository structure for dated discovery and topic-focused research.
- [x] Formalized model-relevant challenges: partial observability, correlated labels, long-tail support, contextual shortcuts, calibration, and interpretability.
- [x] Completed a dated broad state-of-the-art discovery grounded in the repaired-benchmark objective and 8 GB compute constraint.
- [x] Retained the architecture, pretraining, source, implementation, and resource evidence from the 2026-08-22 selector-oriented discovery as reusable input where its transfer boundary matches a 4A research question; its selector dispositions are not an experiment shortlist.
- [x] Created the dedicated [`experimental_model_research.md`](plans/experimental_model_research.md) plan with an extensive evidence flow and four stages: broad discovery, candidate deep research, two-family selection, and separately planned custom attention-model research.
- [x] Completed 4A.1 with a problem-to-model requirements matrix, five **new** retained family/protocol candidates, grouped exclusions, a primary-source catalog, and a formal handoff to 4A.2; ResNet and DINOv2 are retained as already-used baseline anchors and excluded from the new-candidate count.
- [x] Adopted the established-family [experimental portfolio](project_objective/experimental_model_portfolio.md), with source-linked rationale, preferred protocols, resource fallbacks, and a handoff to custom research and implementation.
- [x] Recorded the [custom-model problem brief and component evidence](research/topics/custom_attention_model_design/README.md), with primary sources, negative evidence, provisional component routes and implementation limitations.
- [x] Defined [compatible custom design routes and a bounded scaling envelope](research/topics/custom_attention_model_design/architecture_compatibility_synthesis.md), with reproducible scalar estimates and explicit initialization, vocabulary and diagnostic integration limits.
- [x] Compared [three custom topologies](research/topics/custom_attention_model_design/topology_proposals.md) and adopted P2-S in [4A-D2](project_objective/experimental_model_portfolio.md#4a-d2--custom-attention-topology), completing the custom and parent 4A plans without model execution.

#### Planned scope and completion

- [x] Review at least the two preceding discoveries before the 4A.1 discovery.
- [x] Execute 4A.1 and retain three to five scientifically distinct, accessible family-level candidates after mapping the current problem constraints.
- [x] Complete one normalized deep-research dossier per candidate with explicit transfer boundaries, resource metadata, and falsifiable benchmark hypotheses.
- [x] Select exactly two established families through the 4A.3 qualitative gate and record the binding shortlist.
- [x] Created the dedicated [`custom_attention_model.md`](plans/custom_attention_model.md) feature plan with four research subphases: problem/evidence synthesis, component research, compatibility synthesis, and three topology proposals.
- [x] Complete the four 4A.4 research subphases, compare three topology-level proposals with S/M/L scales, and select one for implementation.

#### Completion gate

The dedicated 4A plan is complete; a primary-source evidence chain supports exactly two established model families and one selected custom attention topology; all three categories have falsifiable benchmark hypotheses and credible resource and implementation paths; the custom research preserves three reviewed topology proposals with S/M/L scaling rules; and no candidate training, HPO, or test outcome influenced selection.

This research gate is satisfied. Actual loading, training stability, resource feasibility and comparative effectiveness are not established.

#### Next action

Maintain the source-to-decision record and carry the three-category [portfolio](project_objective/experimental_model_portfolio.md) into Phase 5 planning after Data 2.4 readiness. The selected custom starts at S with a same-S frozen-encoder adaptation fallback; larger scales and extra ablations are not mandatory campaigns. Subphase 4B is independently complete and does not change the 4A portfolio.

### 4B. Reference-selector research

**Status:** Done

This subphase asks which single model protocol is a sufficiently sensitive, interpretable, reproducible, and affordable measurement instrument for Phase 3 label learnability. It does not choose the final experiment winner and does not own the broader model shortlist.

- **Completed evidence:** R0.1 provides the selector-oriented landscape across supervised, visual self-supervised, vision-language, food-domain, and structured multi-label families. R0.2 maps credible paths to the `v5` interface, 8 GB boundary, and shared instrumentation/provenance gap. R1 retains three protocol-level finalists. R2 selects full-fine-tuned EfficientNetV2-S after source inspection and bounded synthetic output, gradient, provenance-path, and 8 GB checks. R3 freezes its exact model-side protocol as [4B-D1](project_objective/model_comparison_methodology.md#4b-d1--frozen-reference-selector-protocol), records [D12](project_objective/benchmark_decisions.md#d12-frozen-reference-selector-boundary), and hands the remaining campaign work to Phase 3 without claiming comparative accuracy.
- **Pending:** No 4B task remains. EfficientNetV2-S integration, AP instrumentation, pilot execution, and vocabulary generation belong to Macro-section 3; its campaign freeze is complete under Phase 3-D1.
- **Completion gate:** Met on 2026-09-15: one justified `M_ref` passed the declared gates, its reproducible model-side protocol and limitations are binding, and Macro-section 3 accepted the instrumentation and execution handoff.
- **Next action:** Maintain the completed [`reference_selector_research.md`](plans/reference_selector_research.md) record. Execute Phase 3 P2 without reopening the selector from downstream outcomes.

**Model-research macro-section completion gate:** Met on 2026-09-15. Subphase 4A provides the independent experiment-model portfolio and Subphase 4B provides the frozen reference selector; neither decision substitutes for the other.

## 5. Additional model implementation

**Status:** In progress

This macro-section covers architectures selected by Subphase 4A that are not already implemented in the repository.

The [experimental portfolio](project_objective/experimental_model_portfolio.md) provides the binding established-pair/DICA-Net-S handoff (historically P2-S); the 384-pixel selector stays separate. The [operational plan](plans/additional_model_implementation.md) tracks implementation and qualification. [5.1–5.3](implementation_details/experimental_model_contract.md) supply common foundations, experimental EfficientNet/MaxViT and tested opt-in Lightning/configuration restoration. MaxViT's original artifact and DICA-Net's tensor graph are verified; custom initialization/persistence/integration and remaining artifact/resource/consumer acceptance remain. Implementation completion does not launch comparative training.

### Existing foundation

- [x] Shared `BaseModel` abstraction and model-dependent transforms exist.
- [x] ResNet, DenseNet, DINOv2, and dummy models exist.
- [x] Lightning integration and experiment configuration are available.

Existing models are historical baselines, not evidence that the additional-model phase is complete.

### First-level work packages

All execution detail and lower-level custom checkpoints belong to the [feature plan](plans/additional_model_implementation.md).

| Work package | Status | Dependency | Completion gate and next action |
| --- | --- | --- | --- |
| 5.1 Shared implementation contract and foundations | **Done** | Completed 4A portfolio, Data 2.4 and P7 | Versioned preprocessing/identity, initialization/offline construction and exact batching are tested; engineering policy declared. 28 focused and 177 repository tests pass. Actual adapter/consumer integration remains subsequent work. |
| 5.2 Experimental EfficientNetV2-S | **Done** | 5.1 | Distinct 224-pixel pooled head, full/frozen state and strict exact-batch/full-light persistence pass; 57 focused/206 repository tests. Actual qualification remains 5.5. |
| 5.3 Experimental MaxViT-T | **Done** | 5.1 | Intact backbone/common readout, explicit normalization/geometry, approved artifact, strict offline full/light restoration and CPU diagnostics pass; actual qualification remains 5.5. |
| 5.4 DICA-Net-S custom model | **In progress** | 5.1 and reusable 5.2 foundations | Tensor graph, numerical attention and label-row invariants pass. Complete approved initialization, adaptation, persistence and capability-aware integration, then qualify through 5.5. |
| 5.5 Measured qualification and Phase 6 handoff | **Pending** | Each model's implementation gates | All three pass reproducible bounded resource/train/restore/consumer checks and affected regressions; hand off measured capabilities and unresolved comparison-policy choices. |

### Resume gate

Subphase 4A approved the portfolio and Data 2.4 verified the canonical data contract. The planning resume gate is satisfied on 2026-10-06; each new model still requires its own implementation, resource and integration checks.

### Completion gate

Every selected experimental model passes its interface, state, persistence, vocabulary and diagnostic-capability tests, integrates with the canonical training path, and has reproducible bounded measured resource/runtime evidence. Adopted fallbacks and comparison consequences are explicit, current documentation is synchronized and affected legacy compatibility is preserved. This gate does not complete the later metric suite, HPO policy or final test evaluation.

### Next action

Continue [5.4](plans/additional_model_implementation.md#54--dica-net-s-custom-implementation) with DICA-Net-S initialization, adaptation and persistence, reusing the verified tensor core and common/experimental Lightning contracts. The custom remains S with the adopted same-topology frozen fallback. Actual real-consumer/CUDA acceptance remains 5.5; no Phase 5 CUDA run or benchmark campaign has started.

## 6. Training and hyperparameter tuning

**Status:** Deferred

This macro-section covers baseline training, controlled model training, hyperparameter search, run selection, and reproducibility under the frozen benchmark.

The user-authorized parallel historical follow-up has prepared
[selected-task ResNet/DINO launchers](implementation_details/selected_vocabulary_training.md)
while Phase 5 proceeds. They transfer the reviewed full-task trial configurations
to the shared D6 vocabulary, with fresh initialization and no new HPO. Preparation
and dry-run verification are complete; the user will launch the runs. This does
not release the final benchmark gate or redefine historical hyperparameters as
the future `H_base(m)`.

The binding design is in [`project_objective/model_comparison_methodology.md`](project_objective/model_comparison_methodology.md): tune each model category once on the common full vocabulary, use transferred hyperparameters for the shared selected-vocabulary ablation, use support-matched random vocabulary controls with the reference selector, and keep any selected-task local adaptation separate.

### Existing historical infrastructure

- [x] Lightning trainers, checkpointing, logging, and early stopping exist.
- [x] One-shot experiment launchers exist.
- [x] Optuna-based hyperparameter tuning exists.
- [x] CSV, TensorBoard, and offline W&B logging paths exist.
- [x] Historical experiments and plots exist for legacy data and configurations.

Historical runs remain useful engineering evidence but are **Superseded** for final comparison by the future benchmark protocol.

### Pending after resume

- [ ] Freeze one declared seed per configuration, budgets, stopping rules, transforms, metrics, logging requirements, and the single-run reporting limitation.
- [ ] Run prevalence and cuisine-prior non-visual baselines.
- [ ] Run a simple supervised convolutional baseline.
- [ ] Run a frozen pretrained visual encoder with a linear multilabel head.
- [ ] Define model-specific hyperparameter spaces before opening each study.
- [ ] Verify Optuna study isolation, resumption behavior, and trial traceability.
- [ ] Run one bounded `V_base` tuning study per approved model category using validation only and fixed comparable budgets.
- [ ] Train the shared selected-vocabulary ablation with each category's unchanged full-task hyperparameters, then evaluate the paired full-task predictions on the same selected labels.
- [ ] Run the reference-selector selected-versus-support-matched-random vocabulary controls under unchanged full-task hyperparameters.
- [ ] If selected-task optimized results are required, run the same predeclared small local-adaptation panel around each category's full-task configuration; keep its results separate from the vocabulary-effect ablation.
- [ ] Calibrate and select thresholds using validation only after model selection.
- [ ] Preserve configurations, checkpoints, metrics, environment information, and run identifiers.

### Resume gate

Resume when the benchmark, selected ingredient vocabulary, data loader, evaluation implementation, and at least the required baseline models are ready.

### Completion gate

Every required baseline and shortlisted model has comparable full-task and shared selected-task runs under the frozen protocol; the reference-selector random controls and any local-adaptation results are separately identified; no test set was used during selection.

### Next action

Do not launch final training or tuning on the legacy 182-label split. Resume after Subphase 4A freezes the experiment-model shortlist and Subphase 4B plus Macro-section 3 freeze `M_ref` and the shared selected vocabulary; small smoke tests remain allowed when clearly marked as engineering validation.

## 7. Results comparison

**Status:** Deferred

This macro-section covers the frozen evaluation protocol, statistical comparison, qualitative analysis, and final interpretation of model behavior.

A preparatory [artifact and observability audit](implementation_details/experiment_artifacts.md) was completed on 2026-09-14 for the existing target-v5 ResNet and DINOv2 campaigns. It records mixed loss objectives, pruning, restart conflicts, sparse checkpoint selection and AMP-scaled gradient limitations. The [comparison application](implementation_details/experiment_comparison.md) is implemented and validated on those campaigns; final benchmark analysis remains deferred, and the exploratory report does not establish final model rankings or satisfy the resume gate.

The reviewed [historical basic_v5 comparison](experiment_results/basic_v5_resnet_dinov2.md) retains the bounded empirical outcome: full fine-tuning of pretrained ResNet18 trial 77 outperformed the frozen-backbone DINOv2-B/14 linear-probe trial 61 on the shared unweighted validation-loss cohort, with lower aggregate threshold metrics and lower compute and artifact cost. This is historical validation evidence without test evaluation, repeated fixed-configuration seeds, per-ingredient metrics, or a matched adaptation protocol. It does not change the final benchmark methodology, the adopted model portfolio, or this macro-section's **Deferred** status.

### 7.1 Experiment observability and comparison tooling

**Status:** Done

Optional per-ingredient logging is implemented in the Lightning model, defaulting to false, together with an offline N-experiment comparison command that generates strict JSON and self-contained HTML. The tool reconciles current CSV, TensorBoard, Optuna, W&B and checkpoint evidence, separates objective cohorts and retains audited limitations.

**Dependencies:** Existing artifact formats and the canonical Lightning configuration/logging path. The implementation remains independent of final test-set access and does not release the benchmark resume gate.

**Operational plan:** [`plans/experiment_comparison.md`](plans/experiment_comparison.md). The maintained runtime contract is [`implementation_details/experiment_comparison.md`](implementation_details/experiment_comparison.md).

**Completion gate:** Met on 2026-09-14 through focused unit/integration tests, a bounded Lightning fit, and an end-to-end run over both 100-trial target-v5 campaigns with selected W&B/checkpoint evidence.

**Next action:** Use the tool for exploratory and future benchmark-ready artifact inspection; resume final comparison only after Macro-section 6 produces comparable selected runs.

### Methodological decisions already completed

- [x] Selected label-macro mean average precision and micro F1 as paired primary metrics.
- [x] Defined secondary ranking, set-prediction, calibration, cardinality, and observability-slice metrics.
- [x] Required validation-only calibration and threshold selection.
- [x] Fixed a one-declared-seed-per-configuration resource limit and required finite-sample uncertainty reporting without treating it as seed-level stability.

### Pending implementation and analysis

- [ ] Implement and unit-test the metric, calibration, threshold, aggregation, and bootstrap suite.
- [ ] Freeze the comparison table schema before inspecting test results.
- [ ] Compare model categories under identical data, vocabulary, transforms, budgets, declared single-run constraints, and selection rules for Q1 and Q2.
- [ ] Report the Q3 selected-versus-random vocabulary ablation only for the reference selector and keep it distinct from model-category rankings.
- [ ] Report any Q4 local-adaptation results as optimized selected-task results, distinct from the transferred-hyperparameter ablation.
- [ ] Report primary and secondary metrics with uncertainty.
- [ ] Optionally interpret visual/contextual examples in the appendix after future results; manual categories do not define required comparison slices or primary rankings.
- [ ] Analyze performance by ingredient, support tier, cuisine, image quality, and predicted cardinality.
- [ ] Inspect representative successes, false positives, false negatives, and shortcut behavior.
- [ ] Report compute, memory, training time, and inference cost.
- [ ] Evaluate selected final configurations on test exactly once.
- [ ] State which hypotheses are supported, rejected, or inconclusive.

### Resume gate

Resume comparative analysis after Macro-section 6 produces comparable selected runs. Evaluation code may be implemented and tested earlier without final test access.

### Completion gate

The comparison has complete provenance, reports finite-sample uncertainty without overstating the single-run evidence, includes failure analysis and resource costs, keeps Q1–Q4 claims distinct, and directly answers the research questions without overstating ingredient visibility.

### Next action

Use the completed [comparison tooling](implementation_details/experiment_comparison.md) for artifact inspection while retaining the final-comparison resume gate. Once Macro-section 6 produces comparable selected runs, freeze the final table schema before test evaluation.

## 8. Thesis writing

**Status:** Pending

This macro-section turns the verified project evidence into the thesis manuscript. Writing should progress alongside the project rather than begin only after experiments finish.

### Planned work packages

| Work package | Status | Dependency |
| --- | --- | --- |
| 8.1 Thesis outline and claim map | **Pending** | Stable research questions and institutional format. |
| 8.2 Background and related work | **Pending** | Model discovery and topic research. |
| 8.3 Dataset and methodology | **Pending** | Frozen benchmark and experiment protocol. |
| 8.4 Experimental setup | **Deferred** | Final training configuration. |
| 8.5 Results and discussion | **Deferred** | Completed comparison. |
| 8.6 Limitations, conclusion, and revision | **Deferred** | Complete draft and final evidence. |

### Existing source material

- [x] Repository and implementation knowledge is documented.
- [x] The project objective and research questions are documented.
- [x] The Yummly audit and benchmark decisions are documented.
- [x] Model and technical deep dives provide reusable background material.

These documents are inputs to the thesis, not substitutes for a coherent manuscript.

### Pending

- [ ] Obtain the required thesis structure, formatting rules, and submission constraints.
- [ ] Create a chapter outline linked to research questions and evidence artifacts.
- [ ] Maintain a claim-to-evidence table so every quantitative claim is reproducible.
- [ ] Draft stable chapters early and mark unresolved result-dependent passages.
- [ ] Consolidate citations from discovery and topic research using primary sources.
- [ ] Add figures and tables only from versioned analysis outputs.
- [ ] Perform technical, editorial, citation, and formatting reviews.
- [ ] Freeze the final manuscript and archive the exact supporting benchmark, code version, configurations, and results.

### Completion gate

The final manuscript is internally consistent, all claims trace to evidence, required reviews are complete, formatting requirements pass, and the submitted artifact is archived with its reproducibility references.

### Next action

Create the thesis outline and claim map as soon as the institutional template and submission requirements are available.

## Cross-phase dependency flow

```text
project foundation [Done] -> data [Done]
data -> 4A experimental-model research [Done] -> additional models [In progress: 5.1–5.3 Done, 5.4 active]
data -> 4B reference-selector research [Done] -> ingredient selection [Done]
4A <-> shared discoveries, source catalogs, and technical evidence <-> 4B
additional models + ingredient selection -> training and HTuning [Deferred] -> results comparison [Deferred] -> thesis completion

Thesis outline and stable chapters may progress in parallel.
```

## Project history log

This table is append-only. Add one row when a macro-section or first-level work package changes status, a major artifact is completed, or an earlier result is superseded. Detailed lower-level history remains in the responsible feature plan or evidence document; do not add it here.

| Date or period | Area | Event | Resulting status | Evidence |
| --- | --- | --- | --- | --- |
| Known by 2026-08-02 | Model implementation | ResNet, DenseNet, DINOv2, dummy models, and the shared model abstraction already exist. | Existing foundation **Done** | [`implementation_details/models.md`](implementation_details/models.md) |
| Known by 2026-08-02 | Training infrastructure | Lightning, one-shot training, Optuna tuning, checkpointing, and experiment logging already exist. | Existing infrastructure **Done** | [`../README_PROJECT_KNOWLEDGE.md`](../README_PROJECT_KNOWLEDGE.md) |
| 2026-08-02 | Documentation | English documentation conventions and the research directory hierarchy were formalized. | Project foundation **In progress** | [`README.md`](README.md), [`research/README.md`](research/README.md) |
| 2026-08-02 | Project objective | Problem definition, success criteria, and benchmark decisions were formalized. | Project foundation **Done** | [`project_objective/README.md`](project_objective/README.md) |
| 2026-08-02 | Data | Full legacy Yummly audit and deeper reproducibility, collision, duplicate, and shortcut analyses were completed. | Data **In progress** | [`project_objective/yummly_data_audit.md`](project_objective/yummly_data_audit.md) |
| 2026-08-02 | Planning | The roadmap was converted into a permanent whole-project tracker organized by thesis macro-section. | Overall project **In progress** | This document |
| 2026-08-02 | Project governance | Reading and updating this tracker was made mandatory in the repository knowledge document. | Tracking policy **Done** | [`../README_PROJECT_KNOWLEDGE.md`](../README_PROJECT_KNOWLEDGE.md) |
| 2026-08-02 | Planning | The whole-project tracker was renamed to `general_plan.md`, and `docs/plans/` was introduced for implementation-level execution plans. | Planning structure **Done** | This document, [`plans/README.md`](plans/README.md) |
| 2026-08-02 | Planning | A task-level progress tracker was made mandatory in every implementation plan. | Implementation tracking policy **Done** | [`plans/README.md`](plans/README.md) |
| 2026-08-02 | Model research | A broad primary-source discovery synthesized models, data processing, augmentation, leakage control, evaluation, and a compute-aware experimental program. | Work package 4.3 **Done**; Model research **In progress** | [`research/discovery/2026-08-02/README.md`](research/discovery/2026-08-02/README.md) |
| 2026-08-03 | Planning | Feature plans became the operational trackers during implementation; general-plan synchronization was moved to feature completion except for material project-level changes. Necessary-only code comments were established as a common implementation directive. | Implementation planning policy **Done** | [`plans/README.md`](plans/README.md) |
| 2026-08-03 | Planning | Feature-plan tracking was changed from continuous updates to step-completion checkpoints. | Implementation tracking policy **Done** | [`plans/README.md`](plans/README.md) |
| 2026-08-02 | Data planning | Defined the smaller data pipeline: immutable legacy artifacts, in-memory compatibility, deterministic target generation, exact-image grouping, split metadata as the source of truth, and runtime integration. | Data work packages 2.1–2.4 **In progress/Pending** | [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md), [`project_objective/benchmark_decisions.md`](project_objective/benchmark_decisions.md) |
| 2026-08-03 | Data implementation | Created a verified common image store and first deterministic target metadata generation without changing legacy artifacts. | Data work packages 2.1–2.3 **Done**; 2.4 **In progress/Pending** | [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md) |
| 2026-08-03 | Data planning | Separated target-vocabulary evidence and extractor decisions from the implementation sequence; the initial candidate remained a validated comparison point. | Data work package 2.2 **In progress/Pending**; 2.3 **Deferred** | [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md) |
| 2026-08-04 | Data planning | Deferred historical compatibility until retained experiments are selected and accepted the new multi-label output policy while preserving selected legacy artifacts. | Data work package 2.1 **Deferred**; 2.4 **In progress** | [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md) |
| 2026-08-04 | Data analysis and decision | Completed the candidate vocabulary audit and approved recognizability-led, deterministic mapping decisions for the target pipeline. | Data work package 2.2 **In progress — decision gate closing** | [`project_objective/ingredient_vocabulary_audit.md`](project_objective/ingredient_vocabulary_audit.md), [`project_objective/benchmark_decisions.md`](project_objective/benchmark_decisions.md) |
| 2026-08-04 | Data documentation | Created the long-term source of truth for custom target mappings, multi-target expansions, exclusions, retained distinctions, and collision boundaries. | Data work package 2.2 **In progress — mapping contract recorded** | [`implementation_details/ingredient_mapping_rules.md`](implementation_details/ingredient_mapping_rules.md) |
| 2026-08-04 | Data implementation | Implemented the approved target rules and generated the selected `v4` candidate. It passed image, split, leakage, distribution, vocabulary, and deterministic-rebuild checks while leaving legacy artifacts unchanged. | Data work packages 2.2–2.3 **Done**; 2.4 **In progress** | [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md), [`implementation_details/ingredient_mapping_rules.md`](implementation_details/ingredient_mapping_rules.md) |
| 2026-08-04 | Data design | Reopened the target-generation design after observing information loss in the current candidate; the active Data implementation plan now governs the research and follow-on implementation before runtime integration resumes. | Data **In progress**; runtime integration **Deferred** | [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md) |
| 2026-08-04 | Data research and decision | Completed controlled-vocabulary research, the support-threshold sweep, and the exact-association decision. FoodOn is pinned as the semantic reference, fuzzy association is rejected, local concepts are retained, and the standard target vocabulary is shared across models. | Data work package 2.2 **Done** | [`plans/data_ingredient_refactor/controlled_vocabulary_evaluation.md`](plans/data_ingredient_refactor/controlled_vocabulary_evaluation.md), [`implementation_details/ingredient_mapping_rules.md`](implementation_details/ingredient_mapping_rules.md) |

| 2026-08-05 | Data implementation | Implemented the approved FoodOn-first pipeline, generated `ingredients_target_v5_metadata.json`, and passed exact-image-aware validation with 47,965/5,996/5,996 records and 165 train-supported targets. A clean rebuild matched the saved JSON objects byte-for-byte; legacy generations remain unchanged. | Data work packages 2.2–2.3 **Done**; 2.4 **Deferred** | [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md), [`../src/data_processing/resources/foodon_food_product_v2025_07_31.json`](../src/data_processing/resources/foodon_food_product_v2025_07_31.json) |
| 2026-08-06 | Data tooling | Added a reusable read-only audit for one metadata generation, split, and selected ingredient field. It reports complete value counts, distinct recipe support, cardinality distributions, cuisine summaries, and optional current-normalizer and co-occurrence views. | Data tooling **Done**; runtime integration **Deferred** | [`../src_scratches/data_anlysis/metadata_field_audit.py`](../src_scratches/data_anlysis/metadata_field_audit.py), [`../src_scratches/data_anlysis/README.md`](../src_scratches/data_anlysis/README.md) |
| 2026-08-06 | Runtime integration | Selected `v5` as the new default, removed `<UNK>` from new multi-label output space, retained the serialized robust encoder contract for legacy experiments, and updated the image-statistics consumer for the common image store. All 16 executable tests pass; full ML smoke execution awaits a compatible Torch/NumPy/Lightning environment. | Work package 2.4 **In progress** | [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md), [`../tests/test_multilabel_encoder_contract.py`](../tests/test_multilabel_encoder_contract.py) |
| 2026-08-06 | Benchmark methodology | Confirmed one frozen, exact-duplicate-group-aware multi-label-stratified Yummly split for all standard model comparisons. The test split remains unavailable to selection decisions; pure random splitting is not used. | Data split policy **Active** | [`research/topics/dataset_splitting/split_strategy.md`](research/topics/dataset_splitting/split_strategy.md), [`technical_details/data/yummly_benchmark_split/explaination.md`](technical_details/data/yummly_benchmark_split/explaination.md), [`project_objective/benchmark_decisions.md`](project_objective/benchmark_decisions.md) |
| 2026-08-06 | Project governance | Consolidated the cross-category documentation rules: durable storage, source-of-truth boundaries, directory responsibilities, provenance, retention, and completion checks. | Documentation organization **Done** | [`README_DOCS_ORGN.md`](README_DOCS_ORGN.md) |
| 2026-08-06 | Project governance | Reduced the Data section to first-level work packages and moved lower-level implementation detail to the active feature plan and durable evidence documents. | Documentation organization **Done**; Data summary **Synchronized** | [`README_DOCS_ORGN.md`](README_DOCS_ORGN.md), [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md) |
| 2026-08-10 | Data compatibility | Reconstructed the November 2024 ResNet ingredient-selection evidence, selected the minimum retained artifact set and three executable checkpoint anchors, and resumed the read-only compatibility work. | Work package 2.1 **In progress**; retention dependency **Done** | [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md) |
| 2026-08-12 | Data compatibility | Closed 2.1c with a 72-entry SHA-256 retention manifest, exact four-run/40-label reproduction, shared-image metadata smoke checks, and H1/H2/H3 checkpoint-anchor loads. No legacy artifact was rewritten or deleted. | Work package 2.1 **Done**; Data 2.4 remains **In progress** | [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md), [`../scripts/validate_legacy_experiments.py`](../scripts/validate_legacy_experiments.py) |
| 2026-08-10 | Ingredient selection | Accepted the historical max-train-F1 Q3 intersection as a baseline, resolved its reporting discrepancies, and opened a reproducible `v5` learning-dynamics feature plan with controls and observability evidence. | Work package 3.1 **Done**; 3.2 **In progress** | [`plans/recognizable_ingredient_selection.md`](plans/recognizable_ingredient_selection.md) |
| 2026-08-12 | Ingredient selection methodology | Adopted a research-informed decision-profile framework: train AP trajectory for optimization, validation AP for held-out generalization, fixed-policy F1 only as a diagnostic, and separate stability, support, mechanism, and observability evidence. Numerical gates remain a bounded-pilot decision. | Work package 3.2 **In progress** | [`plans/recognizable_ingredient_selection.md`](plans/recognizable_ingredient_selection.md), [`research/topics/label_learnability/learnability_assessment.md`](research/topics/label_learnability/learnability_assessment.md) |
| 2026-08-12 | Ingredient selection protocol | Removed repeated-seed training because the available time cannot support it. The `v5` study uses one declared seed per configuration, temporal/configuration checks, and finite-validation-sample uncertainty where feasible; it makes no seed-level stability claim. | Work package 3.2 **In progress** | [`plans/recognizable_ingredient_selection.md`](plans/recognizable_ingredient_selection.md) |
| 2026-08-12 | Comparative methodology | Bound the shared-vocabulary design: Macro-section 4 selects the reference selector, Macro-section 3 produces the selected vocabulary, and Macro-sections 6–7 separate full-task model comparison, transferred vocabulary ablation, support-matched random controls, and optional local adaptation. | Benchmark methodology **Active** | [`project_objective/model_comparison_methodology.md`](project_objective/model_comparison_methodology.md), [`project_objective/benchmark_decisions.md`](project_objective/benchmark_decisions.md) |
| 2026-08-12 | Ingredient selection | Deferred new `v5` selection execution until Macro-section 4 chooses the justified reference selector. Historical reconstruction and decision-profile planning remain retained. | Macro-section 3 and work packages 3.2–3.5 **Deferred**; Model research **In progress** | [`plans/recognizable_ingredient_selection.md`](plans/recognizable_ingredient_selection.md), [`project_objective/model_comparison_methodology.md`](project_objective/model_comparison_methodology.md) |
| 2026-08-12 | Model research planning | Opened Work package 4.6 and its bounded research plan to select the reference selector independently from the final model shortlist. | Work package 4.6 **Pending**; Macro-section 3 remains **Deferred** | [`plans/reference_selector_research.md`](plans/reference_selector_research.md) |
| 2026-08-16 | Runtime integration | Restored compatible WSL ML execution and made image-loader pinned memory platform-aware. Automatic mode is enabled only on native Windows, avoiding the observed WSL pin-memory-thread OOM while retaining portable saved configurations. | Work package 2.4 **In progress**; remaining smoke checks narrowed to training completion, checkpoint reload, and dashboard | [`implementation_details/image_data_loading.md`](implementation_details/image_data_loading.md), [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md) |
| 2026-08-22 | Reference-selector research | Started R0 with an explicit broad candidate-landscape discovery, including supervised, self-supervised, contrastive, and domain-pretrained options; existing ResNet, DenseNet, and DINO paths are not an exhaustive candidate set. | Work package 4.6 **In progress**; Macro-section 3 remains **Deferred** | [`plans/reference_selector_research.md`](plans/reference_selector_research.md) |
| 2026-08-22 | Reference-selector research | Completed R0.1 with a source-catalogued, non-ranked selector landscape and explicit boundaries for pretraining, adaptation, downstream label text, and structured heads. | Work package 4.6 **In progress**; R0.2 inventory is next; Macro-section 3 remains **Deferred** | [`research/discovery/2026-08-22/README.md`](research/discovery/2026-08-22/README.md), [`plans/reference_selector_research.md`](plans/reference_selector_research.md) |
| 2026-08-22 | Reference-selector research | Completed R0.2 with a repository-backed candidate/instrumentation matrix, verified and conditional intake tiers, explicit re-entry conditions, and a shared observability/provenance prerequisite; no selector was chosen. | Work package 4.6 **In progress**; R1 rubric is next; Macro-section 3 remains **Deferred** | [`research/discovery/2026-08-22/candidate_integration_inventory.md`](research/discovery/2026-08-22/candidate_integration_inventory.md), [`plans/reference_selector_research.md`](plans/reference_selector_research.md) |
| 2026-08-27 | Reference-selector planning | Simplified the remaining decision from five research/administrative stages to three outcome-driven stages: bounded shortlist, decision-relevant verification and choice, then freeze and handoff. Exhaustive candidate dossiers, numeric scoring, and comparative candidate training are no longer required. | Work package 4.6 **In progress**; R1 bounded shortlist is next; Macro-section 3 remains **Deferred** | [`plans/reference_selector_research.md`](plans/reference_selector_research.md) |
| 2026-08-27 | Model research planning | Split Macro-section 4 into primary Subphase 4A for experiment-model research and Subphase 4B for the reference-selector decision. Discoveries, source catalogs, and technical audits may feed both, but their criteria and decisions remain independently owned. | 4A and 4B **In progress**; the 4A plan is next and 4B continues at R1; Macro-section 3 depends only on 4B | [`research/discovery/2026-08-02/README.md`](research/discovery/2026-08-02/README.md), [`research/discovery/2026-08-22/README.md`](research/discovery/2026-08-22/README.md), [`plans/reference_selector_research.md`](plans/reference_selector_research.md) |
| 2026-08-28 | Experimental-model research planning | Created the dedicated 4A plan. It now selects two established families from a three-to-five-family research set and one custom attention topology from three proposals, while preserving primary-source evidence and explicit stage handoffs. | Subphase 4A **In progress**; 4A.1 broad family discovery is next | [`plans/experimental_model_research.md`](plans/experimental_model_research.md), [`project_objective/model_comparison_methodology.md`](project_objective/model_comparison_methodology.md) |
| 2026-08-28 | Experimental-model research | Completed 4A.1 with the problem-to-model requirements matrix, five retained family/protocol candidates, grouped exclusions, primary-source catalog, and the formal handoff to 4A.2. | Subphase 4A **In progress**; 4A.2 deep research is next | [`research/discovery/2026-08-28/README.md`](research/discovery/2026-08-28/README.md), [`plans/experimental_model_research.md`](plans/experimental_model_research.md) |
| 2026-08-28 | Experimental-model research | Refined the 4A.1 candidate count: ResNet and DINOv2 are already-used baseline anchors (their papers remain retained evidence), while the five selection candidates are now EfficientNetV2, Swin V2, SigLIP2, a structured query/set head, and MaxViT. | Subphase 4A **In progress**; 4A.2 deep research is next | [`research/discovery/2026-08-28/candidate_landscape.md`](research/discovery/2026-08-28/candidate_landscape.md), [`plans/experimental_model_research.md`](plans/experimental_model_research.md) |
| 2026-08-28 | Experimental-model research | Completed 4A.2 with five normalized candidate dossiers and a qualitative 4A.2 → 4A.3 synthesis. Architecture, evidence transfer limits, provenance/access, resource assumptions, and falsifiable hypotheses are recorded; no local training or test outcome influenced the handoff. | Subphase 4A **In progress**; 4A.3 established-family selection is next | [`research/topics/experimental_model_candidates/README.md`](research/topics/experimental_model_candidates/README.md), [`research/topics/experimental_model_candidates/comparative_synthesis.md`](research/topics/experimental_model_candidates/comparative_synthesis.md), [`plans/experimental_model_research.md`](plans/experimental_model_research.md) |
| 2026-09-07 | Experimental-model portfolio | Adopted two established experiment families with common-protocol starting choices, explicit fallbacks, exclusions and custom-research/implementation handoffs. No local model training or test outcome informed the decision. | Subphase 4A **In progress**; custom-model planning is next; first Phase 5 research handoff available, data readiness still pending | [`project_objective/experimental_model_portfolio.md`](project_objective/experimental_model_portfolio.md), [`plans/experimental_model_research.md`](plans/experimental_model_research.md) |
| 2026-09-07 | Custom attention-model planning | Opened the 4A.4 feature plan with four staged research subphases and a source-to-decision flow; no custom topology or implementation is selected. | Subphase 4A **In progress**; 4A.4.1 problem/evidence synthesis is next | [`plans/custom_attention_model.md`](plans/custom_attention_model.md), [`project_objective/experimental_model_portfolio.md`](project_objective/experimental_model_portfolio.md) |
| 2026-09-08 | Custom attention-model research | Completed the problem brief and component evidence collection, including source corrections and negative evidence; the next project action is compatible design synthesis. | Subphase 4A **In progress**; custom topology and implementation remain open | [Research collection](research/topics/custom_attention_model_design/README.md), [custom feature plan](plans/custom_attention_model.md) |
| 2026-09-08 | Custom attention-model design synthesis | Recorded compatible tensor/initialization routes, a bounded scaling envelope and reproducible scalar resource estimates; final topology comparison is now the next action. | Subphase 4A **In progress**; no custom topology selected or model executed | [Compatibility synthesis](research/topics/custom_attention_model_design/architecture_compatibility_synthesis.md), [custom feature plan](plans/custom_attention_model.md) |
| 2026-09-08 | Experimental-model research completion | Completed the three-topology comparison and adopted the custom P2-S design alongside the established pair; all three research handoffs include explicit implementation/resource gates. | Subphase 4A **Done**; Macro-section 4 **In progress** for independent 4B; Phase 5 remains **Deferred** for Data readiness | [Portfolio](project_objective/experimental_model_portfolio.md), [three proposals](research/topics/custom_attention_model_design/topology_proposals.md), [completed 4A plan](plans/experimental_model_research.md) |

| 2026-09-14 | Results-comparison preparation | Audited all 200 target-v5 trial configurations, CSV/TensorBoard inventories and relevant Optuna studies; demonstrated local W&B histogram extraction for one session per family and inspected selected checkpoints. Recorded persistence semantics and analysis limits without training or final evaluation. | Preparatory feasibility audit **Done**; Macro-section 7 final comparison remains **Deferred** | [Artifact contract and audit](implementation_details/experiment_artifacts.md), [reproducible probe](../src_scratches/experiment_comparison_audit/README.md) |

| 2026-09-14 | Results-comparison planning | Opened the operational plan for optional Lightning-model ingredient logging and local N-experiment JSON/HTML analysis, using recorded parameter histograms and preserving audited comparison limits. | Work package 7.1 **Pending**; final comparison remains **Deferred** | [Feature plan](plans/experiment_comparison.md) |
| 2026-09-14 | Experiment observability and comparison tooling | Implemented optional epoch-level ingredient precision/recall/F1 in the Lightning model and the local N-experiment comparator; validated all 200 target-v5 trials, resumed TensorBoard histories, selected W&B parameter trajectories and checkpoint metadata. | Work package 7.1 **Done**; final comparison remains **Deferred** | [Implementation contract](implementation_details/experiment_comparison.md), [completed plan](plans/experiment_comparison.md) |

| 2026-09-15 | Reference-selector research | Completed R1 by applying the mandatory gates to the R0 inventory and retaining three distinct protocol-level finalists: supervised ResNet-50 full fine-tuning, supervised EfficientNetV2-S full fine-tuning, and frozen DINOv2 B/14-register linear transfer. Exclusions are grouped by redundancy, unresolved prerequisites, disproportionate cost, or measurement confounding; no `M_ref` was selected and no candidate was trained. | Subphase 4B **In progress**; R2 decision-relevant verification and choice are next; Macro-section 3 remains **Deferred** | [R1 checkpoint](plans/reference_selector_research.md#r1-completion-checkpoint--2026-09-15) |
| 2026-09-15 | Reference-selector research | Completed R2 and selected supervised EfficientNetV2-S full fine-tuning with an independent 165-logit pooled head as `M_ref`. The source audit and bounded synthetic smoke verified the direct maintained-library route, finite output/head gradients, and 384-pixel FP32 batch-8 operation within the 8 GB boundary; no candidate training, AP, selected-vocabulary result, or test evidence informed the choice. | Subphase 4B **In progress**; R3 exact protocol freeze and Macro-section 3 handoff are next; Macro-section 3 remains **Deferred** | [R2 checkpoint](plans/reference_selector_research.md#r2-completion-checkpoint--2026-09-15) |
| 2026-09-15 | Reference-selector freeze and Phase 3 handoff | Completed R3 and froze 4B-D1: exact EfficientNetV2-S weights and supervised prior, full adaptation, full-frame 384-pixel transform, independent head, weighted BCE, FP32 batch-8 target, interpretation limits, and required provenance. Macro-section 3 accepted the handoff at P1; no implementation or label outcome informed the freeze. | Subphase 4B and Macro-section 4 **Done**; Macro-section 3 and Work package 3.2 **In progress** | [4B-D1](project_objective/model_comparison_methodology.md#4b-d1--frozen-reference-selector-protocol), [R3 checkpoint](plans/reference_selector_research.md#r3-completion-checkpoint--2026-09-15), [Phase 3 plan](plans/recognizable_ingredient_selection.md) |
| 2026-09-23 | Historical experiment comparison | Reviewed the existing basic_v5 ResNet and DINOv2 campaigns with the maintained comparator. ResNet18 trial 77 is the stronger observed validation artifact, while the frozen DINOv2 linear-probe boundary and missing final-benchmark evidence remain explicit. | Historical result **Done**; Macro-section 7 final comparison remains **Deferred** | [Reviewed experiment result](experiment_results/basic_v5_resnet_dinov2.md) |
| 2026-09-24 | Ingredient-selection campaign freeze | Completed P1 and adopted Phase 3-D1 without inspecting a new selector outcome. The single seed-42 AdamW/warm-up/cosine campaign, 20-epoch budget, deterministic two-epoch AP/F1 audits, bootstrap/control boundary, blind 24-label pilot, one-run P3/P4 reuse, and output manifest are binding. | Work package 3.2 **Done**; Work package 3.3 **Pending** at P2 | [Phase 3-D1](project_objective/model_comparison_methodology.md#phase-3-d1--frozen-selector-campaign-and-measurement-protocol), [Phase 3 plan](plans/recognizable_ingredient_selection.md) |
| 2026-09-24 | Ingredient-selection implementation | Completed P2 without inspecting selector outcomes. Added the exact selector and fit/pad transform, train/validation-only data boundary, weighted loss and frozen schedule, fixed-state audits, AP/F1/bootstrap analysis, pilot/rule hash gate, manifests and historical regression. All 64 tests pass; the real batch-8 FP32 gate used 3,632.20 MiB allocated and 5,362.00 MiB reserved on the 8 GB RTX 4060. | Work package 3.3 **In progress** at P3 | [Implementation contract](implementation_details/ingredient_selection.md), [Phase 3 plan](plans/recognizable_ingredient_selection.md) |
| 2026-09-27 | Ingredient-selection execution amendment | Adopted Phase 3-D2 by user request: interrupted v1 before per-label inspection, retained its artifacts, returned execution to the main workspace, and requested effective batch 128 with measured physical capacity and exact Lightning accumulation. Quick CUDA trials reject 128/64/32/16; physical 8 with accumulation 16 passes. A full-epoch gate and fresh source snapshot are mandatory before v2. | Work package 3.3 **In progress** at P3 | [Phase 3-D2](project_objective/model_comparison_methodology.md#phase-3-d2--effective-batch-and-main-workspace-execution-amendment), [implementation](implementation_details/ingredient_selection.md) |
| 2026-09-27 | Ingredient-selection budget amendment | Adopted Phase 3-D3 by user request before v2 campaign launch: 40 epochs with 2 warm-up and 38 cosine epochs, unchanged two-epoch audits and relocated final windows. The incomplete disposable v2 gate was interrupted; v3 requires a fresh full-epoch gate against the committed sources. | Work packages 3.2 **Done**, 3.3 **In progress** at P3 | [Phase 3-D3](project_objective/model_comparison_methodology.md#phase-3-d3--forty-epoch-campaign-amendment), [Phase 3 plan](plans/recognizable_ingredient_selection.md) |
| 2026-09-27 | Ingredient-selection campaign launch | The v3 full-epoch training gate passed with 375 CUDA updates, and the fresh 40-epoch effective-batch-128 campaign started at 16:17 UTC from revision `192059e`. Manifest, gate and all 152 source-snapshot entries were verified without per-label outcome inspection. | Work package 3.3 **In progress**; pilot analysis and gate freeze pending | [Launch checkpoint](plans/recognizable_ingredient_selection.md#p3-resource-gate-and-launch-checkpoint--2026-09-27), [verification and device limitation](implementation_details/ingredient_selection.md#verified-resource-and-test-evidence) |
| 2026-09-28 | Ingredient-selection P3 completion | The same v3 campaign completed 40 epochs. Blind analysis of only 24 pilot labels froze the absolute D4 rule with bootstrap overlap handling and preserved pilot artifacts; no non-pilot label outcome or test split was inspected. | P3 **Done**; Work package 3.3 **In progress**, P4 **Deferred** pending separate authorization | [D4 decision](project_objective/model_comparison_methodology.md#phase-3-d4--pilot-frozen-numerical-profile-rule), [pilot result](experiment_results/phase3_d1_v3_pilot.md), [feature-plan checkpoint](plans/recognizable_ingredient_selection.md#p3-pilot-and-numerical-rule-completion--2026-09-28) |
| 2026-09-28 | Ingredient-selection P4 full profile | After separate authorization, applied the unchanged D4 gates to all 165 labels from the same v3 campaign; verified pilot parity, deterministic full report and figures, and no test access. The 25 numerical candidates remain subject to relevance and observability review. | P4 and Work package 3.3 **Done**; 3.4/P5 **Deferred** | [P4 result](experiment_results/phase3_d1_v3_full_profile.md), [feature-plan checkpoint](plans/recognizable_ingredient_selection.md#p4-full-profile-completion--2026-09-28) |
| 2026-09-29 | Ingredient-selection P5 pilot preparation | Adopted the separate semantic/visual rubric, generated a deterministic 64-pair blind validation packet for two human reviewers, and implemented agreement scoring. No reviewer result or selected vocabulary exists yet. | Work package 3.4/P5 **In progress** | [P5 protocol](project_objective/ingredient_observability_protocol.md), [feature-plan checkpoint](plans/recognizable_ingredient_selection.md#p5-preparation-and-pilot-packet--2026-09-29) |
| 2026-10-04 | Ingredient-selection scope amendment | Adopted D5: manual relevance and visual-observability reviews are outside model-learnability selection and retained as an optional future-results appendix. D4 gates and P4 evidence are unchanged; no human annotation or final vocabulary is claimed. | Work package 3.4/P5 **Superseded**; 3.5/P6 **Pending** without reviewer dependencies | [D5 decision](project_objective/model_comparison_methodology.md#phase-3-d5--numerical-selection-and-optional-interpretation-appendix), [optional appendix](project_objective/ingredient_observability_protocol.md), [active plan](plans/recognizable_ingredient_selection.md) |
| 2026-10-05 | Ingredient-inclusion policy review | Reviewed the user's objective-alignment concern without assuming the vocabulary must grow. Retained original D4/P4 results and documented gate effects, uncertainty mismatch, sensitivity and a concrete proposed amendment; no replacement vocabulary is frozen. | Work packages 3.2–3.3 **In progress** (reopened); 3.5 **Pending** | [Reviewed audit](experiment_results/phase3_d1_v3_inclusion_policy_audit.md), [methodology proposal](project_objective/model_comparison_methodology.md#post-p4-inclusion-policy-review--proposed-amendment), [active plan](plans/recognizable_ingredient_selection.md) |
| 2026-10-05 | Ingredient-inclusion amendment and corrected analysis | Adopted user-approved D6 and completed the separate saved-score profile with paired image-cluster uncertainty, independent diagnostics, fixed sensitivity and exact rerun verification. Original D4 evidence and full-vocabulary default remain unchanged; no final projection is exported. | Work packages 3.2–3.3 **Done**; 3.5 **Pending** | [D6 decision](project_objective/model_comparison_methodology.md#phase-3-d6--held-out-quality-inclusion-policy), [reviewed result](experiment_results/phase3_d1_v3_d6_profile.md), [active plan](plans/recognizable_ingredient_selection.md) |
| 2026-10-05 | Shared ingredient-vocabulary freeze | Published the versioned D6 selected projection with saved original indices, independent excluded/uncertain reasons and verified evidence/source hashes. Regeneration is exact; full vocabulary and split metadata remain unchanged. Runtime integration and retention/parity checks remain. | Work package 3.5 **In progress**; Ingredient selection **In progress** | [Published definition](../src/ingredient_selection/resources/ingredients_selected_v5_d6_v1.json), [reviewed publication](experiment_results/phase3_d1_v3_d6_profile.md#p6-publication--2026-10-05), [active plan](plans/recognizable_ingredient_selection.md) |
| 2026-10-06 | Shared-vocabulary runtime integration | Completed P7 opt-in configuration, fixed selected order, checkpoint guards and analysis projection. All 140 tests, real train/validation parity and historical retention/restore checks pass. Historical scripts are deprecated for new work but retained unchanged; no physical cleanup or new campaign. | Work package 3.5 and Ingredient selection **Done**; independent Data 2.4 gates remain | [P7 checkpoint](plans/recognizable_ingredient_selection.md#p7-integration-and-retirement-checkpoint--2026-10-06), [runtime and retention contract](implementation_details/ingredient_selection.md#p7-runtime-projection) |
| 2026-10-06 | Data runtime completion | Closed 2.4 with four real CUDA updates, exact full-checkpoint reload and actual dashboard/browser smoke verification. Model-specific dashboard preprocessing is preserved; 149 repository tests pass. Metadata, user edits and retained experiments remain unchanged. | Work package 2.4 and Data **Done**; Additional model implementation **Pending** | [Runtime contract and evidence](implementation_details/image_data_loading.md), [completed Data plan](plans/data_ingredient_refactor/yummly_data_phase.md) |
| 2026-10-06 | Additional-model planning | Created the operational Phase 5 plan from the completed portfolio and a read-only runtime/design survey. Five first-level packages cover shared foundations, the established pair, P2-S and bounded measured qualification; the selector and scientific decisions remain unchanged. | Macro-section 5 and Work packages 5.1–5.5 **Pending**; plan ready, 5.1 next | [Implementation plan](plans/additional_model_implementation.md), [binding portfolio](project_objective/experimental_model_portfolio.md) |
| 2026-10-06 | Shared experimental-model foundations | Completed versioned 224 preprocessing/identity, deterministic head initialization, offline-construction checks, exact batch/tail arithmetic and declared engineering policy; 28 focused and 177 repository tests pass. Existing interfaces, selector and evidence remain unchanged; no CUDA qualification or campaign. | Work package 5.1 **Done**; Macro-section 5 **In progress**; 5.2 next | [Foundation contract](implementation_details/experimental_model_contract.md), [feature checkpoint](plans/additional_model_implementation.md#51-completion-checkpoint--2026-10-06) |
| 2026-10-06 | Experimental EfficientNet implementation | Completed the distinct 224 adapter, full/frozen state, exact experimental Lightning and strict full/light offline/class-order persistence; 57 focused/206 repository tests pass. Legacy/selector/data stay unchanged; actual artifact/real-consumer/CUDA acceptance remains. | Work package 5.2 **Done**; Macro-section 5 **In progress**; 5.3 next | [Runtime contract](implementation_details/experimental_model_contract.md#experimental-efficientnet-and-canonical-runtime--52), [feature checkpoint](plans/additional_model_implementation.md#52-completion-checkpoint--2026-10-06) |
| 2026-10-06 | Experimental MaxViT implementation | Completed whole-readout replacement with intact pretrained state, explicit normalization/geometry, original-artifact verification and strict offline persistence; CPU diagnostics and regressions pass. Actual CUDA/real-consumer acceptance remains. | Work package 5.3 **Done**; Macro-section 5 **In progress**; 5.4 next | [Runtime and artifact contract](implementation_details/experimental_model_contract.md#experimental-maxvit-and-artifact-provenance--53), [feature checkpoint](plans/additional_model_implementation.md#53-completion-checkpoint--2026-10-06) |
| 2026-10-06 | Custom model implementation | Adopted the user-approved DICA-Net name for historical P2 and implemented the S tensor graph with numerical attention, both-path gradients and label-row invariants; 234 repository tests pass. Custom persistence/integration and resource qualification remain. | Work package 5.4 **In progress**; Macro-section 5 **In progress** | [Tensor-core contract](implementation_details/experimental_model_contract.md#dica-net-s-tensor-core--541), [custom execution plan](plans/additional_model_implementation.md#54--dica-net-s-custom-implementation), [naming amendment](project_objective/experimental_model_portfolio.md#rationale-and-falsifiable-claim) |
| 2026-10-07 | Selected-task baseline launcher preparation | Prepared rerunnable ResNet18/DINOv2 historical-configuration transfers to the shared 59-label D6 vocabulary, with fresh initialization, provenance and explicit resume guards. No training, HPO or predictive test evaluation was started; the user will launch the runs while Phase 5 proceeds. | Launcher preparation **Done**; final Macro-section 6 gate remains **Deferred** | [Launcher contract](implementation_details/selected_vocabulary_training.md), [commands](../scripts/launch_exps/selected_ingredients/README.md) |

## Tracker maintenance rules

1. Update this file when a feature plan is completed, or earlier when work changes a project-level status, priority, dependency, scope, completion gate, or material blocker. Track ordinary implementation progress in the target feature plan.
2. Keep the project overview, macro-section status, work-package tables, checklists, and history log synchronized.
3. Never delete completed or superseded work solely because the project moved to a later phase.
4. Check a task only when its durable artifact or verification evidence exists.
5. Add links to implementation, manifests, reports, research folders, experiments, plots, and thesis material close to the relevant task.
6. Explain every **Deferred**, **Blocked**, or **Superseded** state and name its resume or replacement condition.
7. Do not mark a macro-section **Done** while its completion gate is unmet.
8. When reopening completed work, retain the old completion event in the history log and add a new transition.
9. Refresh `**Last updated:**`, `**Overall status:**`, `**Current macro-phase:**`, and `**Current focus:**` whenever project priorities change.

## Related documents

- [`project_objective/problem_definition.md`](project_objective/problem_definition.md) defines the research question and success criteria.
- [`project_objective/yummly_data_audit.md`](project_objective/yummly_data_audit.md) contains the evidence behind the current data priorities.
- [`project_objective/ingredient_vocabulary_audit.md`](project_objective/ingredient_vocabulary_audit.md) contains the evidence and provisional classifications for the candidate ingredient vocabulary.
- [`research/topics/ingredient_vocabularies/README.md`](research/topics/ingredient_vocabularies/README.md) indexes the reusable, dataset-independent vocabulary catalog.
- [`plans/data_ingredient_refactor/controlled_vocabulary_evaluation.md`](plans/data_ingredient_refactor/controlled_vocabulary_evaluation.md) records the Yummly-specific controlled-vocabulary evidence and implementation decision gate.
- [`project_objective/benchmark_decisions.md`](project_objective/benchmark_decisions.md) contains the binding benchmark policies and readiness checklist.
- [`project_objective/model_comparison_methodology.md`](project_objective/model_comparison_methodology.md) owns the binding methodology for shared vocabulary selection, model comparison, random-reduction controls, and local adaptation.
- [`project_objective/experimental_model_portfolio.md`](project_objective/experimental_model_portfolio.md) owns the selected established pair and custom topology, protocol choices, rationale, and unverified implementation gates.
- [`plans/data_ingredient_refactor/yummly_data_phase.md`](plans/data_ingredient_refactor/yummly_data_phase.md) is the active implementation plan for the Data work packages summarized in this section.
- [`plans/recognizable_ingredient_selection.md`](plans/recognizable_ingredient_selection.md) is the completed implementation plan for Macro-section 3 and the maintained home of the historical discrepancy resolutions.
- [`plans/experimental_model_research.md`](plans/experimental_model_research.md) is the operational plan for Subphase 4A broad discovery, candidate deep research, selection of two established families, and the separately planned custom attention-model research.
- [`plans/custom_attention_model.md`](plans/custom_attention_model.md) retains the completed 4A.4 research subphases, three-proposal decision and custom implementation handoff.
- [`plans/additional_model_implementation.md`](plans/additional_model_implementation.md) is the operational Phase 5 plan for the three adopted experimental models, their shared contract and engineering acceptance gates.
- [`plans/reference_selector_research.md`](plans/reference_selector_research.md) is the operational research and decision plan for Subphase 4B.
- [`research/topics/label_learnability/learnability_assessment.md`](research/topics/label_learnability/learnability_assessment.md) provides the reusable evidence behind the Phase 3 decision-profile framework.
- [`research/README.md`](research/README.md) defines where model discovery and topic research must be stored.
- [`implementation_details/models.md`](implementation_details/models.md) describes the model implementations currently available.
