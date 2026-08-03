# General project plan

**Created:** 2026-08-02  
**Last updated:** 2026-08-03
**Overall status:** In progress  
**Current macro-phase:** Data
**Current focus:** Implement the shared image store while preserving legacy metadata, configurations, and checkpoints through in-memory compatibility.

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
| 2 | Data | **In progress** | Complete the shared image store and legacy compatibility layer, then generate deterministic `ingredients_target` metadata. |
| 3 | Ingredient selection | **In progress** | Formalize relevance and visual-distinguishability criteria. |
| 4 | Model research | **In progress** | Convert the broad discovery into focused topic records and an approved bounded shortlist. |
| 5 | Additional model implementation | **Deferred** | Resume after the research shortlist and model hypotheses are approved. |
| 6 | Training and hyperparameter tuning | **Deferred** | Resume after the benchmark, selected ingredients, and model contracts are frozen. |
| 7 | Results comparison | **Deferred** | Resume after comparable benchmark runs are complete. |
| 8 | Thesis writing | **Pending** | Define the thesis outline and map project evidence to chapters. |

## 1. Project foundation

**Status:** Done

### Work packages

| Work package | Status | Evidence |
| --- | --- | --- |
| Repository and pipeline understanding | **Done** | [`../README_PROJECT_KNOWLEDGE.md`](../README_PROJECT_KNOWLEDGE.md) |
| Documentation conventions and structure | **Done** | [`README.md`](README.md) |
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

**Status:** In progress

The Data macro-section covers source understanding, shared image storage, historical compatibility, deterministic target standardization, split generation, and runtime integration. Persistent outputs are intentionally limited to the common image collection and one selected metadata file in each split.

### Work-package status

| Work package | Status | Next action |
| --- | --- | --- |
| 2.1 Yummly exploratory audit | **Done** | Re-run only when a new metadata generation needs comparison. |
| 2.1b Shared image store and metadata decoupling | **In progress** | Implement loader path separation and a verified image-layout migration. |
| 2.1c Historical experiment compatibility | **Pending** | Add in-memory adapters and validate representative saved experiments without rewriting them. |
| 2.2 Improved `ingredients_target` standardization | **Pending** | Define the exact preprocessing rules and implement the new deterministic script. |
| 2.3 Deterministic metadata generation and split | **Pending** | Generate an exact-SHA-aware 80/10/10 split after target rules are approved. |
| 2.4 Runtime target integration and `<UNK>` decision | **Pending** | Change the new default and investigate `<UNK>` before modifying encoder behavior. |

### 2.1 Yummly exploratory audit

**Status:** Done

#### Completed

- [x] Audited all 65,146 processed metadata records and images.
- [x] Reconstructed the historical target-generation behavior and tested exact reproducibility.
- [x] Quantified substring collisions, alias fragmentation, support, cuisine shortcuts, duplicate images, and split contamination.
- [x] Computed exact and perceptual duplicate evidence for analysis.
- [x] Determined that attempt 1 is the probable lineage of the flat `ingredients_ok` targets and that attempt 2 is a different nested-category experiment.
- [x] Inventoried historical configurations and checkpoints under `experiments/basic`.
- [x] Consolidated permanent findings and removed temporary research notes.

#### Evidence

- [`project_objective/yummly_data_audit.md`](project_objective/yummly_data_audit.md)
- [`plans/yummly_data_phase.md`](plans/yummly_data_phase.md)
- [`../src_scratches/data_anlysis/README.md`](../src_scratches/data_anlysis/README.md)

#### Completion gate

The legacy dataset and its consumers are understood sufficiently to implement the approved replacement pipeline. This gate is satisfied.

### 2.1b Shared image store and metadata decoupling

**Status:** In progress

#### Purpose

Move image storage out of the split directories. Train, validation, and test retain same-named metadata generations, while every generation resolves its relative `image` references against `data/input/yummly/imgs/standard/`.

#### Required work

- [x] Inspect the current layout, metadata generations, loader path construction, dashboard consumer, and experiment storage.
- [x] Define the separate metadata-root and image-root contract.
- [x] Write the detailed implementation plan for Data Work packages 2.1b–2.4.
- [ ] Add loader contract tests using multiple metadata generations and one image directory.
- [ ] Add a relative `images_subdir` configuration with `imgs/standard` as its default.
- [ ] Refactor the DataModule and other image consumers to resolve images independently of split metadata.
- [ ] Implement a dry-run-first image migration that verifies filenames, counts, and SHA-256 values.
- [ ] Validate all current metadata generations before retiring split-local image copies.

#### Evidence

- [`plans/yummly_data_phase.md`](plans/yummly_data_phase.md)

#### Completion gate

All Yummly consumers resolve images from the shared collection, both existing metadata generations load successfully, migration checks pass, and obsolete split-local copies can be retired safely. No legacy metadata, configuration, or checkpoint is rewritten.

#### Next action

Add path-contract tests before changing the on-disk layout.

### 2.1c Historical experiment compatibility

**Status:** Pending

#### Purpose

Keep prior experiments loadable after the image-layout and future target-default changes without changing their saved semantics.

#### Required work

- [ ] Preserve `metadata.json` and `sel_ing_2410_metadata.json` unchanged, including `ingredients_ok`.
- [ ] Supply the new shared image root in memory when a legacy configuration does not contain it.
- [ ] Preserve every explicitly saved `feature_label="ingredients_ok"` value.
- [ ] Translate the known older DenseNet configuration keys into current DataModule arguments in memory.
- [ ] Add a read-only validation script for the known historical schemas.
- [ ] Smoke-test representative JSON-driven, light-checkpoint, and full-checkpoint experiments.
- [ ] Confirm that metadata, configurations, checkpoints, label encoders, output dimensions, and `<UNK>` behavior remain byte- and semantically unchanged.

#### Completion gate

Every recognized experiment under `experiments/basic` can load its unchanged legacy metadata through the shared image layout. Unknown schemas fail explicitly instead of being guessed or rewritten.

#### Next action

Implement the compatibility adapter together with the 2.1b loader refactor, then run the read-only inventory and smoke tests.

### 2.2 Improved `ingredients_target` standardization

**Status:** Pending

#### Purpose

Create a new deterministic script that derives `ingredients_target` from the original `ingredients` lines for new metadata generations.

#### Required work

- [ ] Use `prev_attempts/attempt1/preprocessing_v2.py` as forensic input, not as an executable dependency.
- [ ] Define the normalization, support, alias, generalization, and record-retention rules before implementation.
- [ ] Replace unbounded substring edits with token- or phrase-bounded rules.
- [ ] Count support by distinct recipes rather than distinct raw strings.
- [ ] Replace similarity-based and unordered merges with explicit deterministic rules.
- [ ] Preserve `ingredients` and write an ordered, duplicate-free `ingredients_target` list.
- [ ] Add regression tests for confirmed legacy collisions and every retained historical rule.
- [ ] Emit concise aggregate generation statistics without creating per-line mapping or review artifacts.

#### Completion gate

Identical inputs and rules produce identical targets and ordering; known collision cases pass regression tests; and the preprocessing effect is understood before split generation.

#### Next action

Discuss and freeze the exact standardization rules, support threshold, minimum target count, and desired level of ingredient generalization.

### 2.3 Deterministic metadata generation and split

**Status:** Pending

#### Required work

- [ ] Generate `ingredients_target` from the approved Work package 2.2 rules.
- [ ] Apply automatic image existence and decoding checks only.
- [ ] Compute SHA-256 during generation and keep byte-identical images in the same allocation group.
- [ ] Do not create groups from perceptual similarity, recipe names, ingredient similarity, or manual review.
- [ ] Assign exact-image groups to a deterministic 80/10/10 train/validation/test split.
- [ ] Balance cuisine and target distributions within documented tolerances.
- [ ] Write the same selected metadata filename under all three split directories.
- [ ] Enforce uniqueness, referential integrity, split ratio, distribution, exact-leakage, and deterministic-rerun assertions.

#### Completion gate

The three metadata files are reproducible, pass every assertion, contain valid `ingredients_target` lists, and have no byte-identical image crossing split boundaries. No auxiliary split, family, vocabulary, review, or validation manifest is required.

#### Next action

Implement only after the target standardization rules in Work package 2.2 are approved.

### 2.4 Runtime target integration and `<UNK>` decision

**Status:** Pending

#### Required work

- [ ] Keep `feature_label` configurable and change only its default to `ingredients_target` for new configurations.
- [ ] Derive the vocabulary deterministically from training metadata and preserve it with each experiment rather than in a standalone data artifact.
- [ ] Investigate how filtered vocabularies, validation/test-only labels, cuisine filters, multi-label encoders, and sequence encoders use `<UNK>`.
- [ ] Distinguish ingestion or sequence-token behavior from a trainable multi-label output class.
- [ ] Decide and test the new behavior before changing an encoder.
- [ ] Preserve all historical `<UNK>` behavior for legacy experiments.
- [ ] Update statistics and dashboard consumers, then run minimal training, reload, and visualization smoke tests.

#### Completion gate

New experiments default to `ingredients_target`, alternative feature fields remain supported, historical experiments preserve their semantics, and the `<UNK>` policy is explicit and covered by tests.

#### Next action

Begin after the new metadata generation exists; the investigation may start earlier but must not alter legacy artifacts.

### Data macro-section completion gate

The Data macro-section is **Done** only when shared image loading, legacy compatibility, deterministic target generation, exact-duplicate-safe splitting, runtime integration, and the `<UNK>` decision all pass their completion gates.

## 3. Ingredient selection

**Status:** In progress

This macro-section selects ingredients that are scientifically relevant and sufficiently distinguishable for meaningful image-based evaluation. Relevance and visual distinguishability are separate criteria and must not be collapsed into raw frequency.

### Work-package status

| Work package | Status | Next action |
| --- | --- | --- |
| 3.1 Preliminary frequency and quality evidence | **Done** | Use it to design the selection protocol. |
| 3.2 Relevance criteria | **Pending** | Define semantic and support requirements. |
| 3.3 Visual-distinguishability protocol | **Pending** | Define instance-level annotation and agreement rules. |
| 3.4 Candidate ingredient analysis | **Pending** | Apply the criteria without using test outcomes. |
| 3.5 Final vocabulary tiers | **Pending** | Freeze headline and exploratory vocabularies after the grouped split. |

### Completed evidence

- [x] Quantified legacy label support and long-tail behavior.
- [x] Identified corrupted frequent labels and obvious alias fragmentation.
- [x] Demonstrated cuisine-dependent contextual shortcuts.
- [x] Defined the observability states `direct`, `contextual`, `not_inferable`, and `uncertain`.
- [x] Set preliminary support requirements for reliable headline evaluation.

### Pending

- [ ] Define what “relevant” means for the thesis question independently of frequency.
- [ ] Define annotation instructions for visual distinguishability at recipe-label level.
- [ ] Select an audited subset with coverage across labels, cuisines, support tiers, and image conditions.
- [ ] Use at least two reviewers and measure agreement before adjudication.
- [ ] Estimate direct/contextual/not-inferable proportions per candidate ingredient.
- [ ] Combine semantic validity, support, observability, and research value into a documented selection rule.
- [ ] Freeze headline and exploratory vocabulary tiers using training support only.

### Completion gate

The final ingredient sets are versioned, justified by explicit criteria, supported by an annotation report, and frozen without test-driven selection.

### Next action

Write the relevance and visual-distinguishability protocol, then pilot it on a small stratified sample before large-scale annotation.

## 4. Model research

**Status:** In progress

This macro-section surveys candidate methods and turns them into testable architectural hypotheses for this specific dataset and objective.

### Work-package status

| Work package | Status | Next action |
| --- | --- | --- |
| 4.1 Existing-model inventory | **Done** | Maintain when implementation changes. |
| 4.2 DINOv2 deep dive | **Done** | Revisit only if its integration changes. |
| 4.3 State-of-the-art discovery | **Done** | Refresh only when a material research update justifies a new dated snapshot. |
| 4.4 Focused model topics | **Pending** | Open the prioritized target-processing, augmentation, representation/head, and calibration topics. |
| 4.5 Candidate shortlist | **Pending** | Convert focused evidence into the final approved implementation shortlist. |

### Completed

- [x] Documented existing ResNet, DenseNet, DINOv2, and dummy-model implementations.
- [x] Produced a technical deep dive for DINOv2 ViT-B/14.
- [x] Defined the repository structure for dated discovery and topic-focused research.
- [x] Formalized model-relevant challenges: partial observability, correlated labels, long-tail support, contextual shortcuts, calibration, and interpretability.
- [x] Completed a dated broad state-of-the-art discovery grounded in the repaired-benchmark objective and 8 GB compute constraint.

### Pending

- [ ] Review at least the two preceding discoveries before every new discovery when they exist.
- [ ] Investigate the discovery's prioritized topics: ingredient parsing and standardization, food-safe preprocessing and augmentation, representation/class-query implementation, and multi-label calibration.
- [ ] Compare candidates on scientific fit, data requirements, compute, calibration, interpretability, and integration cost.
- [ ] Produce a bounded shortlist with an explicit hypothesis for each proposed model.

### Completion gate

The research record supports a prioritized shortlist of models, and every candidate has a falsifiable hypothesis, baseline comparison, resource estimate, and implementation plan.

### Next action

Create focused topic records from the [2026-08-02 discovery](research/discovery/2026-08-02/README.md), starting with target standardization and data preprocessing; approve the final model shortlist only after the data contract is stable.

## 5. Additional model implementation

**Status:** Deferred

This macro-section covers architectures selected by Model research that are not already implemented in the repository.

### Existing foundation

- [x] Shared `BaseModel` abstraction and model-dependent transforms exist.
- [x] ResNet, DenseNet, DINOv2, and dummy models exist.
- [x] Lightning integration and experiment configuration are available.

Existing models are historical baselines, not evidence that the additional-model phase is complete.

### Pending after resume

- [ ] Write an implementation contract for each shortlisted model.
- [ ] Add the architecture using existing abstractions where appropriate.
- [ ] Add shape, forward-pass, serialization, checkpoint, and transform tests.
- [ ] Add launcher and configuration support without duplicating the training pipeline.
- [ ] Verify frozen and fine-tuned modes where relevant.
- [ ] Document architecture-specific assumptions, memory use, and limitations.
- [ ] Run a small overfit or smoke test before full training.

### Resume gate

Resume when Macro-section 4 approves at least one additional model and the DataModule contract needed by that model is stable.

### Completion gate

Every selected model passes its tests, integrates with the canonical training path, is documented, and can complete a reproducible small run.

### Next action

No implementation action until the first model shortlist is approved.

## 6. Training and hyperparameter tuning

**Status:** Deferred

This macro-section covers baseline training, controlled model training, hyperparameter search, run selection, and reproducibility under the frozen benchmark.

### Existing historical infrastructure

- [x] Lightning trainers, checkpointing, logging, and early stopping exist.
- [x] One-shot experiment launchers exist.
- [x] Optuna-based hyperparameter tuning exists.
- [x] CSV, TensorBoard, and offline W&B logging paths exist.
- [x] Historical experiments and plots exist for legacy data and configurations.

Historical runs remain useful engineering evidence but are **Superseded** for final comparison by the future benchmark protocol.

### Pending after resume

- [ ] Freeze seeds, budgets, stopping rules, transforms, metrics, and logging requirements.
- [ ] Run prevalence and cuisine-prior non-visual baselines.
- [ ] Run a simple supervised convolutional baseline.
- [ ] Run a frozen pretrained visual encoder with a linear multilabel head.
- [ ] Define model-specific hyperparameter spaces before opening each study.
- [ ] Verify Optuna study isolation, resumption behavior, and trial traceability.
- [ ] Run tuning using validation only and fixed comparable budgets.
- [ ] Retrain selected configurations across the required seeds.
- [ ] Calibrate and select thresholds using validation only after model selection.
- [ ] Preserve configurations, checkpoints, metrics, environment information, and run identifiers.

### Resume gate

Resume when the benchmark, selected ingredient vocabulary, data loader, evaluation implementation, and at least the required baseline models are ready.

### Completion gate

Every required baseline and shortlisted model has reproducible selected runs under the same benchmark protocol, with no test-set use during selection.

### Next action

Do not launch final training or tuning on the legacy 182-label split. Small smoke tests remain allowed when clearly marked as engineering validation.

## 7. Results comparison

**Status:** Deferred

This macro-section covers the frozen evaluation protocol, statistical comparison, qualitative analysis, and final interpretation of model behavior.

### Methodological decisions already completed

- [x] Selected label-macro mean average precision and micro F1 as paired primary metrics.
- [x] Defined secondary ranking, set-prediction, calibration, cardinality, and observability-slice metrics.
- [x] Required validation-only calibration and threshold selection.
- [x] Required multiple seeds and confidence intervals that do not treat byte-identical image groups as independent samples.

### Pending implementation and analysis

- [ ] Implement and unit-test the metric, calibration, threshold, aggregation, and bootstrap suite.
- [ ] Freeze the comparison table schema before inspecting test results.
- [ ] Compare models under identical data, vocabulary, transforms, budgets, seeds, and selection rules.
- [ ] Report primary and secondary metrics with uncertainty.
- [ ] Compare direct, contextual, not-inferable, and uncertain ingredient slices.
- [ ] Analyze performance by ingredient, support tier, cuisine, image quality, and predicted cardinality.
- [ ] Inspect representative successes, false positives, false negatives, and shortcut behavior.
- [ ] Report compute, memory, training time, and inference cost.
- [ ] Evaluate selected final configurations on test exactly once.
- [ ] State which hypotheses are supported, rejected, or inconclusive.

### Resume gate

Resume comparative analysis after Macro-section 6 produces comparable selected runs. Evaluation code may be implemented and tested earlier without final test access.

### Completion gate

The comparison is reproducible, statistically supported, includes failure analysis and resource costs, and directly answers the research questions without overstating ingredient visibility.

### Next action

Implement metric-level tests with small hand-verifiable multilabel examples while the data benchmark is being built.

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
project foundation [Done]
          |
          v
       data [In progress] <------+
          |                       |
          +--> ingredient selection [In progress]
          |                       |
          +--> model research [In progress]
                        |         |
                        v         |
          additional models [Deferred]
                        |         |
                        +---------+
                        v
        training and HTuning [Deferred]
                        |
                        v
          results comparison [Deferred]
                        |
                        v
              thesis completion

Thesis outline and stable chapters may progress in parallel.
```

## Project history log

This table is append-only. Add one row when a macro-section or significant work package changes status, a major artifact is completed, or an earlier result is superseded. Do not delete historical rows when plans change.

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
| 2026-08-02 | Data planning | The shared image store was introduced as prerequisite 2.1b, and a codebase-grounded implementation plan was created for Data Work packages 2.1b–2.7. | Work package 2.1b **In progress** | [`plans/yummly_data_phase.md`](plans/yummly_data_phase.md) |
| 2026-08-02 | Data planning | The initial manifest-heavy benchmark design was superseded by a smaller pipeline: immutable legacy artifacts, in-memory experiment compatibility, deterministic `ingredients_target` generation, exact-SHA grouping only, and split metadata as the source of truth. | Work packages 2.1b–2.4 **In progress/Pending** | [`plans/yummly_data_phase.md`](plans/yummly_data_phase.md), [`project_objective/benchmark_decisions.md`](project_objective/benchmark_decisions.md) |

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
- [`project_objective/benchmark_decisions.md`](project_objective/benchmark_decisions.md) contains the binding benchmark policies and readiness checklist.
- [`plans/yummly_data_phase.md`](plans/yummly_data_phase.md) is the active implementation plan for Data Work packages 2.1b–2.4.
- [`research/README.md`](research/README.md) defines where model discovery and topic research must be stored.
- [`implementation_details/models.md`](implementation_details/models.md) describes the model implementations currently available.
