# Project progress and next steps

**Created:** 2026-08-02  
**Last updated:** 2026-08-02  
**Overall status:** In progress  
**Current macro-phase:** Data
**Current focus:** Build and freeze a reproducible Yummly benchmark.

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
| 2 | Data | **In progress** | Implement the deterministic benchmark schemas and builder. |
| 3 | Ingredient selection | **In progress** | Formalize relevance and visual-distinguishability criteria. |
| 4 | Model research | **In progress** | Complete systematic discovery after the benchmark definition is stable. |
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
- [x] Fixed the policies that govern target generation, ontology, observability, grouping, splitting, vocabulary, evaluation, calibration, and thresholds.

### Completion gate

The research problem, documentation method, decision authority, and project-tracking mechanism are explicit and linked. This gate is satisfied.

### Maintenance action

Reopen this section only when the thesis objective, scope, or binding methodological decisions change. Record the change in the history log.

## 2. Data

**Status:** In progress

The Data macro-section covers source understanding, target reconstruction, image quality, duplicate families, benchmark construction, splitting, and integration with the training pipeline.

### Work-package status

| Work package | Status | Next action |
| --- | --- | --- |
| 2.1 Yummly exploratory audit | **Done** | Re-run against each benchmark candidate. |
| 2.2 Benchmark artifact schemas | **Pending** | Specify manifests, tables, identifiers, and versioning. |
| 2.3 Deterministic benchmark builder | **Pending** | Implement the build from all 66,615 source records. |
| 2.4 Target normalization and ontology | **Pending** | Convert audited errors and aliases into tested rules. |
| 2.5 Image and recipe-family adjudication | **In progress** | Create manifests and review remaining candidates. |
| 2.6 Grouped split and benchmark freeze | **Pending** | Wait for targets, exclusions, and families to be frozen. |
| 2.7 Data-loader integration | **Pending** | Define the loader contract after schemas are fixed. |

### 2.1 Yummly exploratory audit

**Status:** Done

#### Completed

- [x] Audited all 65,146 processed metadata records and images.
- [x] Reconstructed the historical target-generation behavior and tested exact reproducibility.
- [x] Quantified substring collisions, alias fragmentation, support, cuisine shortcuts, duplicate images, and split contamination.
- [x] Computed SHA-256, dHash, and pHash evidence for all processed images.
- [x] Confirmed eight invalid-image groups covering 84 records.
- [x] Saved reproducible scripts, structured outputs, review tables, and visual artifacts.
- [x] Consolidated permanent findings and removed temporary research notes.

#### Evidence

- [`project_objective/yummly_data_audit.md`](project_objective/yummly_data_audit.md)
- [`../src_scratches/data_anlysis/README.md`](../src_scratches/data_anlysis/README.md)

#### Completion gate

The legacy dataset's known defects are quantified sufficiently to design a replacement benchmark. This gate is satisfied; it does not imply complete manual adjudication of every source record.

### 2.2 Benchmark artifact schemas

**Status:** Pending

#### Required work

- [ ] Define stable identifiers and versioned schemas for source records and images.
- [ ] Define the normalized-ingredient and mapping-review table.
- [ ] Define the ontology and alias-rule format.
- [ ] Define image-review and recipe-family edge manifests.
- [ ] Define split, vocabulary, benchmark-record, and validation-report schemas.
- [ ] Specify the output directory layout and artifact lifecycle.

#### Completion gate

Every planned benchmark artifact has a documented schema, owner, version, relationships, and validation rules.

#### Next action

Write the schema specification before implementing transformation logic.

### 2.3 Deterministic benchmark builder

**Status:** Pending

#### Required work

- [ ] Implement one deterministic build starting from all 66,615 source records.
- [ ] Generate a byte-level checksum manifest for input metadata and images.
- [ ] Record the code version, configuration, rule versions, environment, and input/output checksums.
- [ ] Remove unordered execution and manually edited intermediates from benchmark semantics.
- [ ] Add schema, count, referential-integrity, uniqueness, and deterministic-rerun checks.
- [ ] Generate a machine-readable validation report for each version.

#### Required outputs

1. source manifest;
2. normalized ingredient table;
3. versioned ontology;
4. image-review manifest;
5. recipe-family manifest;
6. grouped split manifest;
7. final benchmark metadata;
8. validation report.

#### Completion gate

Two clean builds from identical inputs and configuration produce byte-identical semantic artifacts and matching checksums.

#### Next action

Begin after Work package 2.2 fixes the artifact contracts.

### 2.4 Target normalization and ontology

**Status:** Pending

#### Required work

- [ ] Implement a boundary-aware normalizer over original ingredient lines.
- [ ] Encode transformations as ordered, versioned, testable rules.
- [ ] Add regression tests for every confirmed legacy substring collision.
- [ ] Apply only semantically justified aliases and preserve documented non-merges.
- [ ] Keep unresolved mappings in review data without creating an output `<UNK>` class.
- [ ] Audit mapping precision on frequent, rare, ambiguous, and cuisine-specific samples.
- [ ] Calculate recipe-level support after normalization.

#### Completion gate

The normalizer, mapping rules, ontology, manual audit, and change log are versioned; reruns produce identical candidate targets.

#### Next action

Convert confirmed collision and alias evidence into initial rules and regression fixtures.

### 2.5 Image and recipe-family adjudication

**Status:** In progress

#### Completed

- [x] Identified exact-image, perceptual-image, exact-ingredient-list, and high-similarity recipe candidates.
- [x] Confirmed eight invalid-image groups covering 84 records.
- [x] Demonstrated that candidate grouping connects 438 validation and 413 test records to training records under the legacy split.

#### Pending

- [ ] Convert confirmed exclusions into a machine-readable manifest.
- [ ] Define reviewer states, decision reasons, evidence, and adjudication rules.
- [ ] Review remaining duplicate candidates and suspicious singleton images.
- [ ] Build family components from accepted evidence edges.
- [ ] Review large, heterogeneous, or ambiguous connected components.
- [ ] Freeze record-to-family assignments.

#### Completion gate

Every exclusion and accepted family edge is traceable to evidence, and final components pass size and heterogeneity review.

#### Next action

Create the review and family-edge schemas, then import the eight confirmed exclusions as the first reviewed entries.

### 2.6 Grouped split and benchmark freeze

**Status:** Pending

#### Dependencies

- Work packages 2.3 and 2.4 provide reproducible targets.
- Work package 2.5 provides frozen exclusions and family components.
- Macro-section 3 provides the final ingredient-selection policy.

#### Required work

- [ ] Assign whole recipe families to train, validation, or test with a deterministic 80/10/10 procedure.
- [ ] Balance record count, cuisine distribution, and label support without splitting a family.
- [ ] Fit vocabulary-dependent processing on training data only after assignment.
- [ ] Verify that no accepted family, exact image, or normalized recipe duplicate crosses a split boundary.
- [ ] Confirm the required per-label support in every split.
- [ ] Freeze the benchmark version and keep test unavailable to selection decisions.

#### Completion gate

All leakage, support, and reproducibility assertions pass; every retained record has one stable split; and the benchmark-readiness checklist is complete.

#### Next action

Prototype the allocator with synthetic components without freezing a real split before upstream decisions are complete.

### 2.7 Data-loader integration

**Status:** Pending

#### Required work

- [ ] Update the data-loading path to consume benchmark manifests and the frozen vocabulary.
- [ ] Remove `<UNK>` from the output space.
- [ ] Preserve target, split, exclusion, family, and source identifiers through evaluation.
- [ ] Use aspect-ratio-preserving and comparable transforms across model families.
- [ ] Validate class weights and sampling logic using training data only.
- [ ] Add a deterministic end-to-end data-loading test.

#### Completion gate

A clean environment can load every split, reproduce tensors and transforms, and trace each sample to its benchmark artifacts.

#### Next action

Design the loader contract after Work package 2.2 fixes the artifact schemas.

### Data macro-section completion gate

The Data macro-section is **Done** only when the benchmark is frozen, all readiness checks pass, and the training pipeline consumes it reproducibly.

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
| 4.3 State-of-the-art discovery | **Pending** | Start a dated discovery after reviewing prior discovery folders. |
| 4.4 Focused model topics | **Pending** | Create one topic folder per selected research question. |
| 4.5 Candidate shortlist | **Pending** | Compare hypotheses, expected value, compute, and implementation cost. |

### Completed

- [x] Documented existing ResNet, DenseNet, DINOv2, and dummy-model implementations.
- [x] Produced a technical deep dive for DINOv2 ViT-B/14.
- [x] Defined the repository structure for dated discovery and topic-focused research.
- [x] Formalized model-relevant challenges: partial observability, correlated labels, long-tail support, contextual shortcuts, calibration, and interpretability.

### Pending

- [ ] Perform a broad, dated state-of-the-art discovery.
- [ ] Review at least the two preceding discoveries before every new discovery when they exist.
- [ ] Investigate focused topics such as visual-language pretraining, multilabel dependency modeling, open-vocabulary recognition, and partial-label learning when supported by discovery evidence.
- [ ] Compare candidates on scientific fit, data requirements, compute, calibration, interpretability, and integration cost.
- [ ] Produce a bounded shortlist with an explicit hypothesis for each proposed model.

### Completion gate

The research record supports a prioritized shortlist of models, and every candidate has a falsifiable hypothesis, baseline comparison, resource estimate, and implementation plan.

### Next action

Begin the first dated discovery once the benchmark task definition and ingredient-selection protocol are stable enough to evaluate model relevance.

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
- [ ] Verify Optuna study isolation, resumption behavior, and trial provenance.
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
- [x] Required multiple seeds and recipe-family bootstrap confidence intervals.

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

## Tracker maintenance rules

1. Update this file in the same change that starts, completes, defers, blocks, supersedes, or reopens tracked work.
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
- [`research/README.md`](research/README.md) defines where model discovery and topic research must be stored.
- [`implementation_details/models.md`](implementation_details/models.md) describes the model implementations currently available.
