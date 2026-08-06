# Documentation organization directives

**Created:** 2026-08-06  
**Last updated:** 2026-08-06

This document is the authoritative guide for storing durable information and organizing documentation in the Ingredient Recognition project. Read it together with [`README.md`](README.md) before creating, moving, or substantially revising a document. Directory-specific READMEs add local detail and must remain consistent with these directives.

## 1. Core principles

1. **Store durable knowledge once.** Every long-term fact, decision, result, contract, or explanation has one authoritative home. Other documents link to it instead of copying it.
2. **Separate evidence, decisions, execution, and status.** A research observation is not automatically a project decision; an implementation plan is not proof that implementation exists; a progress tracker is not a technical specification.
3. **Describe the repository as it is.** Distinguish verified current behavior, historical behavior, planned work, external findings, assumptions, and unresolved questions. Never present a planned change as implemented.
4. **Preserve history.** Do not delete completed, superseded, or rejected work merely because the project moved on. Mark its status, explain the transition, and link the replacement or evidence.
5. **Keep documentation navigable.** Every directory has an entry-point README, every substantial document has a clear scope, and every move or rename updates its nearest index and affected links.
6. **Use English for documentation.** This includes prose, headings, tables, captions, diagrams, and dates. Preserve code identifiers, paths, configuration keys, and quoted terminology in their original spelling.

## 2. Directory responsibilities

The following map is the placement rule for long-term information. Choose the most specific applicable category and link across categories when a subject has multiple aspects.

| Location | Store here | Do not store here |
| --- | --- | --- |
| [`README.md`](README.md) | Documentation navigation, common writing methodology, and high-level directory map | Detailed findings, feature status, or duplicated technical specifications |
| `README_DOCS_ORGN.md` | Cross-category governance directives and long-term storage rules | Project results or implementation-specific decisions |
| [`general_plan.md`](general_plan.md) | Whole-thesis roadmap, macro-section/work-package status, dependencies, gates, next actions, and append-only project history; only macro-sections and first-level work packages (`X.y`) | Detailed implementation instructions, second-level or deeper work packages (`X.ya`, `X.yb`, `X.y.Z`), or a second copy of research evidence |
| [`project_objective/`](project_objective/README.md) | Research problem, objective, scope, non-goals, assumptions, success criteria, binding benchmark decisions, and durable data/objective audits | Temporary exploration, ordinary task progress, or model-specific implementation notes |
| [`research/topics/`](research/topics/README.md) | Focused, reusable research about one well-defined topic, independent of a single implementation when possible | Current repository contracts or decisions that have not been adopted through a project-objective document |
| [`research/discovery/`](research/discovery/README.md) | Date-stamped broad state-of-the-art snapshots, source catalogs, and initial recommendations | Permanent implementation contracts or a replacement for focused topic records |
| [`implementation_details/`](implementation_details) | Current code-facing contracts, supported variants, configuration behavior, integration points, and operational constraints | Broad literature surveys, speculative future behavior, or task-by-task execution logs |
| [`models_deepdive/`](models_deepdive/README.md) | Architecture-specific explanations grounded in the models used by the repository, including representations, tensor flow, pretraining, adaptation, and trade-offs | Cross-cutting bug analyses, patch instructions, or generic introductory tutorials |
| [`technical_details/`](technical_details/README.md) | Durable conceptual or cross-cutting technical explanations: causes, assumptions, formulas/shapes, consequences, and limits | How-to instructions, changelogs, or a copy of an implementation plan |
| [`plans/`](plans/README.md) | Bounded implementation execution plans, dependencies, ordered tasks, verification, completion criteria, and feature progress | Thesis scope decisions, broad research, or undocumented status changes |

`src_scratches/` is outside the durable documentation system. Use it for scripts, exploratory notebooks, temporary reports, and intermediate analysis. Promote important findings into the appropriate `docs/` category, link the reproducible script when useful, and do not cite an unreviewed scratch artifact as the final project record.

## 3. Authority and conflict resolution

Use the following source-of-truth boundaries:

1. `project_objective/` contains the research scope and binding methodological decisions. Research documents provide evidence; they become binding only when the decision is recorded there.
2. `general_plan.md` is authoritative for project-level status, priorities, dependencies, gates, and historical transitions.
3. A feature plan under `plans/` is authoritative for the execution state and task sequence of that feature while it is active.
4. The code is authoritative for observed runtime behavior. `implementation_details/` records the intended and supported contract and must be synchronized with the code.
5. `research/`, `models_deepdive/`, and `technical_details/` preserve evidence and explanations. They do not silently override a binding decision or the current implementation.

When two records disagree, do not hide the conflict by editing one claim in isolation. Identify whether the difference is historical, planned, or an actual defect; update the responsible source-of-truth document and the relevant plan or tracker in the same change; preserve the superseded record when it is part of project history.

## 4. Required structure and naming

- Use Markdown files with lowercase, descriptive filenames and `snake_case` for multi-word directories. Required project-level names such as `README.md`, `README_PROJECT_KNOWLEDGE.md`, `README_DOCS_ORGN.md`, and `general_plan.md` are exceptions.
- Use ISO dates (`YYYY-MM-DD`) for dated folders and document metadata. Discovery folders must use `docs/research/discovery/<discovery_date>/`; focused research must use `docs/research/topics/<topic>/`.
- Every top-level documentation category and every explicitly defined collection directory (for example, a research topic, dated discovery, or implementation-plan directory) must contain a `README.md` explaining its context, scope, file structure, and relationships. Update it whenever a file is added, moved, renamed, or its role changes.
- Keep repository links relative to the linking file. Use stable primary URLs for external sources. Do not embed machine-specific absolute paths in durable documentation.
- Put `**Created:** YYYY-MM-DD` and `**Last updated:** YYYY-MM-DD` immediately below every new document's title. Keep the creation date unchanged and refresh the last-updated date for every substantive edit. Existing documents using an older equivalent header should be normalized on their next substantive edit; do not make a header-only rewrite solely to create noise.

## 5. Content requirements by category

Every substantial document must state its purpose and scope, distinguish evidence from interpretation, identify assumptions and limitations, link related code or documents, and carry its creation and last-modified dates.

### Project objective

State the problem, motivation, inputs and outputs, scope, explicit non-goals, assumptions, constraints, research questions, success criteria, and unresolved questions. Keep binding decisions explicit, versioned where necessary, and separate from exploratory alternatives.

### General project plan

Organize the roadmap into macro-sections (`X`) and first-level work packages (`X.y`). For each first-level package, keep only a concise purpose or outcome, status, dependencies, completion gate, next action, and links to its operational plan or durable evidence. Do not add second-level or deeper work-package sections (`X.ya`, `X.yb`, `X.y.Z`) or their implementation checklists. Those details belong in the relevant feature plan under `plans/` or in the appropriate evidence document. The append-only history log preserves project-level and first-level transitions; it must not be used to reintroduce lower-level implementation detail.

### Topic research

Record one well-defined question or topic. Include the research boundary, search or comparison method, primary sources and cutoff date, findings, uncertainty, implications, and open questions. Keep the topic reusable; place dataset-specific measurements or adopted decisions in the linked project-objective or plan document.

### Discovery research

Record the discovery date, context, broad scope, sources, findings, recommendations, and limitations. Before creating a new discovery, inspect at least the two most recent discovery directories, state what is reused or updated, and avoid duplicating unchanged material. A discovery may propose directions, but adoption belongs in a decision record or plan.

### Implementation details

Describe verified current behavior: interfaces, data contracts, paths, configuration defaults, supported variants, invariants, operational constraints, and relevant source files/tests. Update the document in the same change as the behavior it describes. Mark legacy behavior and compatibility rules explicitly.

### Model deep dives

Identify the exact model variant and local implementation. Explain the tensor flow and shapes, principal blocks, pretraining objective, downstream adaptation, training/inference implications, interpretability limits, and primary sources. Keep architecture-specific reasoning here; place cross-cutting failures in `technical_details/`.

### Technical details

Explain a reusable phenomenon or problem before discussing the local case. Include the conceptual or mathematical model, necessary conditions, cause, consequences, interpretative limits, and a concise summary. These notes are explanatory records, not patch instructions or changelogs.

### Implementation plans

Link the plan to a macro-section and work package in `general_plan.md`. State objective, scope, non-goals, status, dependencies, assumptions, affected components, ordered tasks, validation, completion criteria, and a decision/change log. Include a progress tracker near the beginning. Update it at step-completion checkpoints: record the completed result and evidence, decisions or newly discovered work, resulting statuses, and next action. Intermediate progress does not require a plan edit. Synchronize `general_plan.md` when the feature is complete, or earlier only for a material project-level change.

### Index READMEs

Explain what belongs in the directory, show its structure, list the current files, and link to the applicable parent rules. Keep index READMEs lightweight and navigational; detailed evidence remains in the referenced documents.

## 6. Long-term information lifecycle

For every new finding, decision, or artifact:

1. **Classify it.** Decide whether it is temporary exploration, evidence, a binding decision, current implementation behavior, technical explanation, execution state, or historical context.
2. **Choose one durable home.** Use the directory map and source-of-truth boundaries above. Do not create a second document merely because another category links to the same subject.
3. **Record provenance.** State the dataset/code version, experiment or script, source, date/cutoff, and verification method needed to reproduce the claim.
4. **Separate certainty levels.** Label facts, inferences, recommendations, assumptions, deferred choices, and unresolved questions. Use status vocabulary consistently (`Done`, `In progress`, `Pending`, `Deferred`, `Blocked`, `Superseded`).
5. **Cross-link the record.** Update the nearest README and link the authoritative record from affected plans, objective documents, or repository knowledge when appropriate.
6. **Update synchronously.** Modify documentation in the same change as the code, data, or decision it describes. Update feature plans at completed-step checkpoints and the general plan when its project-level state changes.
7. **Verify before retaining.** Check links, dates, terminology, code paths, source quality, Markdown rendering, and the absence of secrets or private machine information.
8. **Preserve superseded knowledge.** Mark the old record as superseded or historical, explain why, and link the replacement. Never silently rewrite history to make the current approach appear to have always been used.

## 7. Evidence, references, and generated material

- Prefer primary sources for research claims: original papers, official repositories, official standards, and official library documentation. Put citations close to the claim and record access or release dates when they affect reproducibility.
- Link aggregate results to the script, input generation, configuration, and validation evidence that produced them. Do not place large generated datasets, logs, checkpoints, credentials, or environment dumps in `docs/`.
- Persist only reviewed, interpretable findings in durable documents. Keep raw or exploratory output in the appropriate data, experiment, or scratch location and promote a concise summary with provenance.
- Treat a document as a record of knowledge, not as a hidden task queue. Use `general_plan.md` and feature plans for status and next actions.

## 8. Completion checklist

Before considering a documentation change complete, confirm:

- the document is in the correct category and has one clear owner/source of truth;
- the language, filename, folder name, dates, headings, and Markdown structure follow the conventions;
- purpose, evidence/provenance, limitations, and related links are present;
- all relative links resolve and nearest indexes are updated;
- current, historical, planned, and uncertain statements are distinguishable;
- affected feature and general trackers are updated at the required checkpoint;
- superseded material was preserved rather than silently deleted; and
- no secrets, credentials, private absolute paths, or unreviewed bulk output were added.
