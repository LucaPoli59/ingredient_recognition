# Documentation category playbooks

Use this reference after reading the live `docs/README_DOCS_ORGN.md`, `docs/README.md`, and the relevant local `README.md`. The live repository documents remain authoritative.

## Contents

1. [Common conventions](#common-conventions)
2. [Documentation entry point](#documentation-entry-point)
3. [Documentation governance](#documentation-governance)
4. [General project plan](#general-project-plan)
5. [Project objective](#project-objective)
6. [Topic research](#topic-research)
7. [Discovery research](#discovery-research)
8. [Implementation details](#implementation-details)
9. [Model deep dives](#model-deep-dives)
10. [Technical details](#technical-details)
11. [Implementation plans](#implementation-plans)
12. [Index READMEs and collection folders](#index-readmes-and-collection-folders)
13. [Repository knowledge](#repository-knowledge)
14. [Scratch and generated material](#scratch-and-generated-material)
15. [Synchronization matrix](#synchronization-matrix)
16. [Compact document outlines](#compact-document-outlines)

## Common conventions

Apply these rules to every durable document under `docs/`:

- Write titles, prose, headings, tables, captions, diagrams, and dates in English.
- Preserve identifiers, paths, configuration keys, commands, and quoted terminology in their original spelling.
- Use Markdown, descriptive lowercase filenames, `snake_case` multi-word directories, and ISO dates.
- Put `**Created:** YYYY-MM-DD` and `**Last updated:** YYYY-MM-DD` immediately below the title.
- Keep the creation date stable. Refresh the last-updated date for every substantive change.
- Use relative links for repository files and stable primary URLs for external evidence.
- Give every substantial record a clear purpose, scope, evidence or provenance, assumptions, limitations, unresolved questions when applicable, and related links.
- Prefer one authoritative statement plus links over duplicated summaries that can drift.
- Describe verified current behavior separately from historical behavior, planned changes, research recommendations, and inferences.
- Preserve meaningful history. Mark obsolete material `Superseded` and link its replacement.

Use the general-plan status vocabulary consistently: `Done`, `In progress`, `Pending`, `Deferred`, `Blocked`, and `Superseded`.

## Documentation entry point

**Location:** `docs/README.md`

**Purpose:** Provide the entry point to the documentation system, the common language and style rules, the high-level directory tree, and concise navigation to every category.

**Organization:** Keep a short project-documentation introduction, language rules, current directory map, category summaries, common writing methodology, style rules, required content, and maintenance checklist.

**Maintenance:** Update it when a top-level documentation category, entry point, common convention, or directory responsibility changes. Keep category-specific detail in the category README or governance document.

**Do not store:** Detailed findings, implementation contracts, feature status, or duplicated category playbooks.

## Documentation governance

**Location:** `docs/README_DOCS_ORGN.md`

**Purpose:** Define the authoritative cross-category rules for long-term information storage, ownership, conflict resolution, lifecycle, retention, and documentation completion.

**Organization:** Maintain core principles, the directory-responsibility map, authority order, naming and structure rules, category requirements, the information lifecycle, evidence policy, and a completion checklist.

**Maintenance:** Change it when the documentation taxonomy, source-of-truth boundaries, tracker rules, or general storage policy changes. Synchronize affected category READMEs, `docs/README.md`, and this skill. Treat governance changes as rules, not project findings.

**Do not store:** Research results, benchmark outcomes, runtime contracts, or feature execution history.

## General project plan

**Location:** `docs/general_plan.md`

**Purpose:** Preserve the whole-thesis roadmap, current project-level state, dependencies, completion gates, next actions, evidence links, and append-only project history.

**Organization:** Use macro-sections `X` and first-level work packages `X.y` only. For each first-level package, record a concise purpose or outcome, status, dependencies when relevant, completion gate, next action, and links to its feature plan or durable evidence.

**Maintenance:** Read it completely before project work. Update it when a feature plan completes, or earlier only for a material project-level status, priority, dependency, scope, gate, next-action, or blocker change. Keep history append-only at macro or first-level granularity.

**Do not store:** Second-level or deeper work packages such as `X.ya`, `X.yb`, or `X.y.Z`; implementation checklists; raw evidence; detailed technical decisions already owned elsewhere.

## Project objective

**Location:** `docs/project_objective/`

**Purpose:** Define the thesis problem and research contract. Own the objective, motivation, inputs and outputs, scope, non-goals, constraints, assumptions, research questions, success criteria, binding benchmark decisions, and durable dataset or objective audits.

**Organization:** Keep one focused file per stable objective aspect. Use `README.md` as the directory index. Typical records include a problem definition, benchmark decisions, and reviewed audits that materially define the problem.

**Maintenance:** Update a binding decision explicitly when evidence changes it. Record status, rationale, affected version, and unresolved consequences. Link supporting research instead of copying it. Update the index whenever files or roles change.

**Do not store:** Temporary exploration, routine task progress, generic literature summaries, or model-specific implementation notes.

## Topic research

**Location:** `docs/research/topics/<topic>/`

**Purpose:** Preserve a focused, reusable investigation of one well-defined question or subject.

**Organization:** Use a lowercase descriptive `<topic>` directory with a mandatory `README.md`. State the research boundary, method, source cutoff, primary sources, findings, uncertainty, implications, and open questions. Keep the material dataset-independent when possible.

**Maintenance:** Add new evidence to the existing topic when it answers the same durable question. Update the topic README when files or their purposes change. Link project-specific measurements and adopted decisions to their owners instead of absorbing them into the general topic.

**Do not store:** Binding project decisions, current repository contracts, feature execution logs, or broad time-stamped surveys.

## Discovery research

**Location:** `docs/research/discovery/<discovery_date>/`

**Purpose:** Preserve a broad, dated state-of-the-art snapshot covering methods, benchmarks, tools, and emerging directions relevant to the current project objective.

**Organization:** Name the directory with an ISO date and include a mandatory `README.md` describing context, scope, method, cutoff, file map, synthesis, recommendations, and limitations. Split large discoveries into focused files such as sources, models, data, evaluation, and recommendations.

**Maintenance:** Before creating a discovery, read the current project-objective documents and at least the two most recent discovery directories when available. State what is reused, what materially changed, and why a new snapshot is justified. Keep old discoveries as dated history.

**Do not store:** Adopted benchmark policy, permanent implementation contracts, or duplicated unchanged findings.

## Implementation details

**Location:** `docs/implementation_details/`

**Purpose:** Document verified current code-facing contracts: interfaces, data contracts, configuration defaults, persistence formats, supported variants, invariants, integration points, and operational constraints.

**Organization:** Use one focused descriptive file per contract. Cite relevant source files and tests. Distinguish current defaults, supported alternatives, legacy behavior, and compatibility rules.

**Maintenance:** Update the document in the same change as the behavior or contract it describes. Recheck code and tests before editing. Add the file to `implementation_details/README.md` and link it from a plan or objective record when it affects a tracked gate or decision.

**Do not store:** Literature surveys, speculative future designs, task-by-task progress, or conceptual explanations that apply beyond the local implementation.

## Model deep dives

**Location:** `docs/models_deepdive/<model_name>.md`

**Purpose:** Explain a model architecture actually used by the repository in enough depth to understand representations, tensor flow, adaptation, and trade-offs.

**Organization:** Identify the exact model variant and local implementation. Cover tensor shapes and flow, principal blocks, pretraining or original objective, downstream multi-label adaptation, training and inference implications, compute constraints, interpretability limits, local code, and primary sources.

**Maintenance:** Revalidate the deep dive when the local model integration, supported variant, transform contract, or architecture-specific behavior changes. Keep the category README indexed.

**Do not store:** Generic beginner tutorials, cross-cutting system failures, patch instructions, or implementation progress.

## Technical details

**Location:** `docs/technical_details/<area>/<problem_title>/explaination.md`

**Purpose:** Preserve reusable reasoning about a cross-cutting technical concept, limitation, or problem that emerged in the project.

**Organization:** Use a domain `<area>` and descriptive `snake_case` `<problem_title>`. Explain the general conceptual, mathematical, architectural, or systems model first; then describe necessary conditions, causes, consequences, limits, and the local relevance. End with a concise synthesis.

**Maintenance:** Update when the explanation, assumptions, or known limits change. Keep case-specific code links as evidence, not as the main organizing principle. Update `technical_details/README.md` when adding or moving a note.

**Do not store:** How-to patch steps, changelogs, task trackers, or a copy of a feature plan.

The filename `explaination.md` is an established project convention despite its spelling. Preserve it unless the documentation governance is explicitly migrated.

## Implementation plans

**Location:** `docs/plans/<implementation_name>.md` or `docs/plans/<implementation_name>/`

**Purpose:** Translate one general-plan work package into bounded execution tasks, dependencies, decisions, validation, evidence, and completion criteria.

**Organization:** Use one file for a bounded implementation. Use a folder when durable supporting records are necessary; its `README.md` must identify exactly one operational plan and index every supporting file. Include objective, scope, non-goals, linked macro/work package, status, dependencies, assumptions, affected components, artifacts, ordered tasks, validation, completion criteria, and a concise decision/change log.

Place a `Progress tracker` near the beginning with overall status, current task, next action, and a task table. Use the general-plan status vocabulary.

**Maintenance:** Treat the active feature plan as the operational source of truth. Update it only at step-completion checkpoints, recording result, verification evidence, decisions, newly discovered work, statuses, and next action. Retain completed and superseded rows. Mark the plan `Done` only after all mandatory tasks and completion criteria pass. Synchronize the general plan at completion or earlier for a material project-level change.

**Do not store:** Thesis-scope decisions, broad research, repository contracts detached from an implementation, or lower-level detail in `general_plan.md`.

## Index READMEs and collection folders

**Locations:** Every top-level documentation category and every explicitly defined collection directory, including each topic, discovery date, and multi-file implementation plan.

**Purpose:** Explain what belongs in the directory and make its contents navigable.

**Organization:** State context, scope, boundaries, structure, current files with one-line roles, parent rules, and related authoritative records. Keep it lightweight.

**Maintenance:** Update the nearest index in the same change whenever a file is added, moved, renamed, removed, or assigned a different role. Add a collection README before adding multiple unindexed records.

**Do not store:** Detailed evidence, duplicated document bodies, or status better owned by a plan.

## Repository knowledge

**Location:** `README_PROJECT_KNOWLEDGE.md`

**Purpose:** Give agents and maintainers a stable, concise map of repository architecture, canonical workflows, data and model contracts, operational entry points, and links to authoritative documentation.

**Organization:** Describe durable repository behavior and navigation. Link `docs/general_plan.md` for current status and `docs/README_DOCS_ORGN.md` for storage rules. Keep detailed evidence and execution state in their owners.

**Maintenance:** Update it after material architectural or functional changes, canonical path/default changes, or new documentation entry points. Verify every statement against code or its authoritative document. Follow its established language unless the project explicitly changes that convention.

**Do not store:** Fine-grained task progress, a second general plan, raw audits, or duplicated implementation specifications.

## Scratch and generated material

**Typical location:** `src_scratches/` or the appropriate data/experiment output directory, outside the durable documentation system.

Use scratch space for scripts, notebooks, temporary notes, intermediate reports, exhaustive raw tables, and exploratory outputs. Keep strong interim findings there while research is active, then consolidate reviewed conclusions into the correct durable category.

Do not cite an unreviewed scratch artifact as the final project record. When promoting a finding, preserve enough provenance to reproduce it and link the script or stable artifact when useful. Do not put large datasets, logs, checkpoints, credentials, or machine-specific environment dumps in `docs/`.

## Synchronization matrix

| Event | Required durable updates |
| --- | --- |
| Add, move, rename, remove, or re-role a document | Document dates/content, nearest index, all affected relative links |
| Complete an implementation step | Feature plan tracker, evidence, decisions, statuses, next action |
| Complete a feature plan | Feature plan final state and `general_plan.md` first-level state/history |
| Material project-level change before feature completion | Feature plan and affected `general_plan.md` summary, package, gate/action, history |
| Change code-facing behavior or contract | Code/tests and owning `implementation_details/` record in the same change |
| Adopt or revise a binding methodology | Owning `project_objective/` decision record, affected plan, general plan if state/gate changes |
| Add research evidence without adopting it | Owning topic or discovery record and its nearest index |
| Change stable architecture, canonical workflow, path, or default | Owning implementation record and `README_PROJECT_KNOWLEDGE.md`; trackers only if state changes |
| Change documentation governance | `README_DOCS_ORGN.md`, affected category/entry READMEs, and this skill |
| Supersede a result or approach | Mark old record, state reason/date, link replacement, preserve relevant history |

## Compact document outlines

Use only the sections that materially help the record. Do not copy a template mechanically.

### Substantial record

```markdown
# <Title>

**Created:** YYYY-MM-DD
**Last updated:** YYYY-MM-DD

## Purpose and scope
## Context or question
## Evidence and method
## Findings or current contract
## Assumptions and limitations
## Open questions or next implications
## Related code and documentation
```

### Collection README

```markdown
# <Collection title>

**Created:** YYYY-MM-DD
**Last updated:** YYYY-MM-DD

## Context and scope
## What belongs here
## Structure
## Files
## Working and maintenance rules
## Related documentation
```

### Feature plan tracker

```markdown
## Progress tracker

**Overall status:** Pending
**Current task:** Not started
**Next action:** <next concrete action>

| # | Task | Status | Evidence or result |
| --- | --- | --- | --- |
| 1 | <bounded task> | **Pending** | — |
```

### Research record

```markdown
# <Research title>

**Created:** YYYY-MM-DD
**Last updated:** YYYY-MM-DD

## Research question and boundary
## Method and source cutoff
## Evidence
## Findings
## Uncertainty and limitations
## Project implications
## Open questions
## References
```
