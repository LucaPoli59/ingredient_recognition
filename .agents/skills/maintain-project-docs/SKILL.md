---
name: maintain-project-docs
description: Store, update, move, review, or retire long-term knowledge in the Ingredient Recognition repository while following its documentation governance. Use when an agent needs to create or edit files under docs/, update README_PROJECT_KNOWLEDGE.md, record research or a binding decision, document current implementation behavior or a technical concept, maintain a feature plan or general_plan.md, reorganize documentation, preserve superseded knowledge, or decide where a durable finding belongs.
---

# Maintain Project Documentation

## Purpose and authority

Preserve durable project knowledge in one authoritative location, with enough evidence and context for a future agent to understand and verify it without conversation history.

Treat the repository documents as authoritative and this skill as their operational guide. Resolve authority by information type:

1. `docs/README_DOCS_ORGN.md` for storage, ownership, lifecycle, and cross-category rules.
2. `docs/README.md` for common writing, language, naming, dating, and maintenance conventions.
3. The nearest category or collection `README.md` for local structure.
4. `docs/project_objective/` for the research scope and binding methodological decisions.
5. `docs/general_plan.md` for project-level status, priorities, dependencies, gates, and history.
6. The relevant feature plan for active implementation state and task sequence.
7. Code and tests for observed runtime behavior, with `docs/implementation_details/` synchronized to the supported contract.
8. `README_PROJECT_KNOWLEDGE.md` for the stable repository-wide map, never as a replacement for the owners above.

If this skill conflicts with a live authoritative source, follow that source and update the skill when the task authorizes it.

## Read before acting

Locate the repository root before resolving any paths. Then:

1. Read `docs/general_plan.md` completely before project work so the relevant macro-section, first-level work package, status, gate, and next action are known.
2. Read `docs/README_DOCS_ORGN.md` and `docs/README.md` before creating, moving, or substantially revising documentation.
3. Read [references/documentation-categories.md](references/documentation-categories.md) before deciding where durable information belongs or how a category-specific document should be structured.
4. Read the nearest directory `README.md`, the existing authoritative document, and any linked active feature plan before editing.
5. Inspect the relevant code, configuration, tests, data version, experiment output, or primary external sources before recording a factual claim.
6. Read `README_PROJECT_KNOWLEDGE.md` when the task concerns stable repository architecture, canonical workflows, active defaults, or documentation entry points.

Do not reconstruct a rule, status, or conclusion from memory when its authoritative file is available.

## Store durable knowledge

### 1. Classify the information

Classify each item before writing:

- temporary exploration or raw output;
- research evidence;
- binding research or benchmark decision;
- verified current implementation contract;
- architecture-specific model explanation;
- reusable technical explanation;
- feature execution state;
- project-level status or history;
- stable repository-wide knowledge;
- documentation governance.

Separate mixed material into the appropriate authoritative records and cross-link them. Do not force evidence, decisions, implementation state, and project status into one document.

### 2. Find the existing owner

Search the relevant documentation and nearest indexes before creating a file. Update the existing authoritative record when it already owns the subject. Create a new document only when it has a distinct durable scope that cannot be expressed clearly in an existing record.

When records conflict:

1. Determine whether the difference is historical, planned, uncertain, or an actual defect.
2. Correct the responsible source of truth rather than one downstream mention.
3. Synchronize affected plans or trackers at the checkpoint required below.
4. Preserve superseded history and link its replacement.

### 3. Choose one durable home

Use the category playbooks in [references/documentation-categories.md](references/documentation-categories.md). Prefer the most specific applicable category. Link to an authoritative statement instead of copying it into multiple files.

Keep scratch scripts, notebooks, raw reports, logs, and intermediate analysis outside `docs/`, normally under `src_scratches/`. Promote only reviewed, interpretable findings with reproducible provenance.

### 4. Write for future verification

Write documentation under `docs/` in clear technical English. Preserve code identifiers and quoted external terminology exactly.

For every substantive record:

- state purpose and scope;
- distinguish verified facts, external evidence, interpretation, recommendations, assumptions, planned work, and unresolved questions;
- identify the relevant code, data or metadata generation, experiment, script, source, cutoff date, and verification method;
- explain limitations and consequences;
- link related authoritative code and documents;
- add `**Created:** YYYY-MM-DD` and `**Last updated:** YYYY-MM-DD` immediately below the title;
- keep the creation date unchanged and update the last-updated date on substantive edits.

Never describe a plan as implemented, a research recommendation as adopted, or an inference as a verified fact.

### 5. Synchronize only the affected records

Apply these checkpoint rules:

- Update the nearest index `README.md` in the same change when adding, moving, renaming, or changing the role of a document.
- Update implementation details in the same change as the code or contract they describe.
- Update a feature plan when an implementation step is completed. Record its result, evidence, decisions, newly discovered work, resulting statuses, and next action. Do not edit the plan merely to narrate intermediate progress within a step.
- Update `docs/general_plan.md` when the feature plan completes, or earlier only for a material project-level change to status, priority, dependency, scope, completion gate, next action, or blocker.
- Keep `general_plan.md` limited to macro-sections (`X`) and first-level work packages (`X.y`). Put all lower-level tasks and checklists in feature plans.
- Append project-level and first-level transitions to the general-plan history. Retain completed, deferred, superseded, and rejected work.
- Update `README_PROJECT_KNOWLEDGE.md` when stable repository architecture, functional behavior, canonical paths, active defaults, or documentation entry points materially change. Do not use it as the current progress tracker.

Avoid unrelated documentation churn. A correct synchronization set is better than touching every file that mentions the general topic.

## Create, edit, move, or retire documents

### Create

1. Confirm no current document owns the information.
2. Choose the category and category-specific outline.
3. Use a lowercase descriptive filename; use `snake_case` for multi-word directories.
4. Add the required title, dates, purpose, scope, evidence, limitations, and links.
5. Add or update the collection `README.md` and any required tracker links.

### Edit

1. Revalidate claims against their current evidence.
2. Preserve the original creation date.
3. Refresh the last-updated date.
4. Keep terminology and links consistent with the source of truth.
5. Normalize a legacy equivalent date header when making a substantive edit; do not create header-only churn.

### Move or rename

1. Confirm the new category owns the content.
2. Preserve Git history when practical.
3. Update the old and new nearest indexes, all inbound relative links, and any tracker or repository-knowledge references.
4. Check that links resolve from each linking file, not from the repository root.

### Supersede or retire

1. Keep historically relevant content.
2. Mark its status and date explicitly.
3. Explain why it was replaced or rejected.
4. Link the replacement or deciding evidence.
5. Remove a document only when it is duplicated, generated, temporary, or factually unsafe to retain and its durable value has been preserved elsewhere.

## Verify the documentation change

Before completion:

1. Confirm the chosen category and source-of-truth owner.
2. Confirm English, filename, directory, headings, dates, and Markdown structure.
3. Check factual claims against code, tests, data, experiments, or primary sources.
4. Resolve every relative Markdown link from the file that contains it.
5. Confirm nearest indexes and required trackers are synchronized at the correct checkpoint.
6. Confirm current, historical, planned, uncertain, and superseded statements are distinguishable.
7. Review the diff for duplicated authority, unrelated rewrites, and accidental loss of history.
8. Confirm no credentials, private absolute paths, large generated outputs, checkpoints, raw logs, or environment dumps entered durable documentation.
9. Run relevant documentation checks when available and use `git diff --check` for formatting errors.

Report which durable record now owns the information, which indexes or trackers changed, and what verification was performed.

## Maintain this skill

Treat this skill as a derived, project-scoped execution guide. When `docs/README_DOCS_ORGN.md`, `docs/README.md`, a category `README.md`, or tracker policy changes materially, compare the new rule with this skill and update the affected instructions and reference in the same documentation-maintenance change when authorized. Never let this skill become a competing source of truth.
