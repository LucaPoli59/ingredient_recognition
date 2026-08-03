# Implementation plans

**Created:** 2026-08-02  
**Last updated:** 2026-08-02

This directory contains the execution plans for concrete project implementations. These plans translate work packages from the project-wide [`general_plan.md`](../general_plan.md) into bounded technical tasks, verification steps, dependencies, and completion criteria.

## Active plans

- [`yummly_data_phase.md`](yummly_data_phase.md) covers Data Work packages 2.1b–2.4: the shared image store, immutable legacy-experiment compatibility, improved `ingredients_target` standardization, deterministic exact-duplicate-safe splitting, and runtime integration.

## Scope

Use this directory when work is ready to move from project-level planning into an actual implementation. Examples include the deterministic benchmark builder, target normalizer, grouped split allocator, evaluation suite, a new model integration, or a training campaign.

Do not use implementation plans to redefine the thesis objective, binding benchmark decisions, or project-wide priorities. Those belong respectively in [`project_objective/`](../project_objective/README.md) and [`general_plan.md`](../general_plan.md).

## Structure

Use one descriptive Markdown file per bounded implementation:

```text
docs/plans/
├── README.md
└── <implementation_name>.md
```

If an implementation requires several durable supporting documents, create a directory instead:

```text
docs/plans/<implementation_name>/
├── README.md
└── <supporting_file>.md
```

The local `README.md` then acts as the implementation plan and indexes its supporting files.

## Required plan content

Every implementation plan must include:

- creation and last-updated dates;
- the linked macro-section and work package in [`general_plan.md`](../general_plan.md);
- objective, scope, and explicit non-goals;
- current status using the status vocabulary defined in the general plan;
- a progress tracker covering every implementation task;
- dependencies, assumptions, and unresolved decisions;
- affected components and expected artifacts;
- ordered implementation tasks;
- tests, validation, and completion criteria;
- a concise decision or change log when the plan evolves.

## Progress tracker

Every implementation plan must contain a `Progress tracker` section near the beginning of the document. It must show the implementation's overall status, current task, and next action, followed by a task-level table using the same status vocabulary as [`general_plan.md`](../general_plan.md).

Use this minimum structure:

```markdown
## Progress tracker

**Overall status:** Pending
**Current task:** Not started
**Next action:** <next concrete action>

| # | Task | Status | Evidence or result |
| --- | --- | --- | --- |
| 1 | <task> | **Pending** | — |
```

Update this tracker in the same change that starts or finishes a task, changes its status, or produces its evidence. Add tasks when newly discovered work is necessary; retain completed and superseded rows so the tracker preserves the implementation history. A plan may be marked **Done** only when all mandatory tasks and its completion criteria are satisfied.

## Synchronization rule

Read [`general_plan.md`](../general_plan.md) before creating or executing an implementation plan. Update the general plan in the same change whenever implementation work starts, completes, becomes blocked, is deferred, is reopened, or is superseded. Keep project-level status and history in the general plan; keep detailed task progress, execution state, and technical decisions in the implementation plan's progress tracker.

When an implementation plan is complete, retain it as project history and mark it **Done**. Do not delete completed or superseded plans.
