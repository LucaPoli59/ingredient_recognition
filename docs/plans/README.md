# Implementation plans

**Created:** 2026-08-02  
**Last updated:** 2026-08-04

This directory contains the execution plans for concrete project implementations. These plans translate work packages from the project-wide [`general_plan.md`](../general_plan.md) into bounded technical tasks, verification steps, dependencies, and completion criteria.

## Active plans

- [`data_ingredient_refactor/`](data_ingredient_refactor/README.md) contains the active Yummly Data plan and its controlled-vocabulary evaluation for Work packages 2.1b–2.4.

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
├── <operational_plan>.md
└── <supporting_file>.md
```

The local `README.md` is the directory entry point. It either contains the implementation plan directly or identifies exactly one operational plan as the source of truth, and it indexes every durable supporting file.

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

Update this tracker when an implementation step is completed. At that checkpoint, record the result and verification evidence, capture any decisions or newly discovered work, update the resulting statuses, and identify the next action. Intermediate progress does not require a plan edit. Retain completed and superseded rows so the tracker preserves the implementation history. A plan may be marked **Done** only when all mandatory tasks and its completion criteria are satisfied.

## Common plan directives

The following directives apply to every implementation plan:

1. Read both [`general_plan.md`](../general_plan.md) and the target feature plan before starting implementation work.
2. Treat the target feature plan as the operational source of truth while the feature is being implemented.
3. Update the feature plan at step-completion checkpoints. When a step is completed, record its result, evidence, decisions and newly discovered work, then update the statuses and next action.
4. Update [`general_plan.md`](../general_plan.md) when the feature plan is completed. An earlier general-plan update is required only when implementation changes a project-level status, priority, dependency, scope, completion gate, or creates a material blocker.
5. Do not add excessive comments to implementation code. Add a code comment only when it is necessary to explain non-obvious intent, a constraint, a workaround, or a risk that the code itself cannot express clearly. This restriction does not limit the detail required in the implementation plan, its progress tracker, decisions, or verification evidence.
6. Preserve completed and superseded tasks as history instead of deleting them.

## Synchronization rule

Read [`general_plan.md`](../general_plan.md) before creating or executing an implementation plan. During implementation, keep detailed task progress, execution state, and technical decisions in the feature plan's progress tracker. Synchronize the completed result back to the general plan when the feature plan is finished, or earlier only when one of the project-level conditions in the common directives applies.

When an implementation plan is complete, retain it as project history and mark it **Done**. Do not delete completed or superseded plans.
