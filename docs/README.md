# Project documentation

**Created:** 2026-08-02  
**Last updated:** 2026-08-02

This directory contains the durable technical documentation for the Ingredient Recognition project. It complements the repository-level [`README_PROJECT_KNOWLEDGE.md`](../README_PROJECT_KNOWLEDGE.md), which provides a concise map of the project and its current state.

## Language

All documentation in `docs/` must be written in English. This includes titles, prose, table headings, diagram labels, captions, and dates. Code identifiers, paths, configuration keys, and quoted external terminology must retain their original spelling.

When an existing document is translated, preserve its technical meaning, Markdown structure, links, formulas, code samples, and level of detail. Do not turn translation into an opportunity to introduce unverified claims or silently change a documented decision.

## Directory structure

```text
docs/
├── README.md
├── general_plan.md
├── plans/
│   ├── README.md
│   └── <implementation_name>.md
├── implementation_details/
│   └── <component>.md
├── models_deepdive/
│   ├── README.md
│   └── <model_name>.md
├── project_objective/
│   ├── README.md
│   └── <files>.md
├── research/
│   ├── README.md
│   ├── topics/
│   │   └── <topic>/
│   │       ├── README.md
│   │       └── <files>.md
│   └── discovery/
│       └── <discovery_date>/
│           ├── README.md
│           └── <files>.md
└── technical_details/
    ├── README.md
    └── <area>/<problem_title>/explaination.md
```

### `implementation_details/`

Use this directory for documentation tied directly to the repository implementation: component contracts, integration points, supported variants, configuration behavior, and operational constraints. These documents should explain what the current code does and link to the relevant source files.

### `models_deepdive/`

Use this directory for research-oriented explanations of model architectures. A deep dive may cover internal representations, tensor shapes, architectural blocks, pretraining objectives, downstream adaptation, trade-offs, and implications for ingredient recognition. Follow the additional conventions in [`models_deepdive/README.md`](models_deepdive/README.md).

### `project_objective/`

Use this directory to formalize the research project's problem, objectives, scope, constraints, assumptions, and success criteria before conducting discovery research. Follow the local conventions in [`project_objective/README.md`](project_objective/README.md).

### `plans/`

Use this directory for execution plans tied to concrete implementations. Each plan translates a work package from the general project plan into technical tasks, dependencies, validation steps, completion criteria, and a mandatory task-level progress tracker. Follow the local conventions in [`plans/README.md`](plans/README.md).

### `technical_details/`

Use this directory for durable analyses of cross-cutting technical problems or concepts, such as gradient behavior, data-processing constraints, library limitations, or interpretability assumptions. These notes explain the phenomenon, its causes, and its implications; they are not patch instructions or changelogs. Follow the additional conventions in [`technical_details/README.md`](technical_details/README.md).

### `research/`

Use this directory for research records that support future project decisions. Topic-focused investigations belong in [`research/topics/`](research/topics/README.md); broad state-of-the-art scans belong in [`research/discovery/`](research/discovery/README.md). Follow the local conventions in those directories.

## Project roadmap

[`general_plan.md`](general_plan.md) is the permanent progress tracker for the whole thesis project. It is organized into macro-sections for project foundation, data, ingredient selection, model research, additional model implementation, training and hyperparameter tuning, result comparison, and thesis writing. It preserves completed and superseded work in an append-only history log and must be updated whenever tracked work changes state. Detailed plans for concrete implementations belong in [`plans/`](plans/README.md).

## Writing methodology

Before writing or updating a document:

1. Inspect the current implementation and configuration. Documentation must describe verified repository behavior, not assumptions based only on class or file names.
2. Define the document's scope and place it in the most specific directory above. Link to related documents instead of duplicating their content.
3. Separate verified facts, external research findings, and open questions. Mark uncertainty explicitly and never present planned behavior as implemented behavior.
4. Prefer primary sources for research claims: original papers, official repositories, and official library documentation. Add links close to the claims they support.
5. Use precise names, tensor shapes, units, defaults, and paths where they aid verification. Keep code snippets minimal and ensure they match the current codebase.
6. Explain consequences and trade-offs, not only mechanics. For model documentation, connect architectural choices to training, inference, memory use, metrics, or interpretability where relevant.
7. Review internal links, headings, formulas, tables, and code references after editing. Confirm that the document remains understandable without relying on undocumented conversation context.
8. Add the document creation date when creating a new document, and update its last-modified date whenever the document is changed.

## Style conventions

- Use clear technical English and concise, descriptive headings.
- Use sentence case for headings.
- Use repository-relative paths in prose and links.
- Wrap code symbols, paths, shapes, configuration values, and commands in backticks.
- Use fenced code blocks with a language identifier when applicable.
- Define abbreviations on first use unless they are universally understood in context.
- Use ISO dates (`YYYY-MM-DD`) for newly created or substantially revised documents.
- Place `**Created:** YYYY-MM-DD` and `**Last updated:** YYYY-MM-DD` immediately below the document title. The creation date must remain unchanged; the last-updated date must be refreshed with every modification.
- Use lowercase descriptive filenames; use `snake_case` for multi-word directory names where the existing structure calls for it.
- Keep links relative for repository files and use stable, authoritative URLs for external sources.

## Required document content

Every substantial document should make the following clear, either explicitly or through unambiguous sections:

- its purpose and scope;
- the implementation, model, or technical question it covers;
- the evidence or sources on which it is based;
- relevant limitations, assumptions, or unresolved questions;
- links to related code and documentation.
- its creation date and the date of its most recent modification.

Model deep dives and technical notes may define more specific required content in their local `README.md` files.

## Maintenance workflow

Update documentation in the same change as the behavior or decision it describes. When adding, moving, or renaming a document, update the nearest directory index and any reference in [`README_PROJECT_KNOWLEDGE.md`](../README_PROJECT_KNOWLEDGE.md).

Before considering a documentation change complete, verify that:

- the content is in English;
- technical statements agree with the current code or cite a primary source;
- relative links resolve from the document's location;
- terminology is consistent with the codebase;
- the creation date is present and unchanged, and the last-updated date reflects the current modification;
- no credentials, local secrets, or machine-specific private data are included.
