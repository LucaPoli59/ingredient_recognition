# Technical details

**Creation date:** 2026-07-29
**Last updated:** 2026-08-06

This folder collects in-depth technical notes on concepts, limitations and problems that emerged in the project. The goal is to preserve the machine learning, neural network, data processing, or systems reasoning behind a technical decision so that it is reusable and verifiable in the future.

These documents are not how-to guides for applying a patch or changelogs of code changes. They can cite a case from the repository as a reason, but they must first explain the phenomenon in general terms: relevant assumptions, representations, formulas or shapes, consequences and interpretative limits.

## Structure

Each note follows the path:

```text
docs/technical_details/<area>/<problem_title>/explaination.md
```

`<area>` identifies the technical domain, for example `dino`, `data`, `lightning` or `dashboard`. `<issue_title>` must be short, descriptive, and use `snake_case`.

Each new document must report its creation date immediately below the title, in the format:

```markdown
**Creation date:** YYYY-MM-DD
```

## Expected content

A note should include, when relevant:

- context and technical question;
- conceptual, mathematical or architectural model;
- cause of the phenomenon and conditions necessary for it to occur;
- implications for the project and limits of the explanation;
- a final summary.

The affected code can be cited as a reference, but detailed implementation instructions belong in the comments, pull requests, or functional documentation of the modules.

## Documents available

- [`dino/gradcam_frozen_vit_tokens/explaination.md`](dino/gradcam_frozen_vit_tokens/explaination.md): Differentiable interpretability, Grad-CAM and token-based representations in DINOv2.
- [`data/yummly_benchmark_split/explaination.md`](data/yummly_benchmark_split/explaination.md): Yummly-specific split allocation, validation guarantees, and evaluation limits.
