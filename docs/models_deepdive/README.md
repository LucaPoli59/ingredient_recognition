# Model deep dives

**Creation date:** 2026-07-29

This folder contains technical deep dives into the machine-learning architectures available in the project. Each document describes the model itself: internal representations, layers and forward flow, pretraining objectives when relevant, architectural choices, the role of the downstream head, and implications for the ingredient-recognition task.

The goal is to go beyond the overview in [`../implementation_details/models.md`](../implementation_details/models.md) without repeating introductory neural-network concepts. Deep dives must be based on the implementations actually used by the repository and, when discussing pretrained architectures or research methods, refer to primary sources.

## Structure

Each deep dive is a Markdown file named after the model:

```text
docs/models_deepdive/<model_name>.md
```

The name must be lowercase and descriptive, such as `dinov2.md`, `resnet.md`, or `densenet.md`.

Each new deep dive must report its creation date immediately below the title, in the format:

```markdown
**Creation date:** YYYY-MM-DD
```

## Expected content

A deep dive should include, when relevant:

- precise model variant and role in the pipeline;
- flow of tensors, shapes and internal representations;
- structure of the main blocks and layers;
- pretraining or original training objective;
- adaptation to the multi-label ingredient classifier;
- trade-offs, limits and consequences for training, inference and interpretability;
- primary sources and references to local code.

Cross-cutting technical issues — such as a gradient error, a library limitation, or an integration choice — do not belong in this folder: they should be documented in [`../technical_details/`](../technical_details/).

## Documents available

- [`dinov2.md`](dinov2.md): ViT-B/14 with register token, DINO+iBOT pretraining, `_lc` head and multi-label pipeline integration.
