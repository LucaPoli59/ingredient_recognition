# Discovery research

**Created:** 2026-08-02  
**Last updated:** 2026-08-28

Use this directory for broad research snapshots of the state of the art, such as recent methods, benchmarks, tools, and emerging directions relevant to the project.

Before starting a discovery, review the current [`project_objective/`](../../project_objective/README.md) documents so that the research is grounded in the project's defined problem, scope, and success criteria.

```text
discovery/
└── <discovery_date>/
    ├── README.md
    └── <files>.md
```

Name each `<discovery_date>` directory with an ISO date (`YYYY-MM-DD`). Its `README.md` must explain the discovery's context, scope, and the purpose and structure of the files it contains.

Before creating a new discovery, review at least the two most recent existing discovery directories. Reuse or link to relevant prior findings, and do not duplicate research unless the new record documents a material update or a distinct perspective.

## Discovery index

- [`2026-08-28/`](2026-08-28/README.md) — Subphase 4A.1 experimental-model
  broad discovery: problem-to-model requirements, five new family/protocol
  candidates, already-used ResNet/DINOv2 baseline anchors, grouped exclusions,
  and the formal handoff to 4A.2.
- [`2026-08-22/`](2026-08-22/README.md) — reference-selector
  candidate landscape across supervised, visual self-supervised,
  vision-language, food-domain, and structured multi-label families, with
  explicit pretraining and interpretation boundaries. Its family, source, and
  technical evidence may also inform Subphase 4A, while selector intake tiers
  and dispositions remain specific to Subphase 4B.
- [`2026-08-02/`](2026-08-02/README.md) — primary broad evidence base for Subphase 4A, covering food ingredient inference, multi-label models, representation learning, data and ontology processing, augmentation, leakage control, calibration, interpretability, and a compute-aware research program; reusable evidence may also inform Subphase 4B.
