# Experiment comparison feasibility probe

**Created:** 2026-09-14
**Last updated:** 2026-09-14

This directory retains a bounded exploratory audit requested before implementing an N-experiment comparison tool. It is not the proposed production CLI and does not generate the final comparison HTML.

- `audit.py`: reads all 200 target-v5 configurations/CSV/TensorBoard inventories, replays relevant Optuna studies from a disposable journal copy, decodes one full local W&B session per family, and opens one selected checkpoint per family on CPU with `weights_only=True`.
- `inventory.json`: generated evidence, including source hashes, trial coverage, Optuna states/distributions, per-file TensorBoard validation values, sampled histogram summaries and checkpoint metadata.
- Reviewed current behavior and limitations belong to [experiment_artifacts.md](../../docs/implementation_details/experiment_artifacts.md).

Run from the repository root with the project's existing ML environment:

```bash
python src_scratches/experiment_comparison_audit/audit.py
```

The probe never initializes a training model, accesses the dataset images, calls W&B's remote API, synchronizes runs, or edits experiment data. It writes only its generated inventory. It uses W&B internal reader details verified against 0.28.0 and makes explicit assumptions about the two current campaigns, including a validation/checkpoint cadence of two epochs. It is intended for reproducibility of this audit, not arbitrary input validation. The source journal is copied before Optuna opens it.

## Proposed production structure — not implemented

Honor the requested `scripts/analise_exp/<name>/` convention, for example:

```text
scripts/analise_exp/compare_experiments/
    __main__.py
    cli.py
    schema.py
    discovery.py
    readers/
        config.py
        csv_metrics.py
        tensorboard.py
        optuna.py
        wandb_local.py
        checkpoints.py
    normalization.py
    comparability.py
    analysis/
        curves.py
        hyperparameters.py
        distributions.py
        aggregation.py
    reporting/
        json_report.py
        html_report.py
        templates/
```

There is already a differently named `scripts/analize_exps/` collection of exploratory notebooks. Treat these as historical references; do not execute notebooks that mutate/rebuild Optuna studies, or rename them as part of this work.

The intended CLI accepts N experiment paths, an optional explicit W&B root and Optuna journal, an output directory, the comparison metric/direction and any declared cohort filters. With the current layout, discover sibling shared logs by default and allow overrides for exported/moved campaigns. If only an experiment folder is supplied without its shared logs, still generate a scalar report and explicitly mark absent W&B/Optuna evidence.

## Proposed data and output contract

Represent `experiment -> numbered trial -> session -> observations`, with aliases and source files recorded separately. Retain original and canonical model/layer keys; split epoch, optimizer step, W&B history step and timestamp into different fields. Carry metric definition, loss weighting, model family, frozen/trainable status, data/vocabulary identity and provenance.

Proposed JSON sections: `schema_version`, `inputs`, `provenance`, `coverage`, `comparability_groups`, `experiments`, `intra_experiment`, `inter_experiment`, and `limitations`. Statistics carry their method, units, observation count, approximation status and missing-data reason. Emit strict JSON with null plus a reason for undefined values, not NaN or fabricated zeroes.

Generate HTML exclusively from that JSON, with experiment/trial/layer filters, sortable tables, scalar curves, distribution evolution, and visible coverage/limitations. A self-contained Plotly-based HTML is a plausible option using the existing Python plotting stack; no server is intrinsically required. Full raw W&B history should not be embedded by default. Keep numerical calculations on full observations; any display downsampling must be identified and must not affect statistics.

## Staged scope

1. **Reliable input normalization:** data-only configuration decoding, trial alias exclusion, study mapping, multi-file TensorBoard recovery, session conflict detection, CSV reconciliation, W&B internal-format adapter and explicit coverage. Use the actual restart/pruning cases as integration fixtures.
2. **Scalar and hyperparameter comparison:** observed/checkpoint/last selection semantics; best epoch, early-to-late change, common-interval learning-curve area, late-window level/slope/variability, descriptive train/validation gap, observed target-crossing time, conditional hyperparameter associations and distributions across trials. Declare windows, direction, budget and censoring. Target not reached is censored/unknown, not a late success.
3. **Distribution analysis:** streamed approximate moments, quantile bands, near-zero/tail bounds, common-support distribution distances and temporal summaries. Preserve original histogram counts/edges when needed in a sidecar; compare corresponding tensors only within compatible architectures. Use normalized block/role summaries for cross-family questions, separating frozen parameters and trainable gradients.
4. **JSON/HTML integration:** render the same validated results, with structured limitations and cohort sizes. Cache by input identity, reader version and analysis settings.
5. **Optional additional training instrumentation/evaluation:** exact unscaled accumulated gradients, AMP scale/non-finite counters, coordinate-wise updates at a selected cadence, initialization/snapshot identity, prediction-level metrics and per-label trajectories. These require producer changes or checkpoint inference and are outside the initial feasibility request.

No single unqualified best-model score should combine losses with different weighting, metrics with different definitions, different target spaces, or unequal search/epoch budgets. Trials sample hyperparameters; they are not repeated-seed replicates. Report temporal and configuration sensitivity without presenting it as seed-level statistical confidence.

## Verification boundaries

All current campaign files were inventoried at the scalar/configuration level. Only `trial_72` from each family was fully decoded for W&B histogram analysis; repeated W&B sessions were inventoried but not merged. Restart evidence was verified from TensorBoard, including a conflicting re-executed step in DINOv2 trial 93. Checkpoints were inspected structurally, without model reconstruction or evaluation.

The remote project page could not be inspected through the web reader. Local records provide the evidence for this audit; current online API parity is unverified. A future optional remote reader should prefer complete history retrieval over sampled plotting data and must preserve sparse rows. W&B's [public run API](https://docs.wandb.ai/models/ref/python/public-api/run#method-runscan_history) specifies that requesting several keys together returns only rows containing all of them.
