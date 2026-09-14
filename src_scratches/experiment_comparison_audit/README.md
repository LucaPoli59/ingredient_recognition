# Experiment comparison feasibility probe

**Created:** 2026-09-14
**Last updated:** 2026-09-14

This directory retains a bounded exploratory audit requested before implementing an N-experiment comparison tool. It is not the proposed production CLI and does not generate the final comparison HTML.

- `audit.py`: reads all 200 target-v5 configurations/CSV/TensorBoard inventories, replays relevant Optuna studies from a disposable journal copy, decodes one full local W&B session per family, and opens one selected checkpoint per family on CPU with `weights_only=True`.
- `inventory.json`: generated evidence, including source hashes, trial coverage, Optuna states/distributions, per-file TensorBoard validation values, sampled histogram summaries and checkpoint metadata.
- [`export_parameter_example.py`](export_parameter_example.py): exports the full logged histogram trajectory for `model.layer4.1.conv2.weight` from the audited ResNet trial 72, preserving original bin edges/counts and computing explicitly approximate moments.
- [`parameter_histogram_example.json`](parameter_histogram_example.json): 168 observations across epochs 0–39, each with 64 bins accounting for 2,359,296 parameter elements; includes source SHA-256, enclosing history coordinates and quantile-bin intervals.
- Reviewed current behavior and limitations belong to [experiment_artifacts.md](../../docs/implementation_details/experiment_artifacts.md).

Run from the repository root with the project's existing ML environment:

```bash
python src_scratches/experiment_comparison_audit/audit.py
```

The probe never initializes a training model, accesses the dataset images, calls W&B's remote API, synchronizes runs, or edits experiment data. It writes only its generated inventory. It uses W&B internal reader details verified against 0.28.0 and makes explicit assumptions about the two current campaigns, including a validation/checkpoint cadence of two epochs. It is intended for reproducibility of this audit, not arbitrary input validation. The source journal is copied before Optuna opens it.

The parameter example can be reproduced separately:

```bash
python src_scratches/experiment_comparison_audit/export_parameter_example.py
```

Its source SHA-256 is checked before and after extraction. The exported bins are the original logged values; mean, standard deviation and RMS use bin midpoints, and quantiles are represented by containing-bin intervals. The example deliberately covers one parameter and one session; it does not merge restarted runs or recover individual tensor coordinates. Existing parameter histograms provide a usable storage/analysis tradeoff for the initial comparison feature.

## Planning handoff

**Status:** Superseded as an implementation-planning source on 2026-09-14.

The requirements, module layout, CLI proposal, logging contract and staged implementation scope previously collected here were consolidated into [the operational comparison plan](../../docs/plans/experiment_comparison.md). That completed plan retains the execution history; the maintained runtime contract is [the experiment comparison implementation detail](../../docs/implementation_details/experiment_comparison.md). It retains the decision history, including the move of the optional logging flag from the internal torch model to the Lightning model, with default false.

This directory remains the reproducibility home for the exploratory audit scripts, raw histogram example and generated evidence. The production feature is still pending; the audit probes are not its CLI.

## Verification boundaries

All current campaign files were inventoried at the scalar/configuration level. Only `trial_72` from each family was fully decoded for W&B histogram analysis; repeated W&B sessions were inventoried but not merged. Restart evidence was verified from TensorBoard, including a conflicting re-executed step in DINOv2 trial 93. Checkpoints were inspected structurally, without model reconstruction or evaluation.

The remote project page could not be inspected through the web reader. Local records provide the evidence for this audit; current online API parity is unverified. A future optional remote reader should prefer complete history retrieval over sampled plotting data and must preserve sparse rows. W&B's [public run API](https://docs.wandb.ai/models/ref/python/public-api/run#method-runscan_history) specifies that requesting several keys together returns only rows containing all of them.
