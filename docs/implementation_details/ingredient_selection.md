# Ingredient-selection workflow

**Created:** 2026-09-24
**Last updated:** 2026-09-27

## Purpose and scope

This document describes the maintained implementation of the Phase 3
ingredient-learnability campaign. The binding scientific choices remain owned
by [4B-D1, Phase 3-D1 and the D2/D3 amendments](../project_objective/model_comparison_methodology.md#phase-3-d3--forty-epoch-campaign-amendment),
while the operational sequence and current status remain owned by the
[ingredient-selection plan](../plans/recognizable_ingredient_selection.md).

The implementation prepares the selector campaign and its deterministic
analysis. It does not contain a `v5` label outcome, numerical P3 profile gates,
or a selected vocabulary.

## Maintained components

| Component | Current responsibility |
| --- | --- |
| `src/models/efficientnet.py` | Builds the exact pretrained EfficientNetV2-S adapter, preserves stock pooling/dropout, installs the biased 165-logit head, and asserts full trainability. |
| `src/data_processing/transformations.py` | Implements RGB conversion, exact long-side-384 round-half-up resize, ImageNet-mean center padding, train-only horizontal flip, and ImageNet normalization. |
| `src/ingredient_selection/data.py` | Loads only the frozen train and validation metadata, establishes the ordered train class space, encodes targets, and provides shuffled training plus ordered deterministic audit loaders. It has no test-split path. |
| `src/ingredient_selection/protocol.py` | Owns frozen constants, deterministic runtime setup, hashes, train-only positive weights, the support-stratified pilot rule, and Git status helpers. |
| `src/ingredient_selection/batching.py` | Resolves exact physical/accumulated batch arithmetic and weights the incomplete final group correctly. |
| `src/ingredient_selection/provenance.py` | Hashes the Python source inventory and saves its exact bytes in a source snapshot, including new maintained source files. |
| `src/training/ingredient_selection.py` | Owns weighted BCE, the one-group AdamW optimizer, linear-warm-up/cosine scheduler, fixed-state audit callback, and real-data CUDA resource gate. |
| `src/ingredient_selection/metrics.py` | Computes per-label AP, fixed-0.5 precision/recall/F1, micro F1, trajectory-window summaries, deterministic AP bootstrap intervals, and application of later P3 gates. |
| `src/ingredient_selection/artifacts.py` | Writes the manifest, cohort, tidy metrics, compressed validation scores, and bootstrap output atomically while rejecting duplicate audit keys. |
| `src/ingredient_selection/analysis.py` | Validates provenance and audit cadence, derives controls and profile evidence, exposes only the 24-label pilot before a hashed rule exists, and writes `validation_summary.json`. |
| `scripts/ingredient_selection/` | Provides thin campaign, analysis, and historical-reproduction commands. |
| `scripts/launch_exps/ingredient_selection/train_selector.py` | Rerunnable launcher: descending short OOM probes or a full-epoch resource gate followed by the fresh campaign. |

## Campaign execution boundary

Run commands from the WSL repository with the `wsl_image_pytorch` interpreter:

```bash
/root/miniconda3/envs/wsl_image_pytorch/bin/python \
  scripts/launch_exps/ingredient_selection/train_selector.py --probe-batches

/root/miniconda3/envs/wsl_image_pytorch/bin/python \
  scripts/launch_exps/ingredient_selection/train_selector.py

# Fresh repeat without overwriting an existing campaign:
/root/miniconda3/envs/wsl_image_pytorch/bin/python \
  scripts/launch_exps/ingredient_selection/train_selector.py --run-name repeat_01

/root/miniconda3/envs/wsl_image_pytorch/bin/python \
  scripts/ingredient_selection/analyze_campaign.py \
  analysis_outputs/ingredient_selection/phase3-d1-v3
```

The resource gate is mandatory and is bound to the protocol ID, ordered-class
hash, train/validation metadata hashes, exact pretrained-weight identity,
batch/worker settings, Git revision, and Python source-content hash. It must
complete a disposable full train epoch and ordered validation inference; short
OOM probes are insufficient to launch a campaign. The full launcher writes
`pilot_cohort.json` before constructing the campaign model, verifies the gate,
preserves `source_snapshot.zip`, and records the initial head hash and every
frozen configuration field in `campaign_manifest.json`. Actual tracked changes
are recorded; a clean working tree is no longer mandatory in v2. The ZIP
contains Python sources, including new maintained modules and launchers, and
excludes data, checkpoints, outputs, and `.env`. The source hash must remain
unchanged throughout the full-epoch gate.

The active `phase3-d1-v3` campaign runs **40 epochs**: 2 linear warm-up epochs
and 38 cosine epochs, with no early stopping. The manifest derives its budget
and scheduler fields from the same protocol object used for training. Analysis
rejects the superseded 20-epoch budget or 18-epoch cosine even when a conflicting
manifest has a self-consistent identity hash.

The requested effective batch is 128. `EfficientNetV2SSelector` exposes the
measured `MAX_ALLOWED_BATCH_SIZE = 8`; the resolver picks the largest divisor
of 128 under the cap and passes physical 8 to the DataModule and accumulation
16 to Lightning. Sampling remains without replacement and `drop_last=False`.
For 47,965 records this produces 375 optimizer updates per epoch and 15,000
over 40 epochs: 374 groups
of 128 and one group of 93. The returned microbatch loss is scaled by
`accumulation * actual_microbatch_records / actual_group_records`, compensating
for Lightning's division by the fixed accumulation count at the tail. The
logged loss remains the unscaled sample-mean BCE. BatchNorm sees physical 8.

CUDA allocation is limited to
`min(total_VRAM - 512 MiB, free_VRAM_at_start - 256 MiB)` in both capacity
tests and campaign startup. The recorded per-process allocator limit prevents
acceptance through the locally observed host/shared-memory oversubscription.
Only CUDA OOM causes the capacity scan to try the next divisor; other failures
stop it for diagnosis. No batch change occurs during the campaign.

The training loader shuffles records without replacement using seed 42. Audit
loaders are ordered, use the validation transform, and run the module in
evaluation mode. Full train and validation audits occur at epochs
`0,2,...,40`; train AP therefore never aggregates predictions emitted while
weights are changing. Only validation record IDs, targets, and logits are
persisted at each audit point. Early, near and late AP windows are `{2,4,6}`,
`{30,32,34,36,38}` and `{32,34,36,38,40}`; final bootstrap uses epoch-40 scores.

Before P3 creates a valid `profile_rule.json`, the analysis command writes only
the deterministic 24-label pilot to `profile_evidence.csv`. A full analysis is
unlocked only when the rule's own hash, campaign identity hash, and pilot hash
all match. Neither path opens test metadata.

## Historical reproduction

`scripts/ingredient_selection/reproduce_historical.py` delegates to the
maintained read-only retention audit. It regenerates the four 46-label Q3 sets
and their exact 40-label intersection, verifies the three projected metadata
hashes and the executable checkpoint anchors, and reports the historical
weighting/augmentation discrepancies.

The two Optuna journals are declared append-only artifacts. A journal may grow
only when its entire originally hashed byte prefix remains unchanged; modifying
that prefix or any other retained artifact still fails verification. This
preserves the historical evidence while allowing later trials to append to the
shared journal.

## Verified resource and test evidence

On 2026-09-24 the real-data gate used an RTX 4060 with 8,187.375 MiB reported
memory and a physical `[8,3,384,384]` batch. The exact forward, weighted BCE
backward, and AdamW step passed in true FP32 with 3,632.20 MiB peak allocated
and 5,362.00 MiB peak reserved. This is a feasibility result, not throughput or
accuracy evidence.

On 2026-09-27, user-requested v2 trials started from physical 128 and tested
descending divisors under the explicit VRAM cap. Physical 128, 64, 32 and 16
raised CUDA OOM. Physical 8 with accumulation 16 completed two real optimizer
updates: 3,879.19 MiB peak allocated and 5,364.00 MiB peak reserved, with a
6,841 MiB allocator allowance at that time. The full-epoch gate remains the
mandatory final check; quick probes alone are not a completed epoch.

The D2 repository suite passed 68 tests, including exact divisor
resolution, sample-mean gradient equivalence for the incomplete group, an
actual Lightning-versus-direct-BCE update test, and rejection of source bytes
changed before snapshot creation. The D3 suite passes 72 tests, additionally
checking the 40-epoch schedule, final windows, manifest/training agreement and
rejection of the old budget/scheduler. Historical P2's 64-test result remains below.

The P2 repository suite passed 64 tests after the original integration. The Phase 3 tests
cover resize/padding/RGB behavior, head construction and trainability,
positive-weight arithmetic, blind cohort reproduction and tamper rejection,
AP/F1 separation, bootstrap determinism, trajectory edge cases, profile
assignment, the complete 20-epoch learning-rate sequence, optimizer/scheduler
resume, eval-mode auditing, test-split isolation, rule-gated label exposure,
and append-only historical retention.

## Limitations and next action

- No selector campaign outcome has been inspected; P3 remains responsible for
  running the sealed campaign and freezing numerical profile gates from only
  the 24-label pilot.
- The single campaign does not estimate seed or configuration stability.
- Bootstrap intervals describe validation-record uncertainty only.
- The cuisine prior is a mechanism diagnostic, not an image-model competitor.
- Checkpoints and generated outputs remain outside durable documentation.

P3 uses the main workspace and the dedicated Python launcher. Complete the
disposable full-epoch gate against the committed sources, then start a newly
initialized 40-epoch v3 campaign. The incomplete v2 capacity gate was interrupted
before any v2 campaign started. The
interrupted v1's artifacts are retained in the original report/experiment
directories and are excluded from replacement learnability evidence.
