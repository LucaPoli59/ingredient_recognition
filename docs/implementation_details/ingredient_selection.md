# Ingredient-selection workflow

**Created:** 2026-09-24
**Last updated:** 2026-10-04

## Purpose and scope

This document describes the maintained implementation of the Phase 3
ingredient-learnability campaign. The binding scientific choices remain owned
by [4B-D1, Phase 3-D1 and the D2/D3 amendments](../project_objective/model_comparison_methodology.md#phase-3-d3--forty-epoch-campaign-amendment),
while the operational sequence and current status remain owned by the
[ingredient-selection plan](../plans/recognizable_ingredient_selection.md).

The implementation ran the v3 selector campaign and provides deterministic,
blind-gated analysis. P3 numerical gates are stored in `profile_rule.json` and
owned methodologically by [Phase 3-D4](../project_objective/model_comparison_methodology.md#phase-3-d4--pilot-frozen-numerical-profile-rule).
The [pilot result](../experiment_results/phase3_d1_v3_pilot.md) owns observed
outcomes; no final selected vocabulary exists yet.

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
| `src/ingredient_selection/analysis.py` | Validates provenance and audit cadence, derives controls and profile evidence, exposes only the 24-label pilot before a hashed rule exists, writes `validation_summary.json`, and classifies only the archived pilot on demand after the rule freeze. |
| `src/ingredient_selection/reporting.py` | Revalidates a completed full profile against the rule, pilot decisions, class order, evidence hash and trajectory table; writes named provisional groups and deterministic SVG/PNG diagnostic figures. |
| `src/ingredient_selection/observability.py` | Uses only the standard library to validate the completed P4 inputs, sample a blind P5 validation-image pilot, verify/harden the packet, render two independently ordered review forms, and score completed human responses without choosing a vocabulary. |
| `scripts/ingredient_selection/` | Provides thin campaign, analysis, pilot/full-report, and historical-reproduction commands. |
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

# After freezing profile_rule.json, report only the archived pilot:
/root/miniconda3/envs/wsl_image_pytorch/bin/python \
  scripts/ingredient_selection/report_pilot.py \
  analysis_outputs/ingredient_selection/phase3-d1-v3

# P4 only, after separate authorization: apply the unchanged rule to all labels
# and render the validated full-profile report and figures:
/root/miniconda3/envs/wsl_image_pytorch/bin/python \
  scripts/ingredient_selection/analyze_campaign.py \
  analysis_outputs/ingredient_selection/phase3-d1-v3
/root/miniconda3/envs/wsl_image_pytorch/bin/python \
  scripts/ingredient_selection/report_campaign.py \
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
all match. The P3 freeze also verifies the exact classifier source hash and the
preserved `pilot_profile_evidence.csv` bytes. The
`report_pilot.py` path reads that preserved 24-row artifact and writes only
`pilot_profile_decisions.csv` and `pilot_profile_summary.json`; it never loads
the 165-label metrics file. Do not rerun `analyze_campaign.py` after rule
creation until P4 is separately authorized, because a valid rule unlocks its
full-label mode. P4 has now been authorized: its full analysis writes a 165-row
`profile_evidence.csv`, while `pilot_profile_evidence.csv` and
`pilot_validation_summary.json` preserve P3's pre-rule evidence. The
`report_campaign.py` path rejects a pilot-only analysis, changed evidence,
modified rule or classifier source, different pilot decisions, missing class
order, inconsistent audit trajectory keys, and any test-split access recorded
by the analysis summary. Its `p4_profile_report.json` contains all provisional
named groups, reason counts, hashes and figure exemplars. Neither analysis
path opens test metadata.

`classify_profile` applies eight absolute gates and a conservative uncertainty
band. It uses the epoch-40 AP bootstrap bounds as a corroboration check around
the late-window validation-AP gate and the image-versus-cuisine-prior margin.
Missing or invalid bootstrap evidence, low support, large temporal IQR/gap,
or a bootstrap interval overlapping either decision boundary yields
`uncertain`; fixed-0.5 F1 never affects classification. The bootstrap is a
record-resampling interval for final AP, not a formal interval for a late
median or AP difference, and not training-seed uncertainty.

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
6,841 MiB allocator allowance at that time. Quick probes alone were not a
completed epoch; the mandatory full-epoch result is recorded below.

The v3 full-epoch gate subsequently passed against Git revision
`192059ef20c1d58e3ce2b2b54c1017b67ebaa5b1`: 47,965 train records, 375 AdamW
updates, physical 8, accumulation 16, true FP32, 3,888.79 MiB peak allocated
and 5,392.00 MiB peak reserved under the 6,841 MiB allowance. The disposable
training plus ordered finite-logit validation check took 3,181.72 seconds.
The raw record is
`analysis_outputs/ingredient_selection/phase3-d1-v3_resource_gate.json`.

**Device limitation:** the training epoch ran on CUDA, but Lightning 2.6.1's
strategy teardown moves the module to CPU when `Trainer.fit` returns.
`run_resource_gate` then uses `module.device` for its post-fit validation loop,
so that finite-logit check ran on CPU. The recorded GPU peak is a training
capacity result, not proof of a complete CUDA validation audit. The actual
campaign's audits run inside Trainer callbacks before teardown. Correcting the
post-fit gate device is a future runtime follow-up; do not silently relabel the
completed measurement or retroactively alter the campaign provenance.

The launcher then constructed a fresh model and started the 40-epoch campaign
at `2026-09-27T16:17:14.169539+00:00` (18:17 Europe/Rome). Its manifest records
the same revision, gate, effective-batch plan, 15,000 planned updates, and source
snapshot. A read-only launch audit verified all 152 archived Python sources
against the manifest inventory, without opening per-label campaign results:

| Artifact | Verified SHA-256 |
| --- | --- |
| Source inventory | `fbb7bcd46dd9587955f681d52aa3312c6f4837e38b3293c9488f8af4473b88a0` |
| Source ZIP | `3d251f0e0a6f1f5ed7e1b95c17921016809bee369cc405b4b1b5cdd3cc5d2a1f` |
| Full-epoch gate | `2a4dae285bcae02bb2e71689813830938cb543cbda881ef9260b776fe7a551aa` |

The D2 repository suite passed 68 tests, including exact divisor
resolution, sample-mean gradient equivalence for the incomplete group, an
actual Lightning-versus-direct-BCE update test, and rejection of source bytes
changed before snapshot creation. The D3 suite passes 72 tests, additionally
checking the 40-epoch schedule, final windows, manifest/training agreement and
rejection of the old budget/scheduler. Historical P2's 64-test result remains below.

On 2026-09-28 the 40-epoch v3 campaign finished at 14:36 UTC with an epoch-40
audit, final validation bootstrap and checkpoint; the manifest status is
`completed`. The pilot-only analysis verified the entire declared audit
cadence, metadata/class order, final score-record order and blind cohort hash,
then exposed only 24 labels. The frozen rule validates its own content hash,
the campaign and pilot identities, preserved pilot evidence and classifier
source hash. The pilot-report command returned exactly 24 decisions and its
summary hashes. The exact post-campaign classifier, analysis and pilot-report
sources are retained in `pilot_analysis_source.zip` (SHA-256
`5d1659465cb3acd9e2b51838a6be2f485c1655b9b78641f604ee164ec5591241`),
separately from the campaign's training source snapshot. The repository suite still passes all 72 tests after adding
the bootstrap-overlap classification and pilot-report route. The reviewed
outcomes are in the [pilot result](../experiment_results/phase3_d1_v3_pilot.md).

P4 reused the same v3 artifacts with the unchanged D4 rule. The full report
validated all 165 class identities, the 24 archived pilot outcomes and the
complete train/validation audit table. Two consecutive rendering runs produced
byte-identical JSON, three SVG figures and their PNG counterparts. The
[reviewed full-profile result](../experiment_results/phase3_d1_v3_full_profile.md)
owns the observed numerical outcomes; figures are diagnostic, not human
observability annotations or final vocabulary decisions.

The P2 repository suite passed 64 tests after the original integration. The Phase 3 tests
cover resize/padding/RGB behavior, head construction and trainability,
positive-weight arithmetic, blind cohort reproduction and tamper rejection,
AP/F1 separation, bootstrap determinism, trajectory edge cases, profile
assignment, the complete 40-epoch learning-rate sequence, optimizer/scheduler
resume, eval-mode auditing, test-split isolation, rule-gated label exposure,
and append-only historical retention.

## Optional appendix: retained former P5 blind review pilot

The [appendix protocol](../project_objective/ingredient_observability_protocol.md)
and its prepared packet are retained for optional interpretation of future
results. Under [D5](../project_objective/model_comparison_methodology.md#phase-3-d5--numerical-selection-and-optional-interpretation-appendix),
manual review is not a selection dependency and cannot affect vocabulary
membership or primary model comparisons. The package initializer loads
the training protocol lazily so the standard-library observability module can
run without importing PyTorch. From the WSL repository root:

```bash
/usr/bin/python3 scripts/ingredient_selection/observability_review.py prepare
```

The command verifies the completed full P4 report, frozen rule and evidence
hashes, campaign class order, validation-only metadata hash and image paths.
It writes, without overwriting changed files, the unblinded
`analysis_outputs/ingredient_selection/phase3-d1-v3/p5_observability/pilot/packet_manifest.json`
and blinded `reviewer_a.html` and `reviewer_b.html`. The packet contains 64
image–label pairs with a unique image for each pair, six recipe positives and
two recipe negatives for each of eight purposively chosen pilot labels. The
HTML uses relative paths to the local shared image store; it does not copy
images or reveal the stratum, recipe name, model score or P4 outcome. Opening
the HTML where its relative image paths resolve is required. Responses stay
in browser-local storage until each reviewer exports their own JSON.

After two *different people* complete the forms, score only their actual
exports with `observability_review.py score --packet ... --reviewer-a ...
--reviewer-b ... --output ...`. The scorer rejects changed packet contents,
changed source-image bytes, wrong/missing pair IDs, invalid categories and
identical reviewer identifiers. Software cannot prove that the named people
worked independently; that remains a study-procedure requirement.
It emits a four-category confusion matrix, raw agreement, Cohen's kappa and
descriptive agreed-category counts for recipe-positive pairs. It never
adjudicates or writes a selected vocabulary. No human results exist yet.
The two focused standard-library tests pass. On 2026-09-29 the full ML test
suite was unavailable because a bare `import torch` segfaulted before test
discovery; this does not constitute a test failure of the new review module.

## Limitations and next action

- P3 inspected only the blind 24-label pilot; P4 subsequently applied its
  frozen rule to the other 141 outcomes without another selector training.
- The single campaign does not estimate seed or configuration stability.
- Bootstrap intervals describe validation-record uncertainty only.
- The cuisine prior is a mechanism diagnostic, not an image-model competitor.
- Checkpoints and generated outputs remain outside durable documentation.

P3 and P4 are complete. Mandatory P5 review is superseded, with its unannotated
packet and tools retained as an optional appendix. P6 owns the shared
numerical-profile-based vocabulary freeze and can proceed without reviewers;
no current report establishes direct visibility or a published final
projection. The incomplete v2 capacity gate was
interrupted before any v2 campaign started. The interrupted v1's artifacts
remain in the original report/experiment directories and are excluded from
replacement learnability evidence. The post-fit gate's CPU validation
limitation above remains explicit.
