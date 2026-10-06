# Ingredient-selection workflow

**Created:** 2026-09-24
**Last updated:** 2026-10-06

## Purpose and scope

This document describes the maintained implementation of the Phase 3
ingredient-learnability campaign. The binding scientific choices remain owned
by [4B-D1, Phase 3-D1 and the D2/D3 amendments](../project_objective/model_comparison_methodology.md#phase-3-d3--forty-epoch-campaign-amendment),
while the operational sequence and current status remain owned by the
[ingredient-selection plan](../plans/recognizable_ingredient_selection.md).

The implementation ran the v3 selector campaign and provides deterministic,
blind-gated original analysis plus separately versioned D6 inclusion review.
P3 numerical gates are stored in `profile_rule.json` and
owned methodologically by [Phase 3-D4](../project_objective/model_comparison_methodology.md#phase-3-d4--pilot-frozen-numerical-profile-rule).
The [pilot result](../experiment_results/phase3_d1_v3_pilot.md) owns observed
outcomes. [D6](../project_objective/model_comparison_methodology.md#phase-3-d6--held-out-quality-inclusion-policy)
owns the adopted post-P4 inclusion policy; P6's final projection is separate.

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
| `src/ingredient_selection/inclusion.py` | Validates retained D4, campaign/snapshot/metadata/score provenance, hashes validation-only image groups, freezes D6 separately and produces a write-once revised report without importing the training stack. |
| `src/ingredient_selection/inclusion_statistics.py` | Computes paired image-cluster bootstrap intervals for median-of-five AP and its excess over resampled prevalence; applies independent D6 evidence axes. |
| `src/ingredient_selection/inclusion_reporting.py` | Renders a deterministic SVG from existing D6 decisions and intervals; plotting does not determine membership. |
| `src/ingredient_selection/projection.py` | Publishes the approved D6 membership as a write-once, versioned vocabulary definition without loading data, predictions or the training stack. |
| `src/ingredient_selection/resources/` | Stores the portable shared P6 definition with ordered names, original indices, excluded groups/reasons and source/evidence hashes. |
| `src/ingredient_selection/runtime.py` | Resolves the approved opt-in vocabulary, validates saved identity/order and base targets, and projects targets or verified full-model output columns. It never reselects labels. |
| `scripts/ingredient_selection/verify_runtime.py` | Read-only train/validation encoding-parity check against the original 165 columns; no test metadata, image inference or output writes. |
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

## D6 saved-score inclusion review

Run the versioned amended analysis with the existing NumPy environment:

```bash
python scripts/ingredient_selection/review_inclusion.py \
  analysis_outputs/ingredient_selection/phase3-d1-v3
```

The command reads only train/validation metadata, retained original D4/P4
artifacts, the complete audit metric table and validation score archives at
epochs 32/34/36/38/40. It does not construct a model, perform inference,
read test metadata/images or change the base vocabulary. It verifies the
completed v3 identity, class order, metadata/supports, training ZIP and all
source members, original classifier/pilot hashes, full-analysis evidence,
every score/target/record alignment and saved per-checkpoint AP parity.

Validation image bytes are hashed to recover exact-image groups in the frozen
record order. Each bootstrap draw samples G groups G times uniformly, carrying
all their records with multiplicity. The same draw is used for every checkpoint
AP and validation prevalence; Q is the median of the five separate APs, not
an ensemble score. Weighted tied-rank AP avoids repeatedly sorting each
resample and is tested against sklearn and explicit record replication.
There are 1,000 valid draws per label with seed `42000 + class_index`, at most
10,000 attempts, invalid-class draw accounting and nominal 95% percentile
intervals. Missing/invalid evidence cannot promote a label.

`classify_inclusion` requires Q and its lower bound at least 0.20, a strictly
positive lower bound for paired Q minus prevalence, and IQR at most 0.03.
All applicable reasons and per-axis states survive; a stable valid interval
wholly below the quality floor is `below_quality_floor`, otherwise non-passing
evidence is `uncertain`. Train support/gain/AP/gap and cuisine comparisons are
retained diagnostics only. The fixed 0.15/0.20/0.25 panel is descriptive.

Outputs live only in `inclusion_d6_v1/` under the campaign:

- `inclusion_rule.json`: adopted policy, exact source inventory/snapshot hash,
  input hashes, image-group hash, environment and Git base revision;
- `source_snapshot.zip`: deterministic archive of the analysis modules and CLI;
- `validation_image_groups.json`: validation-only ordered ID/image/hash inventory;
- `inclusion_report.json`: full statistics, independent decisions/diagnostics,
  D4 membership changes, fixed sensitivity and limitations;
- `inclusion_evidence.csv`: compact per-label inspectable summary;
- `inclusion_decision_map.svg`: derived AP/interval/prevalence figure.

The rule is written before resampling. Re-execution accepts identical bytes
only, preserves the original Git base revision after later commits, and rejects
changed inputs, source or environment. Exclusive atomic file publication avoids
partial artifacts; source/input/image hashes are rechecked before report
publication. Neither original `profile_rule.json` nor `metrics.py` is changed.
The report records eligibility, not a P6 metadata projection or new default.

The original campaign did not hash every score archive or image at launch.
D6 hashes their retained bytes now and checks score/metric parity; it cannot
retroactively prove launch-time byte identity. Its post-outcome policy and
nominal per-label intervals do not establish seed robustness, simultaneous
coverage, unbiased selected-set performance or direct visibility.

Verification on 2026-10-05: 26 new synthetic tests and all 100 repository tests
pass; two full D6 executions reproduce all six artifacts byte-for-byte. The
repository suite includes a pre-existing encoder check that reads test metadata
only for vocabulary compatibility, not test predictions or predictive metrics.
That check is separate from D6's train/validation-only selection I/O and did
not inform the policy. Exact outcomes, artifact hashes and the verification
boundary are recorded in the [reviewed D6 result](../experiment_results/phase3_d1_v3_d6_profile.md).

## P6 frozen projection

The explicit resource
[`ingredients_selected_v5_d6_v1.json`](../../src/ingredient_selection/resources/ingredients_selected_v5_d6_v1.json)
defines 59 selected labels. It is a vocabulary definition, **not** a new split
metadata generation or an automatically active training setting. The full
165-label `ingredients_target` vocabulary remains the default. The binding
[P6 handoff](../project_objective/model_comparison_methodology.md#p6-shared-projection-freeze--2026-10-05)
keeps all models on the same selected task and retains the outcome-informed,
single-selector limitation.

Regenerate from the repository root, with only the Python standard library:

```bash
python scripts/ingredient_selection/export_projection.py \
  analysis_outputs/ingredient_selection/phase3-d1-v3
```

The default destination is the resource above; `--output PATH` supports a
separate write-once verification copy. Existing different bytes are refused.
The exporter reads only the completed campaign manifest, approved D6 rule and
report, D6 source ZIP and its own source files. It verifies canonical identities,
the exact approved rule/report hashes, manifest/ZIP byte hashes and every ZIP
member. It checks every row's original identity, eligibility, aggregate counts
and selected-name agreement, then packages decisions without recomputing gates,
AP or uncertainty. No metadata, images, score arrays, model or test split is
opened. Input and exporter source hashes are rechecked before atomic publication.

Schema version 1 contains:

- `projection_id`, `policy_id`, freeze date and interpretation;
- `base_vocabulary`: the full class order/hash, original metadata filename and
  target field, and frozen train/validation metadata hashes;
- `class_order`, `class_order_hash`, `label_count` and `base_class_indices`:
  selected column `j` corresponds to original column `base_class_indices[j]`;
- `groups_base_indices`: exhaustive, disjoint `included`, `uncertain` and
  `below_quality_floor` groups in base order;
- `excluded_decisions`: all 106 exclusions, with original indices, independent
  reasons and axis statuses copied from D6. Exact statistics, support/prevalence
  and optimization/context diagnostics remain in the hash-bound report at
  `labels[base_class_index]`;
- `evidence`: campaign, rule/report and D6 source-snapshot identities;
- `export_source_inventory`, usage boundaries, limitations, and a canonical
  `artifact_hash` over the object without its own hash field.

P7 must use this saved class order explicitly, intersect existing targets with
the selected names, preserve all records/splits/order (including all-zero
projected targets), and slice full-model predictions by the original indices.
It must not refit the vocabulary independently per split, discard empty rows,
or change the default. The exporter does not implement that runtime wiring.
Phase 6 still owns random-control generation/training and any reduced-task HPO.

Verification on 2026-10-05: repeated exports are byte-identical; all 13 focused tests
and 113 repository tests pass. Focused tests cover ordering/column projection,
empty-target retention in the handoff, independent reasons, identity and tamper
rejection, write-once behavior, mid-export source/input changes, and an explicit
file-read allowlist, snapshot-member validation and the published resource's
agreement with approved evidence/current exporter sources. Import does not
load NumPy, Torch or Lightning. The generic
suite retains the metadata-only test-split compatibility caveat documented
above; no test predictions or predictive metrics are computed. Exact published
identities are in the [reviewed P6 handoff](../experiment_results/phase3_d1_v3_d6_profile.md#p6-publication--2026-10-05).

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

## P7 runtime projection

The canonical opt-in for a **new, separately named** experiment is:

```python
from src.commons.exp_config import ExpConfig

config = ExpConfig(dm_ingredient_projection="ingredients_selected_v5_d6_v1")
```

Leaving `dm_ingredient_projection=None` preserves the full 165-label default.
Use the existing `src/training/` entry points with this configuration; P7 does
not introduce another training pipeline. Auto-resuming an existing experiment
uses its saved configuration: changing incoming options is not a vocabulary
migration. Use a distinct experiment name for the selected task.

`runtime.resolve_projection` accepts only the registered ID or its exact saved
contract. It verifies the approved P6 canonical artifact hash, selected order,
base order and index mapping. `ImagesRecipesBaseDataModule` then requires the
original `ingredients_target_v5_metadata.json`, `ingredients_target`, and no
cuisine/category restriction. It verifies the frozen train/validation metadata
bytes. There is no new split generation or in-place metadata update.

The strict `MultiLabelBinarizer` is fitted explicitly to the saved 59 names
**without inferring classes from any split**. Each original target is checked
against the full base vocabulary before intersection. Known but unselected
labels are dropped; unknown base labels raise an error rather than disappearing.
Record IDs, order and split membership are preserved, including all-zero
selected targets. The generic DataModule still prepares train/val/test/predict
metadata eagerly, as before; this runtime behavior is distinct from the
train/validation-only selector and read-only verification command.

`ExpConfig`, DataModule hyperparameters and the Lightning model persist the
projection ID, canonical artifact hash, exact selected names/order/hash, base
order hash and `base_class_indices`. Model construction uses 59 outputs.
Configuration, startup and restore reject incompatible heads, encoders, orders
or projection markers. Checkpoints also retain a top-level
`ingredient_projection`, including light checkpoints that omit hyperparameter
sections. A selected checkpoint cannot be restored as a full-task model (or
vice versa). Legacy full/robust checkpoints without markers remain supported;
their saved `<UNK>` behavior is not reinterpreted.

### Analysis handoff

Selected output column `j` corresponds to original
`base_class_indices[j]`. For full-model logits, scores or targets:

```python
from src.ingredient_selection.runtime import project_output_columns

selected = project_output_columns(full_values, saved_full_class_order)
```

The helper requires a two-dimensional 165-column NumPy/Torch array and the
exact saved base class order. It preserves row order, dtype and Torch device
and gradients. It does not infer column semantics from width alone or compute
metrics. Compare full-model outputs restricted to the shared columns with
selected-model outputs only under the later benchmark protocol.

The maintained experiment comparator resolves selected label names even when
HPO omits the trial encoder, validates saved projection markers, and includes
projection hash in objective cohorts. Full- and selected-task losses are not
pooled just because their metadata filename is the same. This is engineering
support, not evidence that selection improves prediction.

### Verification checkpoint — 2026-10-06

From the main WSL repository, with the ML interpreter:

```bash
python -m unittest discover -s tests -q
python scripts/ingredient_selection/verify_runtime.py
python scripts/validate_legacy_experiments.py
```

- All **140 repository tests pass**, including 27 new
  [runtime integration tests](../../tests/test_ingredient_selection_runtime.py).
  They cover full/default behavior, exact selected columns, empty rows,
  unknown labels, config and HPO serialization, checkpoint mismatch rejection,
  comparator cohort separation, and synthetic split/metadata immutability.
  A bounded synthetic CPU Lightning fit verifies full/light checkpoint reload
  and a subsequent optimizer-step resume. No real-data model is trained.
- The read-only parity command verifies all **47,965 train and 5,996 validation
  records**, exactly matching their original 165-label encoding at the saved
  selected indices. All **176 train and 18 validation all-zero selected rows**
  are retained. These counts describe projection, not new inclusion criteria.
  Input bytes are unchanged; the command opens no test metadata or image pixels.
- The Data 2.1c validator passes all **72 retained artifacts**, exact historical
  40-label reproduction, metadata compatibility and three executable checkpoint
  anchors. The journal has only the allowed append-only extension; writes are
  empty. An additional current-runtime CPU check loaded the saved H1 trial 21,
  H2 trial 0 and H3 trial 64 configurations/weights with
  `ExpConfig.load_from_file`/`load_from_ckpt_data`,
  `BaseLGNM.load_from_config` and `load_weights_from_checkpoint`. Historical
  HPO trial encoders were recovered from their saved parent `hparam_config.json`.
  Synthetic zero-image forward passes produced finite 183/183/41 outputs;
  all three checkpoint hashes stayed unchanged.
- The full suite includes a **pre-existing metadata-only real test-split
  vocabulary compatibility check**. The legacy validator also checks historical
  metadata/sample loading. Neither is predictive test evaluation. New real-data
  parity verification reads train/validation only; no test metric, campaign,
  HPO run, matched-random control or human review was performed.

### Historical retirement and retention

P7 retires superseded code **from the active workflow**, not from the evidence
archive. The verified Data 2.1c manifest remains authoritative:

| Material | Disposition | Maintained replacement or reason |
| --- | --- | --- |
| Five historical `scripts/analize_exps/*.ipynb` notebooks | Retain byte-identically; deprecated as executable workflow | Historical reconstruction uses `reproduce.py`; the current campaign uses maintained analysis/report/D6 commands. |
| H1–H4 launchers (`htuning_resnets.py`, `train_resnets_bs_f1_ings.py`, `htuning_resnets_sel_ings.py`, `test_best_for_f1.py`) | Retain byte-identically; not templates for new selection | Use `train_selector.py` for the frozen selector and canonical training configuration for the opt-in selected task. |
| Retained metadata, configs, checkpoints, metrics, provenance and D4/D6/P6 inputs | Retain unchanged | Required reproducibility and compatibility evidence, not redundant runtime files. |
| Other old checkpoints, profiling output or logs | Defer physical cleanup | A separate reviewed, exact-scope cleanup decision is still required; passing retention checks does not authorize deletion. |

No historical file is deleted, moved or rewritten. Retention-aware logical
retirement closes P7; disk cleanup is not a hidden dependency or permission.
Broader Data 2.4 GPU-training/dashboard smoke checks and the documented future
resource-gate device fix remain separate follow-ups.

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
- D4 intervals describe final-checkpoint record uncertainty; D6 estimates
  five-checkpoint median/paired-baseline uncertainty using exact-image groups.
  Neither estimates training-seed uncertainty.
- The cuisine prior is a mechanism diagnostic, not an image-model competitor.
- Checkpoints and generated outputs remain outside durable documentation.

P3 and the original P4 application are retained. The revised P4 checkpoint is
tracked in the [active plan](../plans/recognizable_ingredient_selection.md).
Mandatory P5 review is superseded, with its unannotated
packet and tools retained as an optional appendix. P6 published the shared
numerical-profile-based vocabulary definition and P7 integrated its explicit
runtime use with parity and legacy checks. Superseded scripts are retained as
evidence; physical cleanup requires a separate reviewed decision. No current report establishes direct
visibility. The incomplete v2 capacity gate was
interrupted before any v2 campaign started. The interrupted v1's artifacts
remain in the original report/experiment directories and are excluded from
replacement learnability evidence. The post-fit gate's CPU validation
limitation above remains explicit.
