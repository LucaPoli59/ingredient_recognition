# Phase 3 v3 inclusion-policy audit

**Created:** 2026-10-05
**Last updated:** 2026-10-05
**Status:** Retained post-P4 exploratory audit; its recommendation was adopted as D6 on 2026-10-05. The original counterfactuals below are not the revised numerical result or a final vocabulary.

## Question and scope

Does promoting only D4's 25 `generalizable_candidate` labels answer the
project's recipe-ingredient learning objective? The user requested a review
on 2026-10-04, explicitly accepting a small vocabulary if the inclusion policy
is justified. No desired number of ingredients is an optimization target.

The answer is **not without an explicit change of interpretation**. D4 was
implemented faithfully, but its intersection tests more than held-out recipe
prediction: it also demands a margin over privileged cuisine metadata, limits
train fitting relative to validation, and repeats a final-train support cutoff.
These restrictions require justification separately from the desired task.
The [adopted D6 decision](../project_objective/model_comparison_methodology.md#phase-3-d6--held-out-quality-inclusion-policy)
owns the policy; this record preserves its empirical motivation. The later
[D6 numerical result](phase3_d1_v3_d6_profile.md) applies corrected uncertainty
and reports 59 eligible labels; do not substitute the old-interval count 60.

## Evidence and reproduction

This is the same single-seed, 40-epoch `phase3-d1-v3` campaign and the same
165-label train/validation task reviewed in the
[original P4 report](phase3_d1_v3_full_profile.md). No training, inference,
test evaluation, manual review, or mutation of frozen artifacts was performed.
The original P3 pilot and D4 classification remain retained evidence.

The standard-library [audit script](../../src_scratches/ingredient_selection_policy_audit/audit_inclusion.py)
replays every original outcome and reason independently, verifies the campaign
and rule hashes, checks saved class order and all 6,930 metric rows, and emits
deterministic JSON with input hashes, independent failures, names and sensitivity
counts. Run from the repository root:

```bash
python3 src_scratches/ingredient_selection_policy_audit/audit_inclusion.py \
  analysis_outputs/ingredient_selection/phase3-d1-v3
```

Provenance remains:

- D4 canonical rule artifact hash: `7cf03371245860bf1a5be0c61a9fe54282e358fc910f21d1da6204f91354cda1`.
- `profile_evidence.csv` SHA-256: `5eb830a33192d0be77987e89dd4cb637f00ce714c31db0ab504c5f44a9239993`.
- `p4_profile_report.json` SHA-256: `30b9a285ad9cf88b0addf2f12c4d1a4842421997e32c3383bacf77da66eee7f9`.

## What actually restricts the original profile

All 165 labels have positive initialization-to-late train-AP gain, ranging
from `0.1599` to `0.9538`; none fails the `0.10` acquisition gate. No label
fails either `0.03` late-IQR gate. Maximum train and validation IQR are
`0.01771` and `0.01777`, respectively. Temporal oscillation therefore does
not explain the 83 uncertain outcomes.

Those 83 first-failure outcomes consist of 13 support exclusions, 50 validation
AP interval overlaps, 17 cuisine-advantage interval overlaps and 3 train–validation
gap flags. Of the 50 validation overlaps, 22 pass the point AP floor and 28
do not. Of the 17 cuisine overlaps, 10 pass its point margin and 7 do not.
Independent failures overlap: 72 labels exceed the train–validation gap,
but the branch order makes it the recorded first failure for only three.
The audit retains independent flags so a category name cannot hide other evidence.

### Acquisition is not the same as an absolute train-quality floor

The four `no_sustained_optimization` labels all fail only the train AP `0.35`
component of the optimization screen, despite gains between approximately
`0.191` and `0.321`. Canola oil reaches `0.349907`, just below the boundary.
All four also fall below the validation AP `0.20` point floor. Their original
category is reproducible, but must be read as *below the declared train-level
gate*, not as *no learning occurred*. The epoch-0 reference already precedes
all fine-tuning; the gain definition does not penalize learning before epoch 2.

### Cuisine is useful diagnostic information, not a fair hard comparator

The [problem definition](../project_objective/problem_definition.md) explicitly
allows dish-level and contextual inference. D1 itself calls ground-truth cuisine
an unavailable diagnostic for an image-only model. The following examples show
why exceeding that privileged baseline is a different requirement:

| Label | Late validation AP | Validation prevalence | Cuisine-prior AP |
| --- | ---: | ---: | ---: |
| Salt | 0.738 | 0.626 | 0.724 |
| Soy sauce | 0.637 | 0.118 | 0.590 |
| Fish sauce | 0.341 | 0.036 | 0.379 |

A model can exploit information in the photograph and still fail the `+0.10`
cuisine margin. Neither passing nor failing this comparison identifies the
mechanism or proves direct visibility. D4's `context_predictable` name should
be interpreted only as the recorded comparison outcome.

### Train–validation gap is not a monotone measure of held-out usefulness

With the same validation AP of `0.40`, a train AP of `0.95` fails the `0.50`
gap gate whereas `0.85` passes. Stronger train fitting alone is insufficient
reason to reject an otherwise equal held-out predictor. The three actual
first-failure cases are black olive (`0.859` train / `0.331` validation), mint
(`0.773` / `0.264`) and zucchini food product (`0.850` / `0.263`). They warrant
an overfitting diagnostic; their held-out evidence should be assessed directly.

## Policy effects using the existing uncertainty screen

The following are **counterfactual counts**, retaining the original late-median
AP and final-checkpoint interval. They do not implement the proposed corrected
resampling or define a final list. Changes are cumulative:

| Diagnostic policy | Labels passing | Change from preceding row |
| --- | ---: | --- |
| Unchanged D4 intersection | 25 | Original result |
| Cuisine comparison becomes diagnostic | 55 | 13 cuisine-margin failures plus 17 cuisine-overlap uncertain labels |
| Train–validation gap also becomes diagnostic | 58 | Black olive, mint and zucchini food product |
| Other train optimization metrics also become diagnostic | 58 | No additional label passes the held-out requirements |
| Final-train support 500 becomes diagnostic | 60 | Banana and strawberry |

Thus simply adding the 13 contextual labels to the 25 candidates would not
implement a consistent policy: the 17 uncertain labels on the same removed
cuisine criterion must be considered too.

The base vocabulary already passed the Data source-support policy. In the
final campaign, train support ranges from 454 to 30,023 and validation support
from 57 to 3,754. Reapplying 500 after final recipe filtering is a second
selection convention. The proposed alternative uses evidence validity and
sampling uncertainty, records support explicitly, and does not choose another
count threshold to admit particular ingredients.

## What can justify the validation quality floor?

Every label's late validation AP is above its validation prevalence. Moreover,
every existing epoch-40 AP lower bound exceeds the **point** prevalence, with
minimum excess `0.01375`. This is a descriptive check, not a paired confidence
interval or a simultaneous significance result. Nevertheless, a simple
point comparison with the constant-score baseline does not reduce this vocabulary.

The distinction must therefore be explicit: *detectable learning signal* versus
*sufficiently strong held-out ranking for the selected task*. AP `0.20` is an
inherited operational floor for the latter, not a scientific boundary of
learnability, a universal recommendation, or 20% classification accuracy.
AP depends on prevalence; both absolute level and baseline advantage should be
reported. [Saito and Rehmsmeier](https://doi.org/10.1371/journal.pone.0118432)
explain the prevalence-dependent precision–recall reference; the
[official AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
specifies the ranking summary.

With train/cuisine/support vetoes removed and the same inherited uncertainty
screen, the bounded sensitivity panel is:

| Validation AP floor (both median and existing lower bound) | Labels passing |
| --- | ---: |
| 0.15 | 83 |
| 0.20 | 60 |
| 0.25 | 41 |

The floor matters substantially. Retaining `0.20` avoids choosing a new value
to produce a preferred count, but does not prove it is optimal. Normalizing AP
by prevalence changes the scale and would require a further conventional
cutoff; it does not remove the need to define the study's intended quality.
Labels below the chosen floor can still have substantial relative improvement.

## Uncertainty correction required before a revised freeze

D4 compares a late-window median against an epoch-40 bootstrap interval, and
subtracts a point cuisine-prior AP from the image-only interval. Neither is a
formal interval for the tested quantity; extra exclusions do not guarantee
conservative coverage. Final versus late-median AP differs by `0.00108` at the
median and at most `0.01843`; no point estimate changes side of `0.20`. That
small observed difference does not establish that interval decisions are unchanged.

The proposal keeps the fixed five-checkpoint median and recomputes it within
paired resamples, including resampled prevalence for the constant-baseline
difference. Exact-image groups, where present, need to be the sampling units.
The metadata-only check found 5,996 distinct validation IDs and image filenames,
but no content hashes/group fields; distinct paths do not prove distinct bytes.
Validate the grouping from retained generation evidence or validation image
hashes before computing revised intervals. No test data is needed.

The near and late windows share four of five checkpoints. Their agreement is
a local sensitivity check, not independent replication. The revised intervals
also remain conditional on one trained model and nominal per-label coverage;
they cannot establish seed stability or joint certainty about the vocabulary.

## Review conclusion and boundaries

Reopen P4's inclusion interpretation while retaining its original execution as
completed evidence. The recommendation separates train acquisition, held-out
quality and mechanism diagnostics, retains the existing validation quality
floor provisionally, and fixes the uncertainty statistic before P6.
The value 60 above is **not** a selected-vocabulary recommendation or final
membership count. Corrected intervals and the separately recorded amendment
must precede any vocabulary export.

All 165 outcomes were already known when this review began. Any amended rule
is therefore exploratory and outcome-informed, even when justified by the
objective; it cannot be presented as an untouched pilot-confirmed rule.
[Cawley and Talbot](https://jmlr.org/papers/v11/cawley10a.html) explain why
selection criteria themselves can overfit finite validation evidence. Preserve
the original D4 outputs, report sensitivity and changed membership, keep the
test set isolated, and make the shared full-vocabulary comparison the unchanged
anchor. No guarantee of intrinsic learnability or direct visibility follows.
