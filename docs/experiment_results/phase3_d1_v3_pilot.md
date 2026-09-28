# Phase 3-D1 v3 selector pilot

**Created:** 2026-09-28
**Last updated:** 2026-09-28
**Status:** Provisional, pilot-only validation evidence; not a final selected vocabulary or test result.

## Purpose and boundary

This record reviews only the sealed 24-label, train-support-stratified pilot of
the completed `phase3-d1-v3` reference-selector campaign. The other 141 label
outcomes remain unopened for the Phase 3 P4 application gate. The numerical
profile rule is binding in [Phase 3-D4](../project_objective/model_comparison_methodology.md#phase-3-d4--pilot-frozen-numerical-profile-rule),
while this document owns the observed pilot outcome and its limitations.

## Provenance and comparability

- Data: frozen FoodOn-first `ingredients_target_v5_metadata.json`, 47,965 train
  and 5,996 validation records, 165 ordered outputs. Selection and analysis did
  not access test metadata or predictions.
- Selector: full-adaptation EfficientNetV2-S with the exact supervised ImageNet
  weights, 384-pixel full-frame fit/pad, train-only horizontal flip, weighted
  BCE, AdamW, true FP32, seed 42, physical batch 8 and accumulation 16.
- Campaign: 40 complete epochs, 2 warm-up plus 38 cosine; fixed-state audits at
  epoch 0 and every two epochs through 40. It ran from
  `2026-09-27T16:17:14Z` to `2026-09-28T14:36:09Z` from Git revision `192059e`.
  The manifest identifies the exact source snapshot, weights, data, checkpoint,
  configuration and resource gate.
- Inputs: [`campaign_manifest.json`](../../analysis_outputs/ingredient_selection/phase3-d1-v3/campaign_manifest.json),
  [`pilot_cohort.json`](../../analysis_outputs/ingredient_selection/phase3-d1-v3/pilot_cohort.json),
  [`pilot_profile_evidence.csv`](../../analysis_outputs/ingredient_selection/phase3-d1-v3/pilot_profile_evidence.csv),
  and [`validation_summary.json`](../../analysis_outputs/ingredient_selection/phase3-d1-v3/validation_summary.json).
  The pilot evidence has SHA-256 `a6e544fd0be6af106103e60d6705fce78a9a0868c0e97d7ebfb10a7d334acf1a`.
- Frozen decision: [`profile_rule.json`](../../analysis_outputs/ingredient_selection/phase3-d1-v3/profile_rule.json)
  has content hash `7cf03371245860bf1a5be0c61a9fe54282e358fc910f21d1da6204f91354cda1`;
  the pilot-only [`pilot_profile_decisions.csv`](../../analysis_outputs/ingredient_selection/phase3-d1-v3/pilot_profile_decisions.csv)
  has SHA-256 `d2ea86b24fdfb7437de6fb62540fc2037e4c8b9588e220d84c1d083adbfad6bc`.
  Reproduce those decisions with
  [`report_pilot.py`](../../scripts/ingredient_selection/report_pilot.py),
  which reads only the archived pilot evidence and checks the rule, classifier
  source, campaign and cohort hashes.
- A separate post-campaign
  [`pilot_analysis_source.zip`](../../analysis_outputs/ingredient_selection/phase3-d1-v3/pilot_analysis_source.zip)
  preserves the exact classifier, analysis module and pilot-report script used
  for this rule; its SHA-256 is
  `5d1659465cb3acd9e2b51838a6be2f485c1655b9b78641f604ee164ec5591241`.
  It is distinct from the immutable training source snapshot in the campaign
  manifest because the bootstrap-overlap classifier was added after examining
  only the pilot evidence.

## Reviewed observations

All 24 pilot labels had valid deterministic 1,000-resample epoch-40 validation
AP bootstrap intervals. Train support ranged from 483 to 30,023 and validation
support from 61 to 3,754. Across the pilot, initialization-to-late train-AP
gain ranged from 0.160 to 0.868, late train AP from 0.363 to 0.878, and late
validation AP from 0.044 to 0.738. Late-window train and validation AP IQRs
were at most 0.016 and 0.009, respectively. These are descriptive ranges, not
evidence of seed stability.

Under the frozen absolute gates and conservative final-checkpoint bootstrap
band, the provisional outcome counts are:

| Outcome | Pilot labels | Interpretation |
| --- | ---: | --- |
| `generalizable_candidate` | 5 | Baking powder, beansprouts, chicken, corn, and shrimp food product pass the numerical profile; none is yet certified directly visible. |
| `optimization_only` | 5 | Coconut oil, nutmeg, pork, rice, and vegetable broth show train acquisition but validation AP is clearly below the absolute gate. |
| `context_predictable` | 1 | Salt's validation AP is high, but its image-versus-cuisine-prior advantage remains below the required margin even at the bootstrap upper bound. |
| `uncertain` | 13 | Three have final train support below 500; five overlap the validation-AP gate; five overlap the image-advantage gate. No borderline case was forced into a retained-count quota. |

On these 24 labels, Spearman support versus late validation AP was `0.618`,
and support versus train-AP gain was `-0.729`. These associations are
descriptive and do not isolate support from label semantics or prevalence.
The fixed-0.5 F1 trajectory remains a diagnostic; it did not affect the rule.

## Interpretation and limitations

The profile evaluates conditional learnability under this one pretrained
selector, split, 40-epoch budget and seed. The bootstrap describes variation
from resampling validation records at the final checkpoint; it is not an
interval for training-seed or configuration variability. Comparing that final
interval with late-window medians and a point-estimated cuisine prior is a
conservative decision screen, not a formal confidence interval for either
median or model-minus-prior difference. The cuisine prior uses metadata that
the image-only selector does not receive and is a mechanism diagnostic, not a
deployable fair competitor. Neither AP nor image advantage proves that an
ingredient is directly observable in a photograph.

The pilot does not establish a final `V_selected`, a performance benefit from
reducing the label space, or a model-category ranking. Those claims need the
separate Phase 3 P4–P6 and Macro-section 6–7 gates. No result from the other
141 labels was used to choose the rule.
