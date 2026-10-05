# Phase 3-D1 v3 full numerical profile

**Created:** 2026-09-28
**Last updated:** 2026-10-05
**Status:** Provisional, full train/validation profile; not a selected vocabulary or test result.

## Purpose and boundary

Phase 3 P4 applies the [pilot-frozen D4 numerical rule](../project_objective/model_comparison_methodology.md#phase-3-d4--pilot-frozen-numerical-profile-rule) once to all 165 `v5` labels from the *same* completed `phase3-d1-v3` selector campaign. This report reviews that application. It does not revise a gate, train another selector, assess human observability, select a final ingredient tier, or use the test split. The [24-label pilot record](phase3_d1_v3_pilot.md) remains the separate pre-P4 evidence that fixed the rule.

**Review update, 2026-10-05:** the original execution and numbers below remain
unchanged. A separate [inclusion-policy audit](phase3_d1_v3_inclusion_policy_audit.md)
motivated the adopted D6 inclusion amendment. The separate
[D6 result](phase3_d1_v3_d6_profile.md) reports the revised policy with matching
uncertainty, while P6's final projection remains pending. Original D4 numbers,
groups and provenance below are unchanged.

## Inputs and reproducibility

- The frozen FoodOn-first `ingredients_target_v5_metadata.json` contributes 47,965 train and 5,996 validation records and 165 ordered output classes. The completed single-seed EfficientNetV2-S campaign ran 40 epochs with true FP32, effective batch 128, and deterministic audits at epochs 0, 2, …, 40. Its exact training, transform, weights, data and source provenance is in [`campaign_manifest.json`](../../analysis_outputs/ingredient_selection/phase3-d1-v3/campaign_manifest.json); the launch revision is `192059e`.
- The pre-frozen 24-row [`pilot_profile_evidence.csv`](../../analysis_outputs/ingredient_selection/phase3-d1-v3/pilot_profile_evidence.csv) and [`pilot_validation_summary.json`](../../analysis_outputs/ingredient_selection/phase3-d1-v3/pilot_validation_summary.json) were preserved before P4 replaced the working report. The rule hash remains `7cf03371245860bf1a5be0c61a9fe54282e358fc910f21d1da6204f91354cda1`.
- Reproduce P4 from the repository root by running `scripts/ingredient_selection/analyze_campaign.py analysis_outputs/ingredient_selection/phase3-d1-v3`, then `scripts/ingredient_selection/report_campaign.py analysis_outputs/ingredient_selection/phase3-d1-v3` with the maintained ML environment. The first command validates campaign identity, frozen rule, pilot evidence, ordered metadata, 21-point audit cadence, and final validation-score record order before exposing all labels. The second independently reclassifies all rows, confirms the 24 pilot decisions agree exactly, validates the trajectory table, and writes deterministic figures. The maintained reporting source SHA-256 is `9d28ba7c053d31bdd91fed042e68bb003b71ee467cd4cce3421bb37a1902036f`.
- The 165-row [`profile_evidence.csv`](../../analysis_outputs/ingredient_selection/phase3-d1-v3/profile_evidence.csv) has SHA-256 `5eb830a33192d0be77987e89dd4cb637f00ce714c31db0ab504c5f44a9239993`; [`validation_summary.json`](../../analysis_outputs/ingredient_selection/phase3-d1-v3/validation_summary.json) records `analysis_scope: full`, valid hashes/order/cadence and `test_split_accessed: false`. [`p4_profile_report.json`](../../analysis_outputs/ingredient_selection/phase3-d1-v3/p4_profile_report.json) has SHA-256 `30b9a285ad9cf88b0addf2f12c4d1a4842421997e32c3383bacf77da66eee7f9` and contains complete named provisional groups, reason counts, input/source hashes and figure hashes. Repeated report generation reproduced the JSON and all six figure files byte-for-byte.

## Reviewed findings

| Frozen-rule outcome | Labels | Primary reason or interpretation |
| --- | ---: | --- |
| `generalizable_candidate` | 25 | Meets all numerical optimization, validation, stability, support and image-advantage screens; **not** evidence of direct visibility. |
| `optimization_only` | 40 | Train signal exists, but validation AP is clearly below the absolute gate. |
| `context_predictable` | 13 | Held-out AP is adequate, but image advantage over the train-only cuisine-prior diagnostic is clearly below the gate. |
| `no_sustained_optimization` | 4 | Train gain or late train AP fails the absolute optimization screen. |
| `uncertain` | 83 | 13 low support; 50 bootstrap overlap at the validation-AP gate; 17 overlap at the image-advantage gate; 3 late-window or train–validation-gap flags. |

The 25 **numerical** candidates are avocado, baking powder, baking soda, beansprouts, black turtle bean, butter, carrot, cheese, cherry tomatoes, chicken, chicken broth, chickpea, cinnamon, corn, cucumber, egg food product, flour, milk, onion, pea, red bell pepper, red onion, shrimp food product, sugar, and vanilla extract. These names are not yet a published `V_selected`: in particular, a high image score for flour, chicken broth or vanilla extract is not proof that the ingredient itself is visible in a prepared-dish image. Under the 2026-10-04 D5 amendment, P6 defines the shared projection from numerical evidence without a manual observability gate.

The `context_predictable` group is chili pepper, cumin, fish sauce, garam masala, ginger, lime juice, olive oil, salt, scallion, sesame oil, soy sauce, turmeric, and yogurt food product. The four `no_sustained_optimization` labels are canola oil, chili, vegetable oil, and yellow onion. Complete membership, including the 40 optimization-only and 83 uncertain labels, is in the machine-readable report and per-label CSV rather than copied into this review.

All 24 pilot outcomes match their archived P3 decisions exactly. Over all 165 labels, train support ranges from 454 to 30,023, late train AP from 0.271 to 0.963, and late validation AP from 0.043 to 0.738. Spearman train-support versus late validation AP is `0.559`; train-support versus initialization-to-late train-AP gain is `-0.810`. These descriptive associations do not isolate support from prevalence, label semantics, or visual cues.

The reviewed figures show the two decisive held-out dimensions, support dependence, and fixed-state train/validation AP trajectories. Trajectory examples are chosen deterministically as the two labels nearest each outcome group's median late validation AP; they illustrate curve shapes and are not estimates of group-average behavior.

- [Numerical decision map](phase3_d1_v3_p4_decision_map.svg)
- [Support versus validation AP](phase3_d1_v3_p4_support_vs_validation_ap.svg)
- [Illustrative AP trajectories](phase3_d1_v3_p4_ap_trajectory_examples.svg)

## Interpretation limits and next gate

The labels are conditional on one pretrained selector, one split, one training configuration and seed, the fixed 40-epoch budget and D4's pilot-chosen absolute screens. The epoch-40 validation bootstrap is record-resampling uncertainty, not a formal interval for late-window median AP, model-minus-prior AP, training-seed variation or configuration stability. The cuisine prior uses metadata unavailable to the image-only model and is a mechanism diagnostic, not a fair deployable competitor. Neither image advantage nor AP establishes literal ingredient visibility. No selected-vocabulary retraining or matched random-reduction control was run; those belong to Macro-section 6 after a vocabulary is frozen.

The original P4 application is complete. [Phase 3-D5](../project_objective/model_comparison_methodology.md#phase-3-d5--numerical-selection-and-optional-interpretation-appendix), adopted on 2026-10-04, supersedes the former mandatory P5 semantic/observability gate. The prepared review remains an optional interpretation appendix. The subsequent [inclusion-policy audit](phase3_d1_v3_inclusion_policy_audit.md) led to the separately versioned [D6 policy and numerical review](phase3_d1_v3_d6_profile.md), now the P6 handoff. All reported original P4 values, groups and provenance remain unchanged; test outcomes and manual review cannot inform the projection.
