# Experiment results

**Created:** 2026-09-23
**Last updated:** 2026-10-05

## Context and scope

This directory preserves reviewed, reproducible findings from named experiment campaigns. Each record identifies the exact task, artifacts, comparison cohort, analysis method, result status, limitations, and relationship to the binding benchmark methodology.

Results stored here are empirical evidence. They do not redefine the research objective, benchmark policy, or project status. A validation-only or historical comparison must remain explicitly marked as exploratory or historical and cannot be promoted to a final benchmark result without satisfying the gates in [`../project_objective/model_comparison_methodology.md`](../project_objective/model_comparison_methodology.md) and [`../general_plan.md`](../general_plan.md).

## What belongs here

- reviewed outcomes from a named training, HPO, ablation, or evaluation campaign;
- quantitative comparisons whose input artifacts and cohort rules are reproducible;
- interpretation of learning curves, parameter dynamics, compute cost, and failure evidence;
- explicit statements of whether a result is exploratory, historical, provisional, or final.

Raw JSON/HTML reports, checkpoints, event files, W&B stores, exhaustive tables, and temporary plots remain in their experiment or analysis-output locations. Implementation behavior remains in [`../implementation_details/`](../implementation_details/README.md), methodology decisions remain in [`../project_objective/`](../project_objective/README.md), and execution state remains in [`../plans/`](../plans/README.md) and [`../general_plan.md`](../general_plan.md).

## Files

- [`basic_v5_resnet_dinov2.md`](basic_v5_resnet_dinov2.md) records the exploratory validation comparison of the existing `basic_v5` ResNet18 and frozen-backbone DINOv2-B/14 HPO campaigns. It identifies ResNet18 trial 77 as the stronger observed artifact while retaining the final-benchmark gate.
- [`basic_v5_resnet_dinov2.png`](basic_v5_resnet_dinov2.png) is the compact, versioned figure derived from the same reviewed comparison.
- [`phase3_d1_v3_pilot.md`](phase3_d1_v3_pilot.md) records only the sealed 24-label pilot of the completed 40-epoch reference-selector campaign, the provisional numerical-profile outcomes, exact artifact provenance, and interpretation limits; the other 141 labels remain outside this review.
- [`phase3_d1_v3_full_profile.md`](phase3_d1_v3_full_profile.md) reviews P4's unchanged application of the pilot-frozen rule to all 165 labels; its five outcomes are provisional and do not constitute a selected vocabulary.
- [`phase3_d1_v3_inclusion_policy_audit.md`](phase3_d1_v3_inclusion_policy_audit.md) preserves the post-P4 objective-alignment audit and old-interval counterfactuals that motivated the subsequently adopted D6 amendment.
- [`phase3_d1_v3_d6_profile.md`](phase3_d1_v3_d6_profile.md) reviews the adopted D6 profile with matching paired image-cluster uncertainty: 59 eligible, 60 uncertain and 46 below the operational quality floor. It retains D4 comparison/sensitivity and records P6 publication of the shared versioned projection.
- [`phase3_d1_v3_d6_decision_map.svg`](phase3_d1_v3_d6_decision_map.svg) shows the reviewed D6 AP/prevalence/uncertainty outcomes and fixed floor sensitivity.
- [`phase3_d1_v3_p4_decision_map.svg`](phase3_d1_v3_p4_decision_map.svg), [`phase3_d1_v3_p4_support_vs_validation_ap.svg`](phase3_d1_v3_p4_support_vs_validation_ap.svg), and [`phase3_d1_v3_p4_ap_trajectory_examples.svg`](phase3_d1_v3_p4_ap_trajectory_examples.svg) are the compact, reproducible figures reviewed with that result.

## Maintenance rules

Every result record must use repository-relative paths, state its evidence cutoff, distinguish direct measurements from interpretation, and describe missing uncertainty or test evidence. Add a new record only for a distinct campaign or analysis question; update the existing record when rerunning the same campaign under the same result identity. Superseded records remain available and link to their replacement.

The documentation taxonomy and common rules are defined in [`../README_DOCS_ORGN.md`](../README_DOCS_ORGN.md) and [`../README.md`](../README.md).
