# Phase 6 budget baseline from retained campaigns

**Created:** 2026-10-13
**Last updated:** 2026-10-13
**Status:** Reviewed historical cost evidence and explicitly conditional planning arithmetic; not a campaign budget or performance benchmark

## Purpose and provenance

Answer whether the historical ResNet/DINOv2 allocation can price the new portfolio. The answer is **not directly**: nominal trial counts, actual model allocation, pruning, adaptation and numeric precision differ. This review does not launch a timing run or repeat Phase 5 qualification.

The standard-library [audit helper](../../src_scratches/phase6_budget_audit/audit.py) and [retained summary](../../src_scratches/phase6_budget_audit/summary.json) reproduce the figures from:

- the existing [200-trial artifact inventory](../../src_scratches/experiment_comparison_audit/inventory.json), with the actual historical journal and every saved trial configuration checked against its hashes;
- six `experiments/runtime_smoke/phase55-20261007-a4/*/smoke-*/result.json` files from [Phase 5 qualification](../implementation_details/experimental_model_contract.md#measured-cuda-acceptance-and-readiness-matrix);
- timestamps only from `analysis_outputs/ingredient_selection/phase3-d1-v3/campaign_manifest.json`.

The journal SHA-256 is `7d89be6b4297284494702c8224243f95819abfc6e6522a49b3842f1ef0910351`. The summary records exact hashes for all inputs. This read-only review was made against base revision `e0653dd` on 2026-10-13; it did not open checkpoint tensors, predictions or dataset images, including test images. It does not overwrite or replay the historical journal. The older [artifact audit](../implementation_details/experiment_artifacts.md) remains authoritative for restart/session coverage.

## Historical recorded cost

| Journal cohort | ResNet study | DINOv2 study |
| --- | ---: | ---: |
| Started trials | 100 | 100 |
| Sum of recorded trial duration | 41.98 h | 106.91 h |
| COMPLETE trials / duration sum | 22 / 22.68 h | 37 / 71.44 h |
| PRUNED trials / duration sum | 78 / 19.31 h | 63 / 35.46 h |
| Median COMPLETE trial duration | 61.10 min | 114.12 min |
| Median PRUNED trial duration | 12.08 min | 22.49 min |

No trial duration is missing in the retained journal inventory. Nevertheless, these are **recorded end-to-end trial durations, not GPU-active hours or a verified total of all historical execution**. Resumed/re-executed sessions can leave work outside a single duration. The combined 148.89 recorded hours correspond arithmetically to 6.20 continuous days; this is not an elapsed-campaign reconstruction.

The ResNet study contains 15 DummyBN trials (4.29 recorded hours), 28 ResnetLikeV1 trials (11.64 hours) and only 57 pretrained ResNet18 trials (26.06 hours). All 100 DINO trials used the frozen-backbone protocol. Therefore neither “100 ResNet18 trials” nor “200 completed 40-epoch runs” describes the historical investment.

The [historical result](basic_v5_resnet_dinov2.md) remains unchanged: loss-selected ResNet trial 77 and frozen DINO trial 61 are exploratory anchors, not retrospectively mAP-selected winners. The journal durations include weighted and unweighted loss choices; this cost cohort is intentionally broader than the unweighted predictive comparison cohort.

## New-model evidence already available

All six smokes used full adaptation, true FP32, effective batch 128, four optimizer updates over 512 train records and bounded validation. They verify feasibility, not full-epoch throughput. The measured `training_seconds` includes the bounded fit/validation/logging path, not model initialization, restoration and dashboard diagnostics.

| Protocol | Physical batch / accumulation | Full165 smoke | Selected59 smoke |
| --- | --- | ---: | ---: |
| EfficientNetV2-S experimental, 224 pixels | 32 / 4 | 22.39 s | 21.89 s |
| MaxViT-T, 224 pixels | 16 / 8 | 41.27 s | 39.70 s |
| DICA-Net-S, 224 pixels | 32 / 4 | 23.56 s | 23.08 s |

The selected smokes are only about 2–4% faster in these single observations. This is not a speedup estimate with uncertainty. It is evidence against assuming a reduction from 165 to 59 outputs makes a new run cost only `59/165` of a full run: most encoder computation and all input records remain. Price selected/random runs approximately as full runs until actual campaign timing replaces this assumption.

The completed 384-pixel selector took **22.32 hours of manifest wall time**, from 2026-09-27T16:17:14.169539+00:00 to 2026-09-28T14:36:09.010771+00:00, including its periodic audits. This is a different head/loss/input/batch protocol; do not use it as a measured 224-pixel portfolio rate or equate its wall time with GPU-active time.

## Conditional arithmetic, not an ETA

To expose the scale risk without launching another qualification, the helper computes:

`median(last two observed optimizer-update seconds) × 375 updates/epoch × 40 epochs`.

Under the **unverified assumption** that those two update times persist, the update-only arithmetic is 15.42 h for experimental EfficientNet, 36.86 h for MaxViT and 16.75 h for DICA: **69.02 h for one full-horizon run of each**. These numbers exclude full validation, new metric aggregation, checkpoint I/O and interruptions; cold/warm effects can also change the update rate in either direction. They are neither lower bounds, confidence intervals nor an approved cost forecast.

For the plan's illustrative trial caps, the same deliberately unpruned sensitivity gives:

| Trial cap per new model | Conditional days for the three new full-task studies only |
| --- | ---: |
| 20 | 57.52 |
| 40 | 115.03 |
| 100 | 287.59 |

This table is **not expected campaign duration**. It excludes pruning savings as well as anchors, selected transfers, reference random controls, optional adaptation and inference. It shows why copying 100 trials or committing even 20 full runs per new model cannot be justified by the old six-day journal total. A resource-aware allocation must specify both candidate opportunity and the amount of training actually permitted per candidate.

## Planning implications and subsequent decision

1. Keep the old studies unchanged as historical evidence. A matched new comparison needs fresh compatible anchor studies; new validation inference alone can change reported metrics but cannot erase their old search/selection exposure.
2. Reserve cost for the required selected transfers and Q3 controls **before** allocating all available time to full-task HPO. The random controls belong to one declared reference, not every architecture. Optional local adaptation is the first explicit omission to consider if funds are tight.
3. Use a compact search and declared pruning/stopping policy if affordable. Do not promise historical pruning fractions for different models or count every pruned run as a full result. Keep a policy-consistent completion route for the selected artifact.
4. Record actual epoch costs from the first predeclared campaign trials, with a time-only review boundary before further expenditure. This is campaign accounting, not a new prerequisite qualification or an opportunity to adapt the protocol to per-label outcomes.
5. **Subsequent user decision, 2026-10-13:** continuous GPU use is available without a total hour/day ceiling, while individual training/search limits must remain finite and broadly DINO-like. The user also chose fixed-selector reuse for Q3. These decisions do not make the timing extrapolation a forecast. The adopted100-attempt/40-epoch policy, five controls and bounded local panel are owned by [Phase 6-D1](../project_objective/model_comparison_methodology.md#phase-6-d1--finite-training-and-comparison-protocol), not inferred from this empirical audit.

The [Phase 6 plan](../plans/training_hyperparameter_tuning.md#61--freeze-matrix-budget-and-protocol) owns execution and measured-cost releases. This audit remains retained cost evidence; it does not authorize training or establish a deadline estimate.
