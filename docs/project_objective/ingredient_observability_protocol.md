# Optional appendix: ingredient relevance and observability

**Created:** 2026-09-29
**Last updated:** 2026-10-04
**Status:** Optional future interpretation appendix. The mandatory P5 selection gate was superseded on 2026-10-04; no human annotation exists.

## Purpose and boundary

This appendix preserves the former P5 review design for optional interpretation of future results. The project predicts *recipe-level* ingredients from a finished-dish image; an ingredient can be a useful target without being directly visible. Under [Phase 3-D5](model_comparison_methodology.md#phase-3-d5--numerical-selection-and-optional-interpretation-appendix), manual semantic and visual review is outside primary ingredient selection. It cannot change `ingredients_target`, the split, D4 gates, vocabulary membership, tuning or primary model comparisons. P6 determines the shared projection from numerical evidence without waiting for this appendix.

Keep three questions separate: (1) is the normalized concept a valid and scientifically relevant recipe-ingredient target, (2) what information is available to a person from an individual photograph, and (3) did the frozen image model learn a held-out signal? Neither image-model AP nor a human's dish-based inference proves literal visibility or label correctness.

## Semantic relevance rubric

Review the concept definition, mapping rules and representative *training/validation* source ingredient lines, without model scores or test data. Record a rationale and supporting examples for each label under review. Do not use frequency or P4 outcome as the definition of relevance.

| Field | Allowed values | Meaning |
| --- | --- | --- |
| `concept_status` | `valid_specific`, `valid_broad`, `mapping_ambiguous`, `invalid_target` | Whether the `v5` concept is a defensible ingredient target. `valid_broad` permits an intentional generic class such as `cheese`; `mapping_ambiguous` calls for a mapping audit rather than a silent exclusion. |
| `thesis_role` | `direct_visual_question`, `contextual_recipe_question`, `uncertain_role`, `outside_scope` | Which research question this concept could serve. A hidden ingredient may be valuable for *contextual recipe prediction* even when unsuitable for a literal-visibility claim. This is a semantic hypothesis, not an observability verdict. |
| `rationale` | Non-empty free text | Explain concept boundaries, source-line evidence, uncertainty and any known mapping ambiguity. |

These judgments are descriptive appendix evidence, not ingredient-selection exclusions. A broad or hard-to-see ingredient is not invalid merely because it is common, transformed, or not directly visible. Suspected mapping defects go back to the [`v5` mapping authority](../implementation_details/ingredient_mapping_rules.md) for a separate data-version decision; this review cannot edit targets or vocabulary membership.

## Instance-level visual categories

Each reviewer sees the *same* ingredient–photograph pairs, in a different deterministic order, but no recipe name, ingredient list, target-presence flag, model prediction, numerical outcome or other reviewer's answer. The question is what the photograph alone supports:

| Category | Decision rule |
| --- | --- |
| `direct` | The ingredient itself, or a visually identifiable form of it, has visible evidence. Generic dish identity alone is insufficient. |
| `contextual` | Dish appearance makes the ingredient plausible, but the ingredient cannot be individually identified in the pixels. |
| `not_inferable` | The photograph gives no meaningful cue for that ingredient. |
| `uncertain` | Quality, occlusion, look-alike ingredients, preparation state or concept ambiguity prevents a defensible choice; add a short note. |

For example, a baked cake does not make `flour` directly visible; it may provide contextual evidence. Melted or mixed forms need a defensible ingredient-specific visual cue before `direct`. The reviewer must not infer presence from knowledge of the recipe label. Annotate both recipe-positive and recipe-negative pairs blindly: recipe absence is *not* guaranteed visual absence because supervision is weak and incomplete. No annotation overwrites the dataset target.

## Retained former P5 pilot packet

The pilot is a *rubric and agreement check*, not a representative per-label observability estimate. Its eight purposively chosen labels span P4 profiles and plausible direct/hidden cases: `avocado`, `cheese`, `chicken broth`, `flour`, `vanilla extract`, `salt`, `soy sauce`, and `tomato`. The packet takes six target-present and two target-absent validation records per label: 64 distinct photographs, 48 recipe-positive pairs and 16 negative audit pairs. There is no test access. For each label/presence stratum, SHA-256 of pilot ID, label and record ID defines the rank; images already used for another pair are skipped. Source image bytes, validation metadata, class order and P4 provenance are checked and hashed. Reviewers are blind to the stratum and to P4 outcome.

The immutable packet is generated by [`observability_review.py`](../../scripts/ingredient_selection/observability_review.py) into `analysis_outputs/ingredient_selection/phase3-d1-v3/p5_observability/pilot/`. Its unblinded `packet_manifest.json` must stay away from reviewers until both exports are complete. Provide `reviewer_a.html` to one person and `reviewer_b.html` to a *different* person, each with access to the local images referenced by the packet. The HTML stores work locally and exports a JSON answer file; reviewer names/pseudonyms, answers and notes are not committed to Git.

After both independent submissions, the scorer validates every pair ID and category against the packet hash, rechecks source-image bytes, requires distinct reviewer identifiers, then reports the full four-category confusion matrix, observed agreement and unweighted Cohen's kappa. Distinct identifiers cannot establish actual independence; the study coordinator must ensure different people work separately. Both statistics are descriptive; kappa may be sensitive to category prevalence. Per-label tallies on recipe-positive pairs count only *agreed* categories and disagreements. They are not a consensus adjudication, a confidence interval, or a final direct-visible tier. Reviewers can discuss disagreements only after the two raw submissions are sealed; any adjudication must be separately logged, never substituted for raw agreement.

## Optional future execution and limitations

No annotation is required to complete ingredient selection or the primary benchmark. The retained 64-pair packet has no reviewer submissions or measured agreement; preparing it does not establish observability.

If this optional appendix is undertaken later, inspect pilot disagreements and rubric usability before declaring a larger sampling panel. Record actual independent submissions, raw agreement, any later adjudication, uncertain cases and sampling limitations. The existing two-reviewer scorer requires two distinct human submissions; this is a condition for using that optional tool, not a project dependency. Synthetic or model-generated reviews must not be reported as human agreement. Neither the purposive pilot nor reviewer consensus establishes the model's causal mechanism.

The former mandatory semantic review and visual-annotation exit gate is **Superseded**, not completed. On 2026-10-04 the user moved both reviews outside the primary study because its objective is model learnability on the existing ingredient set and subjective review could bias selection. The main-panel design remains unspecified and optional; there are no promotion thresholds from this appendix.

## Related records

- [Research objective](problem_definition.md) and [benchmark decisions](benchmark_decisions.md) define recipe-level targets and test isolation.
- [Numerical profile decision](model_comparison_methodology.md#phase-3-d4--pilot-frozen-numerical-profile-rule) and [P4 result](../experiment_results/phase3_d1_v3_full_profile.md) provide the unchanged model evidence.
- [Phase 3 feature plan](../plans/recognizable_ingredient_selection.md) owns P5 execution status.
- [Current implementation](../implementation_details/ingredient_selection.md) records the packet and scorer contracts.
