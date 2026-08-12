# Comparative model and vocabulary-reduction methodology

**Created:** 2026-08-12
**Last updated:** 2026-08-12
**Status:** Active and binding design; execution is deferred until Macro-section 4 selects the reference selector.

## Purpose and scope

This document defines the research methodology that couples ingredient selection
to model training and comparison. It is the binding owner of the distinction
between:

1. comparing model categories fairly on a common prediction task;
2. testing whether a learnability-selected vocabulary helps beyond an arbitrary
   reduction in output labels; and
3. obtaining a well-adapted model for the reduced task.

It applies to Macro-sections 3, 4, 6, and 7 of
[general_plan.md](../general_plan.md). The selection workflow remains owned by
Macro-section 3 and its operational plan. Training implementation, hyperparameter
optimization (HPO), and final result production remain owned by Macro-sections 6
and 7. This document does not prescribe a particular model, numeric threshold,
HPO budget, or final selected vocabulary.

## Research questions

The methodology keeps these questions separate because they require different
comparisons.

| ID | Question | Required common task |
| --- | --- | --- |
| Q1 | Which model category performs best on the complete normalized ingredient task? | The frozen full v5 vocabulary. |
| Q2 | Which model category performs best on the learnability-selected task under the matched transferred procedure? | One shared selected vocabulary. |
| Q3 | Does the selected vocabulary help more than reducing the number of labels at random? | Like-for-like selected and support-matched random vocabularies, evaluated by the reference model. |
| Q4 | What is the best performance a model can attain after adapting its hyperparameters to the selected task? | The shared selected vocabulary, reported separately from Q3. |

Q1 and Q2 compare model categories. Q3 is a vocabulary-reduction ablation. Q4
is an optimized reduced-task result. Conflating them would either compare
different prediction problems or attribute a hyperparameter change to the
vocabulary selection.

## Terms and fixed boundaries

| Term | Meaning |
| --- | --- |
| V_base | The frozen 165-label FoodOn-first v5 vocabulary derived from ingredients_target. It is the common full task. |
| M_ref | The reference selector architecture and protocol selected by Macro-section 4 before Macro-section 3 resumes. It is a selection instrument, not the automatically preferred final model. |
| V_selected | A versioned, shared projection of V_base produced by Macro-section 3 with M_ref, the learnability decision profile, semantic evidence, and observability review. |
| V_random^(r) | One deterministic random projection of V_base with the same cardinality as V_selected and support strata matched to it; r identifies the draw. |
| H_base(m) | Hyperparameters selected for model category m on V_base using validation only and the predeclared Phase 6 budget. |
| H_local(m) | A small, predeclared local adaptation panel around H_base(m) for V_selected. It is not a second unrestricted HPO campaign. |
| Q_q | One of the frozen headline metrics from D9: label-macro mAP or micro F1. Both are reported; no single scalar silently replaces the pair. |

Every selected or random vocabulary is an explicit experimental projection. It
never replaces the default ingredients_target vocabulary, and validation or test
labels never expand or reorder its saved class order.

## Binding design

### 1. Select the reference selector before vocabulary selection

Macro-section 4 must choose M_ref before new v5 ingredient-selection training
begins. The decision is based on focused model research and a declared selection
protocol: scientific fit to multi-label visual learnability, availability of
per-label score trajectories, representativeness, compute cost, and integration
feasibility. It must not be chosen because it later produces the most favourable
final test result or the largest selected vocabulary.

Macro-section 3 then owns the execution: it freezes the M_ref configuration,
runs the learnability profile without test access, and produces exactly one
shared V_selected plus the associated evidence and provenance. A different
selected vocabulary for each model category is rejected for the primary study,
because it would make a model comparison a comparison of different tasks.

Macro-section 4 may still select several model categories for the benchmark.
M_ref is not thereby declared the winning category; it only fixes the operational
meaning of “learnable” for the vocabulary-selection study.

### 2. Tune each model category once on the full common task

For every approved model category m, Macro-section 6 performs one bounded HPO
campaign on V_base and freezes H_base(m) from validation data. The search
objective, budget, transforms, loss policy, early-stopping rule, and one declared
seed per configuration must be fixed before trials begin. The HPO objective must
be compatible with the paired D9 evaluation policy; the current implementation's
val_loss optimization is not by itself a final methodological choice.

These full-vocabulary selected configurations answer Q1. They also provide the
common starting point for the vocabulary-reduction ablation. No model category
is tuned separately for each ingredient, and no test result selects an
architecture, hyperparameter, threshold, or vocabulary.

### 3. Measure vocabulary reduction with transferred hyperparameters

For every approved model category, train one new V_selected run using its
unchanged H_base(m). Compare it with the corresponding full-vocabulary model
after restricting the full-vocabulary evaluation to the same labels in
V_selected. This is the required transfer ablation for Q2 and, for M_ref, the
selected-vocabulary side of Q3.

For each headline metric q, the vocabulary effect for a vocabulary V is:

\[
\Delta_q(V) =
Q_q\!\left[\operatorname{train}(M_{ref}, V, H_{base}(M_{ref})); V\right]
-
Q_q\!\left[\operatorname{train}(M_{ref}, V_{base}, H_{base}(M_{ref})); V\right].
\]

The notation after the semicolon denotes that both models are evaluated only on
the same vocabulary V. For each model category, the analogous restricted-set
comparison is reported. The random-vocabulary control below is run only for
M_ref unless a later, separately budgeted study expands it.

Transferred-hyperparameter results answer a deliberately narrow causal question:
under the same training procedure, does changing the output vocabulary change
performance on the retained labels? They do not establish that H_base(m) is
optimal for the smaller vocabulary.

### 4. Control for arbitrary label removal

Macro-section 6 generates several deterministic V_random^(r) controls before
their runs. Each has the cardinality of V_selected and is stratified by train and
validation positive-support tiers; prevalence matching and semantic-type matching
are added when the final selected set makes either necessary for a credible
comparison. The number of draws and the training budget are frozen in the Phase
6 plan before any control outcome is inspected.

Using M_ref and the unchanged H_base(M_ref), run the same transfer ablation for
every V_random^(r). Report Δ_q(V_selected) alongside every Δ_q(V_random^(r)), for
both mAP and micro F1, rather than comparing absolute scores from different
label sets. This asks whether removing the selected labels helps more than
removing an equally large, comparably supported set of labels by chance.

The controls are an empirical reference distribution, not replicated
stochastic-training runs. With a small number of draws they support a descriptive
comparison, not a formal significance claim. They also do not prove that every
retained ingredient is directly visible.

### 5. Keep the selected-task local adaptation separate

Removing outputs can alter the optimization problem: in this repository,
BCEWithLogitsLoss uses mean reduction by default, and optional pos_weight values
are recomputed from the active output labels. The gradient balance and the best
learning rate, weight decay, momentum, or scheduler can therefore change when
moving from about 165 labels to a selected set of roughly 50.

After the transfer ablation is frozen, Macro-section 6 may run a small local
adaptation panel around H_base(m) on V_selected. Its candidates and equal
per-category budget must be declared in advance; typical dimensions are a small
multiplicative learning-rate neighbourhood and a bounded weight-decay
neighbourhood. It is not an unrestricted second Optuna study. If an optimized
selected-task headline is reported for several model categories, every category
must receive the same declared local-adaptation opportunity.

H_local(m) answers Q4 only. It must never be used to calculate the Q3
selected-versus-random effect, because it changes both the vocabulary and the
training procedure. If the schedule cannot accommodate the local panel, the
project may report the transferred result but must not describe it as the best
attainable selected-task configuration.

### 6. Preserve selection isolation and report the single-run limit

All vocabulary selection, HPO, local adaptation, control design, thresholds, and
calibration use training and validation data only. Test data is accessed only
after these choices and their report schemas are frozen; it is never used to
revise a vocabulary, selector, model category, hyperparameter, or control.

The project has a resource constraint of one declared seed per configuration.
This applies to selection, full-vocabulary HPO confirmation, transferred
ablation, random controls, and local-adaptation runs. Temporal-window checks,
configuration sensitivity, and resampling of held-out predictions may quantify
limited evidence, but they do not estimate run-to-run stochastic variability.
Every final table and claim must state this limitation and classify borderline
vocabulary decisions as uncertain.

## Rejected primary designs

| Design | Why it is not the primary methodology |
| --- | --- |
| Freeze one selected vocabulary using an arbitrary model without Macro-section 4 review | The learnability conclusion is model-conditional; the selector must be explicitly justified and frozen first. |
| Select a different vocabulary for each model category after full-vocabulary tuning | It changes the prediction task by category, so model rankings cannot be interpreted as a like-for-like comparison. |
| Aggregate model-specific selected vocabularies after every category has trained, then use the aggregate as the primary benchmark | It delays the shared task until after model outcomes influence it and doubles the campaign before a comparable benchmark exists. It may be a separately labelled exploratory study later. |
| Perform a full second HPO on every selected and random vocabulary | It is too costly for the thesis schedule and confounds the primary vocabulary-effect ablation. |
| Transfer H_base(m) and call the result optimized for V_selected | Output reduction can alter loss scaling and class weights; transferred results are causal ablations, not selected-task optima. |
| Use test results to choose the selector, selected vocabulary, HPO values, local panel, or random controls | It leaks final evaluation information into research decisions. |

## Evidence and rationale

| Evidence | Consequence for this design |
| --- | --- |
| [label_learnability/learnability_assessment.md](../research/topics/label_learnability/learnability_assessment.md) | Learnability is conditional on the declared learner and protocol, so the project must name and freeze a reference selector. Train AP, validation AP, mechanism controls, and observability are separate evidence. |
| [Cawley and Talbot (2010)](https://jmlr.org/papers/v11/cawley10a.html) | Performance estimates with finite-sample variance can be overfit during model selection; selection criteria, vocabulary decisions, and final evaluation must be separated. |
| [Varma and Simon (2006)](https://doi.org/10.1186/1471-2105-7-91) | Reusing the data that selected an optimized configuration as its final error estimate is optimistic; keep test data isolated from all vocabulary and HPO decisions. |
| [Ambroise and McLachlan (2002)](https://doi.org/10.1073/pnas.102102699) | Selecting a subset based on observed predictive evidence is analogous to feature selection and requires its own validation boundary. |
| [PyTorch BCEWithLogitsLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) and the current [loss setup](../../src/lightning/lgn_models.py) | Mean reduction and active per-class positive weights make a reduced output space a changed optimization problem, motivating the transfer-ablation/local-adaptation separation. |

## Execution dependencies and open decisions

| Owner | Required decision or artifact | Status |
| --- | --- | --- |
| Macro-section 4 | Choose and justify M_ref; define the model categories to compare. | Pending; [`reference_selector_research.md`](../plans/reference_selector_research.md) is the resume-gate plan for Macro-section 3. |
| Macro-section 3 | Freeze the M_ref learnability protocol and produce versioned V_selected evidence. | Deferred until M_ref is selected. |
| Macro-section 6 | Freeze HPO objectives/budgets, random-control count and matching rules, transfer runs, and any equal local-adaptation panel. | Deferred until the selected vocabulary and models are available. |
| Macro-section 7 | Freeze report schemas, evaluate the already selected configurations on test, and keep Q1–Q4 result statements separate. | Deferred until Macro-section 6 completes. |

## Related documentation

- [problem_definition.md](problem_definition.md)
- [benchmark_decisions.md](benchmark_decisions.md)
- [general_plan.md](../general_plan.md)
- [reference_selector_research.md](../plans/reference_selector_research.md)
- [recognizable_ingredient_selection.md](../plans/recognizable_ingredient_selection.md)
- [label_learnability/README.md](../research/topics/label_learnability/README.md)
