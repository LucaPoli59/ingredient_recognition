# Assessing label learnability in supervised classification

**Research date:** 2026-08-12
**Last updated:** 2026-08-12
**Status:** Evidence synthesis and reusable recommendation; not a binding project decision.

## Question and boundary

The practical question is: *under a declared learner, data split, training
budget, and annotation regime, is there reproducible evidence that a classifier
can learn a useful signal for label* `l`*?*

There is no single, model-independent, standard scalar called “label
learnability”. A label that is unlearnable by one architecture, loss,
augmentation policy, amount of data, or compute budget can become learnable
after any of those conditions changes. Conversely, a label can be predictable
from context or label co-occurrence without being directly observable in the
input. A useful assessment must therefore state the experimental conditions and
separate several kinds of evidence.

This record concerns supervised classification generally, with special
attention to imbalanced multi-label data. It does not establish that a label is
visually recognisable by a person, correctly annotated, causal, or suitable for
a particular application. Those are additional questions.

## Executive conclusion

Use a **learnability profile**, not the maximum F1 score reached during
training. The minimum useful profile has three distinct parts:

1. **Optimization:** did the model acquire a label-specific training signal
   during the declared budget?
2. **Generalization:** does the signal rank held-out positives above negatives
   reproducibly?
3. **Validity and mechanism:** is the apparent signal robust to support,
   imbalance, seed variation, non-visual baselines, and—when important—human
   observability review?

Class-wise average precision (AP) on a held-out split is the preferred primary
evidence of ranking/generalization. A trajectory of class-wise training AP can
be a useful optimization diagnostic. F1 is retained as a secondary metric for a
specific, predeclared binary decision policy; it is not the primary measure of
whether a label has become learnable.

## Evidence reviewed

| Source | What it establishes | Reusable implication | Important boundary |
| --- | --- | --- | --- |
| [Swayamdipta et al., 2020](https://aclanthology.org/2020.emnlp-main.746/) | Per-example confidence and variability across training epochs reveal easy, ambiguous, and hard training examples. | Training dynamics are informative diagnostics beyond final accuracy. | The work is single-label NLP. Aggregating these measures from example-label pairs to a multi-label class is a proposed adaptation, not a validated standard. |
| [Toneva et al., 2019](https://arxiv.org/abs/1812.05159) | Some training examples are repeatedly forgotten; forgetting dynamics contain information about difficulty and possible label problems. | Repeated loss of a learned positive can flag an unstable or ambiguous signal. | It diagnoses examples, not whether a whole label should be retained. |
| [Saito and Rehmsmeier, 2015](https://doi.org/10.1371/journal.pone.0118432) | ROC summaries can look optimistic under severe imbalance; precision-recall views make positive-prediction precision visible. | Report per-label precision-recall behaviour and positive support for rare labels. | It does not make one PR-derived number sufficient for all decisions. |
| [Lipton, Elkan, and Narayanaswamy, 2014](https://arxiv.org/abs/1402.1892) | The F1-optimal decision threshold depends on the score distribution and the evaluation batch; even an uninformative score can yield counter-intuitive F1 decisions. | Never use a per-epoch or per-label maximum F1 with an implicit or changing threshold as the main learnability evidence. | F1 remains meaningful when the output policy and threshold-selection procedure are fixed. |
| [Dembczyński et al., 2013](https://proceedings.mlr.press/v28/dembczynski13.html) | Multi-label F-measure optimisation is a structured prediction problem, not merely an independent score ranking. | Keep score quality and the later threshold/decision rule conceptually separate. | The paper studies F-measure optimisation, not visual learnability. |
| [Marsden et al., 2020](https://doras.dcu.ie/24426/1/ICME2020_CameraReady.pdf) | Multi-label class difficulty is associated with frequency, visual variation, semantic abstraction, and co-occurrence; frequency alone is insufficient. | Interpret per-label metrics together with support, prevalence, visual variation, semantic type, and co-occurrence. | The factors explain difficulty; they do not create a universal inclusion threshold. |
| [Ojala and Garriga, 2010](https://www.jmlr.org/papers/v11/ojala10a.html) | Permutation tests can test whether a classifier exploits structure associated with the labels. | A label-permutation control is a strong optional check that a claimed signal is not a pipeline artefact. | It is a statistical control, not proof of direct visual recognition. |
| [Hoiem et al., 2021](https://proceedings.mlr.press/v139/hoiem21a.html) | Learning curves reveal how performance changes with available training data. | For consequential borderline labels, measure held-out AP across training-set sizes to distinguish data-limited from persistently difficult labels. | This costs extra experiments and is not required for an initial screen. |

## Why maximum F1 is not a sufficient screen

For a label `l`, F1 is calculated only after scores have been converted to
positive or negative predictions. Its value consequently depends on a threshold
policy, score calibration, support, and the selected evaluation subset. Taking
the maximum across epochs adds a second selection effect: a noisy transient
peak is rewarded even if the training trajectory never stabilised. Comparing
such maxima across labels also obscures the very different uncertainty of a
label with ten positives and one with thousands.

This does not make the historical use of training F1 meaningless. With a fixed,
declared threshold, it can show that the optimizer begins to fit a label. It is
an **optimization signal only**. It does not demonstrate held-out
generalization, direct visual evidence, or a causal improvement after a
vocabulary change.

AP is preferable as the primary class-wise score because it evaluates the
ranking of positives against negatives without committing to one operating
threshold. However, AP must still be read together with positive count and
prevalence: the precision of an uninformative ranking, and the uncertainty of
the estimate, both vary markedly with class support. AP is not a magic
prevalence-invariant learnability score.

## Reusable assessment protocol

### 1. Declare the estimand before training

Record what “learnable” means for the study and fix the conditions under which
the conclusion holds:

- task and unit of observation (for example, image-level multi-label
  prediction);
- annotation source and known ambiguity/noise;
- fixed dataset split, architecture family, loss, augmentation, epoch/compute
  budget, and global model-selection policy;
- whether the desired claim is optimization, held-out predictability, direct
  visual observability, or usefulness in a downstream decision; and
- a threshold policy for any reported F1. Select that policy on training and
  validation data only; do not use a test set to decide it.

Without this declaration, “learnable” can silently mean a different thing for
each label or experiment.

### 2. Measure optimization as a trajectory

For every epoch `e`, seed `s`, and label `l`, log class-wise **training AP**.
Use robust summaries rather than one peak. With predeclared early and late
windows, useful descriptive quantities are:

\[
\Delta AP^{train}_{l,s} = \operatorname{median}_{e \in W_{late}} AP^{train}_{l,s,e}
- \operatorname{median}_{e \in W_{early}} AP^{train}_{l,s,e},
\]

\[
AP^{train,late}_{l,s} = \operatorname{median}_{e \in W_{late}} AP^{train}_{l,s,e}.
\]

The first asks whether the model improved during the available budget; the
second asks whether it reached a sustained level. Report both across seeds, not
only their best run. A fixed-threshold train F1 trajectory may be logged next to
them for continuity with a previous study, but it should not replace them.

For deeper diagnosis, store per-example-label score trajectories for positive
pairs. Mean confidence, temporal variability, and fixed-policy forgetting can
then identify labels dominated by unstable or repeatedly lost positives. This
is an explicitly labelled multi-label adaptation of training-dynamics research,
not a standard validated label-level metric.

### 3. Establish held-out generalization separately

Use per-label **validation AP** as the primary generalization measurement and
report its distribution across at least several independent seeds. Alongside
each score, retain:

- validation positive support and prevalence;
- a precision-recall curve or enough scores to regenerate it;
- the train-to-validation gap using comparable AP summaries; and
- fixed-policy validation F1, precision, and recall only when a concrete
  binary-output policy is relevant.

Choose architectures, epochs, and threshold rules globally, rather than
retuning them label by label. If many labels have small support, add uncertainty
intervals or resampling summaries and state when an estimate is too unstable to
rank confidently. The initial study should decide numerical promotion criteria
from a bounded pilot; this general research intentionally does not prescribe a
universal AP or F1 cut-off.

### 4. Check what the apparent signal means

For each label, keep metadata that could explain the metric:

- number of positives and prevalence;
- label co-occurrence and a non-visual prior baseline, where applicable;
- semantic type (directly depicted object, preparation state, dish/context, or
  otherwise non-observable concept);
- known visual variation and likely annotation ambiguity; and
- seed/configuration agreement.

Compare the image model with a simple prevalence or metadata/context baseline
when context is available. A large image-model advantage supports—but does not
prove—a visual signal. For high-stakes claims or suspicious results, run a
label-permutation null experiment and visually review examples. A label that is
predictable mainly from cuisine, recipe context, or co-occurring labels should
not automatically be described as directly recognisable in an image.

### 5. Produce a profile, not a forced binary label

The output can be a compact evidence table rather than a ranking quota:

| Profile | Interpretation | Appropriate next action |
| --- | --- | --- |
| No sustained train improvement | Not learned under the declared budget. | Inspect support/annotation quality; do not infer intrinsic impossibility. |
| Sustained train improvement, weak validation AP | Optimization-only or overfit signal. | Check support, split, regularisation, and label ambiguity. |
| Stable validation AP, strong contextual baseline | Predictable but likely contextual. | Keep distinct from direct visual-recognition claims; consider the downstream use. |
| Stable train and validation AP across seeds, image advantage, plausible visual type | Generalizable candidate. | Consider retention subject to human/semantic review. |
| Wide seed variation or low support | Uncertain. | Gather data, report uncertainty, or defer the decision. |

“Direct visual candidate” should be awarded only after the last row's evidence
is paired with an explicit human observability review. Model metrics alone do
not establish direct visibility.

## Minimum artefacts for reproducibility

For a study to be revisited or compared fairly, retain:

- immutable run configuration, split identifier, code revision, random seeds,
  and checkpoint/epoch selection rule;
- per-epoch per-label train and validation AP, plus the exact F1 threshold
  policy if F1 is reported;
- label support/prevalence for every split and the label vocabulary version;
- raw or regenerable scores sufficient to recreate per-label PR curves and
  uncertainty summaries;
- non-visual baseline outputs and any permutation-control definition; and
- the evidence table and human-review rationale used for the final action.

Do not compare raw training loss between runs with different weighting or loss
definitions as if it were a common learnability scale. Compare a common
evaluation metric under the same split and document any intentional change in
the experimental conditions.

## Anti-patterns to avoid

- Selecting labels by the single highest F1 reached at any epoch.
- Selecting a fixed top fraction or within-run percentile without an absolute
  stability/generalization check.
- Letting a threshold change per epoch or label, then interpreting the best F1
  as learning progress.
- Treating train performance as validation evidence.
- Ranking rare and common labels without displaying support and uncertainty.
- Calling a context-predictable label visually recognisable without an
  appropriate control and semantic review.
- Comparing results across changed splits, vocabularies, losses, or augmentation
  policies without labelling the comparison as non-causal.

## Suggested adoption path for a specific project

This research does not by itself alter a benchmark or ingredient vocabulary. A
project that wants to adopt it should first run a small, frozen pilot with a
fixed configuration panel and at least three seeds. The pilot should log
train/validation AP trajectories, support, and fixed-policy F1, then test
whether the proposed evidence profiles are stable enough to support explicit
criteria. Only after that pilot should the project freeze numerical selection
criteria and implement them as a reusable component.

### Resource-bounded single-run variant

When repeated training across seeds is infeasible, a study may deliberately use
one declared seed per configuration as a **screening variant**. It still needs
the fixed split, per-epoch train/validation AP, fixed-policy F1 diagnostic,
support/prevalence, non-visual controls, and provenance above. It may add
within-run early/late-window sensitivity, configuration sensitivity where a
bounded panel exists, and resampling intervals over held-out scores.

This variant cannot establish run-to-run or seed-level stability. A resampling
interval quantifies finite held-out-sample uncertainty, not variation caused by
training stochasticity. Its reports must preserve the declared seed, state the
limitation prominently, and classify borderline labels as uncertain rather than
claiming stable generalization. It is a resource-constrained adaptation of the
recommended protocol, not equivalent evidence.

For the Ingredient Recognition project, the general evidence profile has been
adopted as the planning framework in
[`recognizable_ingredient_selection.md`](../../../plans/recognizable_ingredient_selection.md).
Its execution is deferred until Macro-section 4 chooses the reference selector;
the binding cross-phase rationale and ownership boundary are in
[`model_comparison_methodology.md`](../../../project_objective/model_comparison_methodology.md).
The selection plan—not this research note—will record the frozen numerical
gates, run panel, single-run resource limit, and implementation decision after
the bounded pilot.

## Limitations and open questions

- AP and F1 quantify predictive behaviour, not truth of the annotation or
  human observability.
- Training-dynamics and forgetting literature is primarily example-level and
  often single-label; the label-level multi-label aggregation above requires
  empirical validation.
- Multiple labels and configurations create a multiple-comparison problem;
  apparent winners from a large exploratory screen can be optimistic.
- Held-out validation guides iterative choices but is no longer an unbiased
  final estimate after repeated reuse. Keep a test evaluation isolated when the
  project reaches a final benchmark stage.
- Learning curves and permutation controls offer stronger evidence but may be
  too expensive for every label. Apply them to important or ambiguous cases
  after the basic screen.

## References

1. Swayamdipta, S., Schwartz, R., Dey, K., and Wang, Y. (2020). *Dataset
   Cartography: Mapping and Diagnosing Datasets with Training Dynamics.* EMNLP.
   https://aclanthology.org/2020.emnlp-main.746/
2. Toneva, M. et al. (2019). *An Empirical Study of Example Forgetting during
   Deep Neural Network Learning.* ICLR. https://arxiv.org/abs/1812.05159
3. Saito, T., and Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More
   Informative than the ROC Plot When Evaluating Binary Classifiers on
   Imbalanced Datasets.* PLOS ONE. https://doi.org/10.1371/journal.pone.0118432
4. Lipton, Z. C., Elkan, C., and Narayanaswamy, B. (2014). *Thresholding
   Classifiers to Maximize F1 Score.* https://arxiv.org/abs/1402.1892
5. Dembczyński, K., Waegeman, W., Cheng, W., and Hüllermeier, E. (2013).
   *Optimizing the F-Measure in Multi-Label Classification: Plug-in Rule
   Approach versus Structured Loss Minimization.* ICML.
   https://proceedings.mlr.press/v28/dembczynski13.html
6. Marsden, M. et al. (2020). *Investigating Class-level Difficulty Factors in
   Multi-label Classification Problems.* ICME.
   https://doras.dcu.ie/24426/1/ICME2020_CameraReady.pdf
7. Ojala, M., and Garriga, G. C. (2010). *Permutation Tests for Studying
   Classifier Performance.* JMLR. https://www.jmlr.org/papers/v11/ojala10a.html
8. Hoiem, D. et al. (2021). *Learning Curves for Analysis of Deep Networks.*
   ICML. https://proceedings.mlr.press/v139/hoiem21a.html
