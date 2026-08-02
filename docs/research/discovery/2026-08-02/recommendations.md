# Recommended Research Program

Created: 2026-08-02  
Updated: 2026-08-02

## 1. Decision

Do not select a final architecture from literature rankings. Build a controlled local evidence chain in which target quality and leakage are fixed first, then test a bounded set of representation, head, loss, and preprocessing hypotheses.

The recommended program is designed for one RTX 4060 with 8 GB VRAM. It favors experiments that distinguish scientific explanations over broad hyperparameter sweeps.

## 2. Dependency gates

### Gate A — benchmark construction

Required before any result can be used for model selection:

- regenerated boundary-aware ingredient targets and full mapping trace;
- versioned ontology, aliases, exclusions, and unresolved-line report;
- invalid-image exclusion manifest;
- recipe-family evidence graph and frozen family assignments;
- deterministic group-disjoint 80/10/10 split;
- headline and exploratory vocabularies with support tables;
- frozen metric, threshold, seed, and bootstrap specification.

Legacy-target runs may be used only as pipeline smoke tests and must be labeled non-benchmark.

### Gate B — data-path validation

Required before training the shortlist:

- one authoritative transform per checkpoint;
- tests proving normalization occurs once;
- aspect-preserving evaluation path;
- image/label/family audit contact sheets;
- tiny-subset overfit test;
- deterministic dataloader and seed behavior documented.

### Gate C — compute feasibility

Required per backbone:

- checkpoint and terms recorded;
- repeatable offline/local load after acquisition;
- forward/backward smoke test;
- peak VRAM and throughput measured;
- smallest useful batch/effective batch recorded;
- frozen pilot completed before adaptation.

## 3. Falsifiable hypotheses

| ID | Hypothesis | Controlled comparison | Evidence that rejects it |
|---|---|---|---|
| H1 | Aspect-preserving preprocessing preserves ingredient cues better than square warp. | Same model/loss; pad or bounded crop vs legacy warp. | No repeatable gain in primary metrics or direct/low-resolution slices. |
| H2 | Self-supervised foundation features transfer better than a supervised CNN at this data scale. | DINOv2 frozen/partial vs convolutional baseline using the same head and input policy. | Difference is within seed/bootstrap uncertainty or loses materially on calibration/cost. |
| H3 | Vision-language pretraining adds useful ingredient semantics. | SigLIP 2 image encoder vs DINOv2 and CNN, first with the same learned head. | Gains occur only with text prompts, only in contextual labels, or fail resource/robustness gates. |
| H4 | Class-specific spatial queries help multi-component dishes. | ML-Decoder vs pooled independent head on the same best affordable backbone. | No macro-AP/direct-slice gain or attribution/robustness worsens. |
| H5 | Explicit label dependencies add beyond prevalence and cuisine priors. | One graph/label-state head vs independent head plus image-free controls. | Gains disappear on direct labels, shuffled-image control, or family-disjoint test. |
| H6 | Imbalance-aware loss improves tail ranking. | BCE vs distribution-balanced loss vs ASL on one frozen architecture. | Gain is unstable, head-only, or offset by unacceptable calibration/common-label loss. |
| H7 | A strictly proper asymmetric objective offers a better ranking/calibration tradeoff. | Selected calibrated objective vs BCE/ASL with identical tuning budget. | No Pareto improvement in ranking and calibration. |
| H8 | Conservative food-safe augmentation improves generalization. | No augmentation vs conservative policy on one fixed model. | No repeatable gain or degradation on direct/small-evidence labels. |

Each hypothesis has one primary comparison. Secondary metrics explain the outcome but should not be mined until something appears favorable.

## 4. Bounded model shortlist

### Tier 0 — non-visual controls

1. all-negative;
2. global training prevalence;
3. validation-selected global top-\(k\)/threshold diagnostic;
4. cuisine-only or metadata-only classifier where fields are available.

These establish how much of the target can be recovered without image evidence.

### Tier 1 — mandatory visual baselines

1. **Supervised convolutional baseline:** pretrained ResNet-50 for continuity or a compact ConvNeXt V2 if its implementation is equally stable.
2. **DINOv2 ViT-B/14-reg:** frozen linear head, then late-block/PEFT or full adaptation as feasible.

The existing DenseNet/ResNet implementations can support continuity checks, but they do not all need full tuning.

### Tier 2 — high-value representation candidates

1. **SigLIP 2 feasible base-scale image encoder:** tests vision-language transfer; start without text-defined decisions.
2. **DINOv3 smallest useful ViT or ConvNeXt variant:** tests newer dense self-supervision if access, dependencies, and memory pass Gate C.

If either candidate cannot meet the gate, record the reason and continue; do not shrink batches/resolution so far that the comparison becomes scientifically different.

### Tier 3 — structured heads on one promoted backbone

1. **ML-Decoder:** first spatial class-query test.
2. **One dependency model:** ML-GCN-style training-only graph or a C-Tran-style label-state head, selected after focused implementation research.

Do not run both heads across every backbone. Select the best affordable representation from Tiers 1–2 and hold it fixed.

## 5. Objective shortlist

Run on one fixed backbone/head and identical sampled batches:

1. sigmoid BCE;
2. distribution-balanced loss;
3. asymmetric loss;
4. a strictly proper asymmetric/calibrated multi-label objective if a verified implementation is available.

Use a small predetermined parameter grid from validation data. Report probability metrics before and after the same allowed post-hoc calibration procedure. Avoid combining loss, resampling, class weights, and logit adjustment in the first study because their effects become uninterpretable.

## 6. Data and augmentation shortlist

### Geometry study

- aspect-preserving resize + pad;
- aspect-preserving resize + bounded training crop/deterministic evaluation crop;
- legacy square warp as diagnostic.

### Augmentation study

- deterministic/minimal preprocessing;
- conservative food-safe policy;
- one filtered TrivialAugment or RandAugment policy;
- optional AugMix robustness study.

Defer MixUp, CutMix, SpliceMix, and generated training images until error analysis demonstrates a remaining problem they plausibly solve.

## 7. Staged experiment sequence

### Stage 1 — benchmark and pipeline controls

Outputs:

- frozen benchmark artifacts;
- non-visual baseline table;
- transform tests and contact sheets;
- one cheap convolutional end-to-end run;
- reproducibility rerun with identical config.

Stop if labels, family leakage, or metrics cannot be reproduced exactly.

### Stage 2 — preprocessing and baseline representation

Use the convolutional baseline to choose the geometry and conservative augmentation policy. Then compare the supervised convolutional model with frozen DINOv2 using the same independent head.

Promote a preprocessing policy only if it is stable across seeds or gives a clearly favorable cost/quality tradeoff on the prespecified slices.

### Stage 3 — adaptation and contemporary representations

Compare DINOv2 frozen, late-block/PEFT, and full fine-tuning where feasible. Run frozen-feature pilots for SigLIP 2 and DINOv3. Promote at most two representations to multi-seed adaptation based on:

- paired primary metrics;
- direct/contextual performance balance;
- calibration;
- VRAM, throughput, and wall-clock;
- implementation stability.

### Stage 4 — head and dependency tests

On one promoted representation:

1. pooled independent head;
2. ML-Decoder;
3. one explicit dependency head if image-free controls are ready.

Use identical features and training budgets where architecture permits. Analyze class-query attention only as supporting evidence.

### Stage 5 — loss and calibration

Freeze representation and head. Compare the objective shortlist and validation-only calibration. Select a model on a predeclared multi-objective rule, not micro-F1 alone.

One defensible rule is: maximize validation macro AP subject to no practically material regression in micro-F1, calibration, and direct-observability performance, with resource use reported as a constraint.

### Stage 6 — confirmatory evaluation

Lock code, configuration, calibration, and threshold. Run the approved seeds once on the full test set. Compute paired family bootstrap intervals and the frozen slice/error reports. Any benchmark correction after test inspection produces a new benchmark version and a new confirmatory cycle.

## 8. Minimal experiment matrix

The following is deliberately much smaller than the full cross-product:

| Experiment family | Approximate variants | Seeds before promotion | Purpose |
|---|---:|---:|---|
| Non-visual controls | 3–4 | deterministic | prior/shortcut floor |
| Geometry | 3 on one CNN | 2 pilot, 3 final | H1 |
| Conservative augmentation | 2–3 on one CNN | 2 pilot, 3 final | H8 |
| Representation pilots | CNN, DINOv2, SigLIP 2, conditional DINOv3 | 1 frozen pilot, 3 promoted | H2–H3 |
| DINOv2 adaptation | frozen, partial/PEFT, full if feasible | 2 pilot, 3 promoted | adaptation/value |
| Head study | pooled, ML-Decoder | 3 | H4 |
| Dependency study | independent, one structured, image-free control | 3 | H5 |
| Loss study | 3–4 | 2 pilot, 3 promoted | H6–H7 |
| Final confirmation | at most 2 models | at least 3 | thesis result |

Pilot runs use validation only and cannot be reported as final evidence. Early stopping rules and the exact promotion margin should be frozen before the pilot table is examined.

## 9. Promotion and stopping criteria

A candidate advances only when:

- the gain is visible across individual seeds, not one outlier;
- paired family-bootstrap uncertainty does not suggest a purely negligible difference;
- neither primary metric suffers an unaccepted regression;
- direct-observability and tail analyses support the stated mechanism;
- calibration can be made acceptable using validation only;
- compute and integration cost fit the thesis schedule;
- the result is reproducible from recorded artifacts.

Stop a branch when:

- it fails the same prespecified criterion twice after one justified correction;
- improvements occur only on legacy/leaked splits;
- dependency gains are matched by image-free controls;
- a larger model offers negligible benefit per unit cost;
- an augmentation or loss improves one headline number but consistently damages calibration or direct evidence;
- implementation instability prevents an auditable comparison.

## 10. Focused research topics to open next

The broad discovery narrows the useful follow-up work to five topic packages:

1. **Ingredient ontology and parser acceptance protocol**  
   Compare deterministic parsing, domain NER, review queues, ambiguity states, and external crosswalks; deliver fixtures and acceptance metrics.

2. **Family graph and constrained group allocation**  
   Define evidence thresholds from reviewed pairs, family construction, multi-label group allocation, and leakage audit reports.

3. **Food-safe preprocessing and augmentation**  
   Specify aspect-preserving transforms, contact-sheet review, operation allow/deny list, and the small H1/H8 ablation.

4. **Representation and class-query implementation shortlist**  
   Verify DINOv2 fixes, DINOv3/SigLIP 2 access and memory, ML-Decoder integration, checkpoint provenance, and parameter-efficient alternatives.

5. **Multi-label calibration and decision protocol**  
   Freeze AP/F1 implementations, objective/calibrator shortlist, reliability reporting, group bootstrap, and optional conformal prerequisites.

Topics 1–2 are benchmark dependencies. Topic 3 can be specified in parallel but evaluated only after the split is frozen. Topics 4–5 convert this discovery into the final model shortlist and metric implementation.

## 11. Research claims the thesis can support

If executed as above, the project can make evidence-backed claims about:

- the effect of correcting ontology and family leakage on ingredient inference;
- the tradeoff between visual self-supervision, vision-language pretraining, and supervised convolutional transfer;
- whether class-specific spatial access improves directly observable ingredients;
- whether label-dependency gains survive image-free and cuisine-prior controls;
- how imbalance-oriented objectives trade ranking quality against calibration;
- how aspect and augmentation choices affect low-resolution food images;
- the gap between recipe-level correctness and visual inferability.

Those claims are stronger and more reproducible than declaring a nominal architecture “state of the art” on an incomparable legacy split.

