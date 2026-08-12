# Historical ingredient-selection logic reconstruction

**Created:** 2026-08-10

**Last updated:** 2026-08-10

## Status and scope

**Reconstruction status:** Reviewed and accepted on 2026-08-10 as the historical baseline and retention evidence.

**Evidence cutoff:** 2026-08-10.

**Covered period:** 2024-10-22 through 2024-11-17, with source files first committed on 2026-01-06.

This report reconstructs what was actually done to select a 40-label legacy ingredient vocabulary, which experiments and data generations were involved, and which conclusions are or are not supported by the saved artifacts. It does not approve the historical rule for the new benchmark, reimplement the analysis, or clean old scripts. Its reviewed conclusions have been promoted to the Data 2.1c retention decision and the Macro-section 3 feature plan; those durable documents now own project status and future methodology.

The old document calls its internal steps “Fase 1” through “Fase 4”. Those names are historical experiment stages and must not be confused with the current project Macro-section 3, **Ingredient selection**.

## Executive reconstruction

The historical process was a four-stage exploratory loop:

1. Tune a mixed model and optimizer search on the full 183-output legacy vocabulary, using half of the training batches per epoch and validation loss as the Optuna objective.
2. Retrain the best three configurations with per-label F1 logging, add an intended weighted-loss control, rank labels by their maximum **training** F1 over 40 epochs, keep the top quartile within each run, and intersect the resulting sets.
3. Project the legacy metadata onto the 40-label intersection, remove recipes with fewer than three retained labels, and tune again on one quarter of the selected training batches.
4. Validate the three best selected-vocabulary checkpoints and compare their validation F1 with an old statistic.

The 40-label result is exactly reconstructible. It came from the intersection of four 46-label top-quartile sets. A fifth set based on the across-run mean was displayed but was not included by the code; including it would not have changed the historical result.

The result should be retained as a historical baseline, not adopted unchanged. The rule measures relative rank by maximum train F1, not whether a label begins to improve or exceeds an absolute learnability criterion. Several surrounding interpretations are invalidated by saved configuration evidence.

## Evidence and confidence

### Primary local evidence

| Evidence | Role | Confidence |
| --- | --- | --- |
| External `Analisi singoli ingredienti.docx`, last modified 2024-11-17 | Human explanation, figures, questions, and claimed conclusions | High for intent; lower when contradicted by saved configs |
| External `09-11-2024/plots/` CSV, TXT, and PNG files | Exported tables, selected sets, and plotted results | High for reported outputs |
| [`../../experiments/basic/resnets_htuning{25k}/`](../../experiments/basic/resnets_htuning%7B25k%7D/) | Initial study configs, per-trial metrics/checkpoints, and summaries | Highest for executed H1 settings |
| [`../../experiments/basic/resnets_training_BM_F1_INGS/`](../../experiments/basic/resnets_training_BM_F1_INGS/) | Four per-label-F1 runs and the exact selection inputs | Highest for executed H2 behavior |
| [`../../experiments/basic/resnets_htuning_sel{10k}/`](../../experiments/basic/resnets_htuning_sel%7B10k%7D/) | Selected-vocabulary tuning study | Highest for executed H3 settings |
| [`../../experiments/basic/resnets_htuning_sel{10k}_test/`](../../experiments/basic/resnets_htuning_sel%7B10k%7D_test/) | Three validation outputs and comparison inputs | Highest for executed H4 evaluation |
| [`../../scripts/analize_exps/analize_f1_ings.ipynb`](../../scripts/analize_exps/analize_f1_ings.ipynb) | Selection and metadata-projection code | High for algorithm; not top-to-bottom reproducible now |
| [`../../scripts/analize_exps/analize_test_f1.ipynb`](../../scripts/analize_exps/analize_test_f1.ipynb) | Selected-checkpoint analysis and old/new plot | High for calculation logic |
| [`../../scripts/analize_exps/refine_study_data.ipynb`](../../scripts/analize_exps/refine_study_data.ipynb) | Optuna/local-metric merge and best-trial tables | High for transformations; contains manual mode-specific cells |
| [`../../scripts/analize_exps/visualize_study_data.ipynb`](../../scripts/analize_exps/visualize_study_data.ipynb) | Hyperparameter visualizations | High for intended plotting path |
| [`../../scripts/analize_exps/fake_study.ipynb`](../../scripts/analize_exps/fake_study.ipynb) | Rebuilt corrupted initial Optuna study | High for reconstruction history; contains the augmentation relabeling defect |
| [`../../experiments/journal.log`](../../experiments/journal.log) and `journal_trash.log` | Historical Optuna journal and repaired copy | High, but not self-contained per experiment |

### Git limitation

The last commit before the experiments was `846d22ac7158add84c2109c4e80506c8e619e4a1` on 2024-10-11. The selection notebooks, launchers, F1 metric support, and associated code were not committed until `b86f5a3caaff626c567f63517e9f8f35b3c40445` on 2026-01-06. Git therefore cannot recover an exact 2024 source snapshot. The saved 2024 configurations, YAML files, metrics, journals, metadata, and exported plots are the authoritative forensic evidence for executed behavior.

Commit `89ec11ef7c1b8eae4792ef8ebac62834aa20dd7e` only moved the launchers into model-specific subdirectories on 2026-03-26; it did not repair their old imports or hard-coded workflow.

## Historical stage inventory

| ID | Historical document stage | Experiment group | Executed scope | Key output |
| --- | --- | --- | --- | --- |
| H1 | Fase 1 — tuning | `resnets_htuning{25k}` | 29 trial directories (`0`–`28`); summary retains 28 non-running trials: 8 complete, 19 pruned, 1 error | Best-trial table and initial Optuna plots |
| H2 | Fase 2 — per-label F1 and selection | `resnets_training_BM_F1_INGS` | Four 40-epoch runs on legacy `metadata.json`, 183 outputs, per-label train/validation F1 logged | `full_f1_train.csv`, `full_labels_stats.csv`, 40 selected labels |
| H3 | Fase 3 — tuning selected labels | `resnets_htuning_sel{10k}` | 73 trials (`0`–`72`): 16 complete and 57 pruned; 41 outputs (40 labels plus `<UNK>`) | Best selected-vocabulary trials |
| H4 | Fase 4 — testing selected labels | `resnets_htuning_sel{10k}_test` | Validation-only execution of selected-study trials 64, 66, and 68; no retraining in this group | Per-label validation F1 and old/new plot |

`densenets_training` is not part of the documented selection process.

### Approximate chronology

- H1 study records begin on 2024-10-22; the current root summaries were regenerated on 2024-11-05 and 2024-11-09.
- H2 checkpoints and metrics were produced on 2024-10-23 and 2024-10-24; the aggregate selection files were regenerated on 2024-11-09.
- H3 began on 2024-10-24 and continued through 2024-11-05.
- H4 validation metric files originate on 2024-10-25; aggregate tables were regenerated through 2024-11-14.
- The current `sel_ing_2410_metadata.json` files were written on 2024-11-09, after H3 and the original H4 validation runs had begun or completed. No training-time metadata hash was saved, so byte-for-byte identity with the exact files used by those runs cannot be proven.
- The Word document was last modified on 2024-11-17 despite living in the `09-11-2024` folder.

## H1 — initial hyperparameter tuning

### Dataset and objective

- Metadata: legacy `metadata.json` with target field `ingredients_ok`.
- Output space: 183 outputs: 182 legacy ingredient labels plus `<UNK>`.
- Batch size: 128.
- Epoch budget: 40.
- Training fraction per epoch: `limit_train_batches=0.5`, roughly 27,000 examples from the 54,724-record train split; “25k” is an approximate experiment name.
- Objective: minimize raw validation `BCEWithLogitsLoss`.
- Sampler: multivariate/grouped TPE, 7 startup trials, 24 EI candidates.
- Pruner: Hyperband, `min_resource=2`, automatic maximum resource, reduction factor 3.
- Configured trial budget: 100; execution stopped with trial directory 28, while the cleaned summary contains trials 0–27.

### Search space

| Parameter | Historical search |
| --- | --- |
| Learning rate | Log-uniform `1e-5` to `1` |
| Scheduler | None; `CosineAnnealingWarmRestarts`; custom `ConstantStartReduceOnPlateau` |
| Optimizer | Adam; SGD with momentum 0.9 |
| Weight decay | Log-uniform `1e-5` to `1e-1` |
| Weighted loss | `True` or `False` |
| Model | `DummyBNModel`; custom unpretrained `ResnetLikeV1`; pretrained torchvision `Resnet18` |
| Augmentation | None; `transform_aug_adv` |

The captured augmentation applies random resized crop with scale `(0.33, 1.0)` and ratio `(0.5, 2.0)`, horizontal and vertical flips, and rotation. The saved function points to the repository implementation captured in 2026; Git does not prove that every probability is byte-identical to the uncommitted runtime source used in 2024.

### Three configurations forwarded to H2

The three lowest-loss complete trials were 21, 2, and 18. All three were the custom `ResnetLikeV1`, not pretrained ResNet18:

| Source trial | LR | Scheduler | Optimizer | Weight decay | Weighted | Actual saved augmentation |
| --- | ---: | --- | --- | ---: | --- | --- |
| 21 | 0.657274 | cosine warm restarts | SGD | 1.4359e-5 | No | No |
| 2 | 0.148173 | none | SGD | 1.3198e-5 | No | Yes |
| 18 | 0.055370 | cosine warm restarts | SGD | 2.6741e-5 | No | Yes |

The `image_augmentation` values in `best_trials.csv` state the opposite because of the analysis defect documented below.

## H2 — F1 logging and label selection

### Executed runs

The launcher loaded the three H1 trial configurations, switched from `OptunaTrainer` to `BaseTrainer`, added per-label F1, and requested `limit_train_batches=1.0`. The saved aggregate has four runs × 40 epochs × 183 per-label F1 columns.

The document describes these runs as using 10k images, but the launcher requests the full 54,724-record training split. The checkpoints prove full-train execution: completed runs record 17,120 global steps over 40 epochs, exactly 428 batches per epoch; the run whose best checkpoint is epoch 37 records 16,264 steps, also 428 per completed epoch. This is incompatible with a 10k-image epoch.

The run mapping is:

| H2 run | H1 source | LR | Intended role | Saved `weighted_loss` |
| --- | --- | ---: | --- | --- |
| 0 | 21 | 0.657274 | Best configuration | `false` |
| 1 | 2 | 0.148173 | Second configuration | `false` |
| 2 | 18 | 0.055370 | Third configuration | `false` |
| 3 | 21 | 0.657274 | Intended weighted-loss copy | `false` |

Run 3 is not a weighted-loss experiment. The launcher changes a pandas row used for notes, but never applies `hp_weighted_loss=True` to the loaded `ExpConfig`. Both `trial_0/hparams.yaml` and `trial_3/hparams.yaml` confirm `weighted_loss: false`. Run 3 is effectively another stochastic execution of trial 21.

### F1 definition and source split

The injected metric is TorchMetrics `F1Score(task="multilabel", average="none")`; the historical configuration does not specify a custom threshold, so it uses the library default behavior. The model passes logits directly to TorchMetrics.

The selection notebook explicitly sets `use_val_data = False`. Therefore all selection statistics come from columns named `train_f1_label_<index>`, not validation F1. Validation F1 was logged but not used to choose labels.

### Exact selection algorithm

For run `r`, label `l`, and epoch `e`:

```text
M[r,l] = max over epochs e of train_F1[r,l,e]
T[r]   = 75th percentile across all 183 M[r,l] values
S[r]   = labels l where M[r,l] >= T[r]
```

The notebook also computes:

```text
A[l]   = mean across runs r of M[r,l]
T_avg  = 75th percentile across labels of A[l]
S_avg  = labels l where A[l] >= T_avg
```

The implemented final set is:

```text
S_final = S[0] intersection S[1] intersection S[2] intersection S[3]
```

The Word document says the intersection uses five sets, including `S_avg`. The code does not include `S_avg`; however, `S_final` is a subset of `S_avg`, so both definitions produce the same historical 40 labels.

### Reconstructed thresholds

| Set | Q3 threshold on per-label max train F1 | Selected labels |
| --- | ---: | ---: |
| Run 0 | 0.087803 | 46 |
| Run 1 | 0.082368 | 46 |
| Run 2 | 0.022505 | 46 |
| Run 3 | 0.103269 | 46 |
| Across-run mean maxima | 0.071998 | 46 |
| Four-run intersection | — | 40 |

This is a relative top-quartile rule. It always retains approximately one quarter of the output space even if no label meets a meaningful absolute learnability threshold. It does not calculate an improvement slope, delta from an initial epoch, confidence interval, or null-model excess.

### Selected legacy vocabulary

The following table reports the historical across-run mean of each selected label's maximum train F1. It is descriptive; it was not the final acceptance threshold.

| Label | Mean maximum train F1 |
| --- | ---: |
| `avocado` | 0.306928 |
| `basil` | 0.088427 |
| `beans` | 0.225610 |
| `beef` | 0.136035 |
| `bread` | 0.256835 |
| `butter` | 0.287280 |
| `cabbage` | 0.146422 |
| `carrot` | 0.258790 |
| `cheese` | 0.571208 |
| `chicken` | 0.308366 |
| `chili` | 0.178162 |
| `chocolate` | 0.313458 |
| `cilantro` | 0.255068 |
| `cinnamon` | 0.131187 |
| `coriander` | 0.241723 |
| `corn` | 0.195191 |
| `cucumber` | 0.246520 |
| `cumin` | 0.292390 |
| `egg` | 0.434272 |
| `flour` | 0.296859 |
| `garam masala` | 0.200029 |
| `garlic` | 0.652868 |
| `lettuce` | 0.118065 |
| `liquor` | 0.314864 |
| `milk` | 0.154715 |
| `oil` | 0.739165 |
| `onion` | 0.643076 |
| `parsley` | 0.151802 |
| `pasta` | 0.184743 |
| `pepper` | 0.668684 |
| `rice` | 0.213431 |
| `salt` | 0.768832 |
| `shrimp` | 0.130084 |
| `soy` | 0.491822 |
| `strawberries` | 0.106617 |
| `sugar` | 0.484211 |
| `tomato` | 0.578186 |
| `turmeric` | 0.431963 |
| `vanilla` | 0.380194 |
| `yogurt` | 0.154489 |

The mean of the maxima for these 40 labels is 0.31846. Across all 182 real legacy labels it is 0.07972, with a median of 0.00309. These values show a strongly separated tail, but they do not distinguish visual learning from prevalence, cuisine priors, co-occurrence, or transient metric noise.

## Selected metadata generation

The notebook projects each record's legacy `ingredients_ok` onto the 40-label vocabulary:

```python
list(set(record_ingredients).intersection(selected_ingredients))
```

It then retains a record only when at least three selected labels remain and writes `sel_ing_2410_metadata.json` to every legacy split.

| Split | Original records | Selected records | Current file SHA-256 |
| --- | ---: | ---: | --- |
| Train | 54,724 | 50,866 | `69189FD97ADB0AAAB6748874F44CD50DAA6FC7D1A739DFE9040EC28018E5623B` |
| Validation | 5,210 | 4,802 | `382D24E2C99DE5C93EEC6EF7E29F00935BB5166C42C98EC37458DA9E3474B102` |
| Test | 5,212 | 4,854 | `0E37DFE11219CD4DCF21185E4FA2AA8D6F218BCF10DC1B8E7EE72D1F4F163FF3` |

The selected train records contain a mean of 6.79 labels instead of 9.13, a median of 6 instead of 9, and a maximum of 22 instead of 42.

The `set` conversion makes target order process-dependent. Multi-label semantics are set-based, but the JSON generation is not byte-deterministic under this implementation. The current files are legacy artifacts and must remain unchanged.

## H3 — tuning on the selected vocabulary

H3 repeats the H1 search space on `sel_ing_2410_metadata.json`:

- 41 outputs: the 40 selected labels plus `<UNK>`;
- 40 epochs;
- batch size 128;
- `limit_train_batches=0.25`, roughly 12,700 train records per epoch; “10k” is approximate;
- TPE sampler with the same settings;
- Hyperband with `min_resource=5` instead of 2;
- raw validation loss objective;
- 73 attempted trials, of which 16 completed and 57 were pruned.

The three configurations forwarded to H4 were:

| Source trial | Model | LR | Scheduler | Optimizer | Weight decay | Weighted | Actual saved augmentation |
| --- | --- | ---: | --- | --- | ---: | --- | --- |
| 64 | Pretrained ResNet18 | 0.000292 | custom plateau | Adam | 0.000297 | No | Yes |
| 66 | Pretrained ResNet18 | 0.000126 | custom plateau | Adam | 0.000821 | No | Yes |
| 68 | Pretrained ResNet18 | 0.000140 | custom plateau | Adam | 0.000257 | No | No |

Again, the exported `image_augmentation` column states the opposite of the saved configurations.

## H4 — selected-checkpoint validation and comparison

[`../../scripts/launch_exps/test_best_for_f1.py`](../../scripts/launch_exps/test_best_for_f1.py) loads H3 trials 64, 66, and 68 and calls `trainer.validate(...)` on their best checkpoints. Despite “test” in the filename and document heading, this is validation, not test-split evaluation. H4 does not train new models.

The three validation runs produce 40 real-label F1 values plus a zero-valued `<UNK>` output. The reported real-label F1 values are broadly consistent across the three selected configurations.

The final “old versus new” plot is not a like-for-like comparison:

- “new” is the mean across three H4 **validation endpoint** F1 values;
- “old” is the mean across three H2 runs of the **mean training F1 over all 40 epochs**;
- the label spaces and record cohorts differ;
- the model selection and training fractions differ.

The plot can illustrate the historical observation, but it cannot establish that the new vocabulary caused the improvement.

## Verified discrepancies and reproducibility defects

### Defects that change interpretation

1. **Selection used train F1.** The document does not make this explicit; the notebook does.
2. **The rule does not measure “starting to improve”.** It selects by maximum F1 rank, regardless of trajectory or absolute level.
3. **The weighted-loss control was not activated.** H2 run 3 has `weighted_loss: false`.
4. **Weighted and unweighted validation losses are not directly comparable.** Weighted mode instantiates `BCEWithLogitsLoss(pos_weight=classes_weights)`, changing the objective's numerical scale. The conclusion that weighting is counterproductive, and its approximately 0.49 fANOVA importance, are dominated by an objective-scale confound.
5. **Augmentation labels are inverted.** `refine_study_data.ipynb` maps `"no_aug"` to `True` and `"aug_hard"` to `False`. The exported tables and some plots therefore reverse the actual saved setting. Importance magnitude is invariant to swapping binary names, but directional interpretation is wrong.
6. **The old/new F1 plot compares different statistics and cohorts.** It is not causal evidence for selection quality.
7. **No seed protocol or uncertainty exists.** The four H2 runs are three configurations plus one accidental unweighted repeat, not controlled repeated seeds.
8. **No prevalence, cuisine-prior, random-40, or non-visual baseline was run.** The Word document lists the random-40 check as an open question; no matching artifact was found.

### Defects that block direct rerun

1. `analize_f1_ings.ipynb` currently loops over three runs in its source, while its saved outputs and later cells require four. Running it top to bottom no longer reproduces the 40-label result without editing state.
2. `refine_study_data.ipynb` contains mutually exclusive H1/H3 repair cells, manual trial patches, and stale state. It was operated interactively rather than as a deterministic pipeline.
3. H1's original Optuna journal was considered corrupted and rebuilt through `fake_study.ipynb` into a shared trash journal.
4. The current `best_trials.csv` for H1 has eight rows, but H2 contains only the historical top three plus the intended control. The launcher reads a mutable file and does not pin the chosen trial list.
5. The launchers retain hard-coded experiment names and old imports such as `from config` and `from training...`; the current repository only exposes `settings.config` and `src.training...`.
6. Search lambdas are serialized only as function descriptions. The currently committed launchers preserve the ranges, but the 2024 executable source was uncommitted.
7. Experiment configurations store paths and encoders but not code commit, environment lock, metadata hash, split hash, or seed.
8. The H3/H4 metadata files were later overwritten and are generated through unordered sets, preventing byte-level proof of the exact input used by the original runs.

## What can be trusted

### Verified historical facts

- The 40 names, their legacy indices, the five 46-label candidate sets, and the four-run intersection.
- The train/validation/test record counts of the selected legacy metadata.
- The H1 and H3 trial configurations saved in JSON and YAML.
- The per-epoch H2 F1 values and H4 validation F1 values saved in CSV.
- The fact that H2 run 3 was unweighted and that augmentation flags in the summary CSVs are inverted.
- The exact current bytes of the legacy selected metadata, which must not be rewritten.

### Historical interpretations that are not verified

- That the selected labels are visually recognizable rather than frequent or context-predictable.
- That weighted loss hurts predictive performance.
- That augmentation was or was not beneficial in the direction shown by the tables.
- That label-space reduction itself caused the plotted F1 increase.
- That Q3 and four/five-set intersection are optimal selection rules.
- That the result generalizes from legacy `ingredients_ok` to the current FoodOn-first `ingredients_target_v5_metadata.json` vocabulary.

## Artifact retention implications for Data Work package 2.1c

No deletion should occur during reconstruction. The minimum historical evidence set to consider for retention is:

1. H1 root configs/summaries and source trials 21, 2, and 18; retain `trials_info.csv` for the full study context.
2. All four H2 run directories plus `full_f1_train.csv`, `full_labels_stats.csv`, and `full_metrics.csv`; these are the direct selection evidence.
3. All three `sel_ing_2410_metadata.json` files byte-for-byte.
4. H3 root configs/summaries and source trials 64, 66, and 68; retain `trials_info.csv` for study context.
5. The complete H4 directory, which is small and contains the final validation evidence.
6. The five analysis notebooks, four relevant launchers, Optuna journals, external Word document, and external plot exports.

Whether non-selected heavy checkpoints from H1 and H3 can be retired is a later 2.1c decision. Root metrics and trial configurations may be enough for study-level analysis, but that must be validated before deleting any checkpoint.

## Implications for the future Macro-section 3 plan

The future implementation should separate two named tracks:

1. **Historical reproduction:** encode the rule above exactly against retained legacy artifacts and prove it regenerates the 40-label list without writing legacy data.
2. **New recognizability selection:** define a benchmark-valid protocol for the frozen `v5` vocabulary, with train F1 used explicitly for optimization learnability, validation-only model and hyperparameter selection, fixed seeds and budgets, comparable ResNet configurations, absolute and stability criteria, prevalence/non-visual controls, and no test access.

At minimum, the new method should distinguish:

- semantic relevance and sufficient support;
- direct visual observability versus contextual prediction;
- absolute per-label performance;
- improvement over a prevalence or non-visual baseline;
- stability across seeds/configurations and epochs;
- uncertainty around the selection decision.

The historical Q3/intersection rule is useful as a baseline or ablation. It should not be the sole acceptance rule because it guarantees a relative quota and is sensitive to transient maxima.

## Planning checkpoint resolution

The 2026-08-10 review resolved the open questions as follows:

1. Historical reproduction must regenerate the exact 40-label result from retained aggregates, and Data 2.1c must also smoke-load a bounded set of full-vocabulary and selected-vocabulary checkpoint anchors.
2. Train F1 was intentional as an optimization-learnability signal. It is not validation generalization or proof of literal visual recognizability.
3. The new method will preserve separate evidence for optimization learnability, validation generalization, semantic relevance, and the `direct`/`contextual`/`not_inferable`/`uncertain` observability states.
4. The 165-label `v5` vocabulary remains the shared default. Any smaller set is a named headline or exploratory projection, never a second implicit default.
5. The precise minimum artifact set and executable anchors are frozen in Data Work package 2.1c. No deletion is authorized until its manifest, reproduction, and compatibility gates pass.
6. The weighted-loss mistake is nonessential and will not be reproduced; the augmentation inversion is a reporting defect; and the old/new plot is retired as comparative evidence.

## Durable owners and next checkpoint

[`../../docs/plans/data_ingredient_refactor/yummly_data_phase.md`](../../docs/plans/data_ingredient_refactor/yummly_data_phase.md) owns retention and compatibility. [`../../docs/plans/recognizable_ingredient_selection.md`](../../docs/plans/recognizable_ingredient_selection.md) owns the accepted discrepancy resolutions, improved `v5` selection design, implementation sequence, plots, controls, and cleanup gates. The next planning checkpoint is to freeze that plan's experimental contract before implementation or new training begins.
