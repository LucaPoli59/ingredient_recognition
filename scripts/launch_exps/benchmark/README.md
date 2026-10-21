# Phase 6 benchmark launchers

**Created:** 2026-10-18
**Last updated:** 2026-10-21

Run from the main WSL repository with the project's ML Python. The [scientific protocol](../../../docs/project_objective/model_comparison_methodology.md#phase-6-d1--finite-training-and-comparison-protocol), [runtime contract](../../../docs/implementation_details/benchmark_training.md) and [execution plan](../../../docs/plans/training_hyperparameter_tuning.md) own policy, implementation and state.

## Current implementation boundary — D3 fresh studies

[Phase6-D3](../../../docs/project_objective/model_comparison_methodology.md#phase-6-d3--fresh-fifty-trial-studies) is implemented under R1: commands use `phase6-comparison-v2`,50 attempts and D2's independent plateau guard. R2/R3 are complete; the original R3 preparation stays empty and the separately source-bound R4 continuous campaign is now running under explicit user approval. Default inspection still starts nothing. New execution uses fresh studies under `experiments/benchmark_v2/`, without importing or resuming the first release. Invocation examples alone are not permission to train.

Preserve `phase62-20261018-v1`, its journals/samplers/checkpoints and registered source patch unchanged. Its five trials remain separate cost/runtime evidence outside the new cohort. Reuse the corrected independent best/latest checkpoint implementation; do not add a v1-to-v2 trial/policy migration just to retain those observations. Future ordinary continuation within each new study remains distinct from importing an old trial.

## Preparation without training

```bash
python scripts/launch_exps/benchmark/prepare.py --run-name NEW_CAMPAIGN
python scripts/launch_exps/benchmark/prepare.py --run-name NEW_CAMPAIGN --write
python scripts/launch_exps/benchmark/prepare.py --run-name NEW_CAMPAIGN --verify
```

The default only prints the plan. `--write` creates five empty studies, fresh samplers and hashed configuration/source/artifact/data manifests, without allocating any trial. It refuses an existing campaign. `--verify` checks the prepared handoff read-only, including unchanged empty journal/sampler bytes. No training, baseline scoring, image inference or download occurs. A failed partial preparation stays retained and is not silently recreated.

R3 prepared `phase62-20261021-v2`, with0 attempts in each50-attempt study. Use `--verify`, not `--write`, for that existing name. Its exact identities and evidence are in the [runtime checkpoint](../../../docs/implementation_details/benchmark_training.md#r3-preparation-and-inspected-handoff--2026-10-21).

## First release

**Current authorization,2026-10-21:** the user accepts historical-cost estimates and waives a repeat cost pause. The continuous release uses the new preparation `phase62-20261021-v2-release1`; the original R3 empty preparation stays retained. This operational amendment leaves all50-attempt/scientific settings unchanged.

```bash
python scripts/launch_exps/benchmark/first_trials.py --run-name PREPARED_CAMPAIGN --prepared --full-budget --cost-reviewed
python scripts/launch_exps/benchmark/first_trials.py --run-name PREPARED_CAMPAIGN --prepared --full-budget --cost-reviewed --execute
```

The first command only prints. Explicit execution runs reference0/model, then49 additional attempts/model serially, without another cost pause. Both approval/prepared flags are mandatory. It records the cost basis before work and stops on errors; no replacement, extra trial or later-phase launch is implied. The ordinary first-release mode below remains available without full-budget approval.

```bash
python scripts/launch_exps/benchmark/first_trials.py --run-name NEW_CAMPAIGN
python scripts/launch_exps/benchmark/first_trials.py --run-name PREPARED_CAMPAIGN --prepared --execute
```

The first command prints the plan. The second explicitly starts the verified empty preparation, computes validation prior controls, then runs trial0 for ResNet18, frozen DINOv2, EfficientNetV2-S, MaxViT-T and DICA-Net-S in series. Each receives40 epochs, effective128 and its frozen physical batch. The queue stops on failure or after all five reach the cost-review boundary; it never releases the other49/model automatically. Without `--prepared`, execution retains the original fresh-directory behavior and refuses any existing campaign. No test/predict image evaluation belongs to this release.

New artifacts live below `experiments/benchmark_v2/NEW_CAMPAIGN/`: queue/baseline state, separate study journals/configurations/source inventories, epoch costs, best/latest checkpoints, stopping policy/pruning reasons and ordered validation exports. A terminal interruption is preserved for explicit recovery, without automatic retraining. The old v1 campaign path stays unchanged and is never an import source for this route.

## Later individual releases

### Reviewed checkpoint-only source repair

**Retained v1 commands only:** this historical helper remains pinned to the v1 policy and rejects further source drift. It is not a preparation step for v2 and must not rewrite the old registered patch to absorb R1 changes. The following commands describe the earlier repair interface, not a new action to perform with the active changed sources.

```bash
python scripts/launch_exps/benchmark/register_runtime_patch.py --run-name phase62-20261018-v1
python scripts/launch_exps/benchmark/register_runtime_patch.py --run-name phase62-20261018-v1 --apply --accept-missing-final
```

The first command only inspects. The second adds the reviewed `final-checkpoint-v1` source snapshot/amendment to every first-release study, explicitly retaining trial0's missing-final exception without rewriting its original manifest, journal, sampler or artifacts. It is idempotent for identical registered sources; unrelated changes fail. It does not train, release budget or enable early stopping. See the [source-transition contract](../../../docs/implementation_details/benchmark_training.md#explicit-checkpoint-repair-source-transition). Future attempts bind their source identity separately. Use `trainer.last_checkpoint_path` for the latest full state; the monitored callback only owns best-AP selection.

### Training after cost/policy approval

```bash
python scripts/launch_exps/benchmark/train.py
python scripts/launch_exps/benchmark/train.py --model maxvit_t --export-config NEW_CONFIG.json
python scripts/launch_exps/benchmark/train.py --execute --model maxvit_t --run-name CAMPAIGN
python scripts/launch_exps/benchmark/train.py --execute --model maxvit_t --run-name PREPARED_CAMPAIGN --prepared
python scripts/launch_exps/benchmark/train.py --execute --model maxvit_t --run-name CAMPAIGN --resume --cost-reviewed --additional-trials N
```

`--prepared` starts a verified empty study with the fixed reference0; it cannot be combined with `--resume` or repeated after startup. The last command is used only after the declared cost acknowledgment and only within a v2 study with finished attempts. It preserves that study/sampler, enforces50 total started attempts including pruned/failed ones and rejects partial/inconsistent states or v1 identities. Trial0 is freshly trained inside50; there is no imported reference. Temperature/global-threshold fitting, selected/random training and local adaptation remain later packages. R3 creates only the empty preparation, not a training launch.
