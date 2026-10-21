# Phase 6 benchmark launchers

**Created:** 2026-10-18
**Last updated:** 2026-10-21

Run from the main WSL repository with the project's ML Python. The [scientific protocol](../../../docs/project_objective/model_comparison_methodology.md#phase-6-d1--finite-training-and-comparison-protocol), [runtime contract](../../../docs/implementation_details/benchmark_training.md) and [execution plan](../../../docs/plans/training_hyperparameter_tuning.md) own policy, implementation and state.

## Current planning boundary — D3 fresh studies

[Phase6-D3](../../../docs/project_objective/model_comparison_methodology.md#phase-6-d3--fresh-fifty-trial-studies) now requires five new50-attempt studies beginning at fresh trial0, without importing or resuming the first release. D2's additional plateau guard must be implemented with that new policy. **The current commands below still implement v1/100 attempts without the guard**: they are retained interface/history examples, not an approved new-campaign recipe. Merely choosing a new directory does not activate D3. No new launch command is claimed until the [ordered restart checks](../../../docs/plans/training_hyperparameter_tuning.md#fresh-fifty-trial-restart--2026-10-21) pass and training is explicitly requested.

Preserve `phase62-20261018-v1`, its journals/samplers/checkpoints and registered source patch unchanged. Its five trials remain separate cost/runtime evidence outside the new cohort. Reuse the corrected independent best/latest checkpoint implementation; do not add a v1-to-v2 trial/policy migration just to retain those observations. Future ordinary continuation within each new study remains distinct from importing an old trial.

## First release

```bash
python scripts/launch_exps/benchmark/first_trials.py --run-name NEW_CAMPAIGN
python scripts/launch_exps/benchmark/first_trials.py --run-name NEW_CAMPAIGN --execute
```

The first command prints the plan. The second computes validation prior controls, then runs trial0 for ResNet18, frozen DINOv2, EfficientNetV2-S, MaxViT-T and DICA-Net-S in series. Each receives40 epochs, effective128 and its frozen physical batch. The queue stops on failure or after all five reach the cost-review boundary. Use a new name; existing campaign directories are rejected. No test/predict image evaluation belongs to this release.

Artifacts live below `experiments/benchmark_v1/NEW_CAMPAIGN/`: `campaign_queue.json`, `baselines/`, separate model study journals/configurations/source inventories, epoch costs, best/last checkpoints and ordered validation exports. A terminal interruption is preserved for explicit recovery, without automatic retraining.

## Later individual releases

### Reviewed checkpoint-only source repair

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
python scripts/launch_exps/benchmark/train.py --execute --model maxvit_t --run-name CAMPAIGN --resume --cost-reviewed --additional-trials N
```

The last command is used only after the declared cost review. It preserves the study and sampler, enforces the remaining100-started-trial allowance and rejects partial/inconsistent recovery states. It does not continue a historical loss-selected study. Temperature/global-threshold fitting, selected/random training and local adaptation remain later execution packages.
