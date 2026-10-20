# Phase 6 benchmark launchers

**Created:** 2026-10-18
**Last updated:** 2026-10-18

Run from the main WSL repository with the project's ML Python. The [scientific protocol](../../../docs/project_objective/model_comparison_methodology.md#phase-6-d1--finite-training-and-comparison-protocol), [runtime contract](../../../docs/implementation_details/benchmark_training.md) and [execution plan](../../../docs/plans/training_hyperparameter_tuning.md) own policy, implementation and state.

## First release

```bash
python scripts/launch_exps/benchmark/first_trials.py --run-name NEW_CAMPAIGN
python scripts/launch_exps/benchmark/first_trials.py --run-name NEW_CAMPAIGN --execute
```

The first command prints the plan. The second computes validation prior controls, then runs trial0 for ResNet18, frozen DINOv2, EfficientNetV2-S, MaxViT-T and DICA-Net-S in series. Each receives40 epochs, effective128 and its frozen physical batch. The queue stops on failure or after all five reach the cost-review boundary. Use a new name; existing campaign directories are rejected. No test/predict image evaluation belongs to this release.

Artifacts live below `experiments/benchmark_v1/NEW_CAMPAIGN/`: `campaign_queue.json`, `baselines/`, separate model study journals/configurations/source inventories, epoch costs, best/last checkpoints and ordered validation exports. A terminal interruption is preserved for explicit recovery, without automatic retraining.

## Later individual releases

```bash
python scripts/launch_exps/benchmark/train.py
python scripts/launch_exps/benchmark/train.py --model maxvit_t --export-config NEW_CONFIG.json
python scripts/launch_exps/benchmark/train.py --execute --model maxvit_t --run-name CAMPAIGN
python scripts/launch_exps/benchmark/train.py --execute --model maxvit_t --run-name CAMPAIGN --resume --cost-reviewed --additional-trials N
```

The last command is used only after the declared cost review. It preserves the study and sampler, enforces the remaining100-started-trial allowance and rejects partial/inconsistent recovery states. It does not continue a historical loss-selected study. Temperature/global-threshold fitting, selected/random training and local adaptation remain later execution packages.
