"""Read retained metadata only; never open a model, images or an Optuna journal writer."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import median


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    sources = {}

    def read(relative):
        path = root / relative
        sources[relative] = sha256(path)
        return json.loads(path.read_text())

    inventory = read('src_scratches/experiment_comparison_audit/inventory.json')
    journal = inventory['journal']
    sources[journal['path']] = sha256(root / journal['path'])
    if sources[journal['path']] != journal['sha256']:
        raise ValueError('Journal differs from retained inventory; audit must be refreshed')
    studies = {}
    for study in journal['studies']:
        name = study['name_suffix'].split('/')[-1]
        trials = study['trials']
        configs = {t['number']: t for t in inventory['experiments'][name]['trials']}
        if len(trials) != 100 or len(configs) != 100:
            raise ValueError('Expected the reviewed 100-trial historical cohort')
        for number, config in configs.items():
            relative = f'experiments/basic_v5/{name}/trial_{number}/trial_config.json'
            actual = sha256(root / relative)
            if actual != config['config_sha256']:
                raise ValueError(f'Configuration changed: {relative}')
            sources[relative] = actual
        summary = {}
        for state in ['ALL', 'COMPLETE', 'PRUNED']:
            group = [t for t in trials if state == 'ALL' or t['state'] == state]
            times = [t['duration_seconds'] for t in group if t['duration_seconds'] is not None]
            if not times or any(t < 0 for t in times):
                raise ValueError('Missing or negative historical duration cohort')
            summary[state] = {
                'count': len(group), 'missing_durations': len(group) - len(times),
                'recorded_hours': sum(times) / 3600,
                'median_minutes': median(times) / 60,
            }
        counts, seconds = Counter(), Counter()
        for trial in trials:
            model = configs[trial['number']]['model']['type']
            counts[model] += 1
            seconds[model] += trial['duration_seconds'] or 0
        summary['model_allocation'] = {
            model: {'trials': count, 'recorded_hours': seconds[model] / 3600}
            for model, count in sorted(counts.items())
        }
        studies[name] = summary

    smoke_root = root / 'experiments/runtime_smoke/phase55-20261007-a4'
    paths = sorted(smoke_root.glob('*/smoke-*/result.json'))
    if len(paths) != 6:
        raise ValueError('Expected all six qualified full/selected smoke results')
    smokes = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        data = read(relative)
        train, batch = data['training'], data['batch_plan']
        if train['completed_optimizer_steps'] != 4 or len(train['step_seconds']) != 4:
            raise ValueError(f'Unexpected smoke step count: {relative}')
        smokes.append({
            'path': relative, 'model_id': data['model_id'], 'selected': data['selected'],
            'physical_batch': batch['physical_batch_size'],
            'accumulation': batch['accumulate_grad_batches'],
            'effective_batch': batch['requested_batch_size'],
            'step_seconds': train['step_seconds'], 'training_seconds': train['training_seconds'],
            # Sensitivity arithmetic, NOT a throughput estimate or confidence bound.
            'conditional_40_epoch_update_hours': median(train['step_seconds'][-2:]) * 375 * 40 / 3600,
        })
    manifest = read('analysis_outputs/ingredient_selection/phase3-d1-v3/campaign_manifest.json')
    if manifest['status'] != 'completed':
        raise ValueError('Selector campaign is not complete')
    selector_hours = (
        datetime.fromisoformat(manifest['completed_at_utc'])
        - datetime.fromisoformat(manifest['started_at_utc'])
    ).total_seconds() / 3600
    triple_hours = sum(s['conditional_40_epoch_update_hours'] for s in smokes if not s['selected'])
    return {
        'schema_version': 1, 'evidence_cutoff': '2026-10-13',
        'limitations': [
            'Journal durations are not GPU-active hours; resumed work may be missing.',
            'Four cold smoke updates are not a full-epoch throughput measurement.',
            'Conditional extrapolation excludes validation, checkpoints and new metric overhead.',
            'No new model construction, training, prediction or test-image access.',
        ],
        'historical_studies': studies, 'smokes': smokes,
        'selector_wall_hours': selector_hours,
        'conditional_new_portfolio_full_run_hours': triple_hours,
        'conditional_unpruned_new_portfolio_days': {
            str(n): n * triple_hours / 24 for n in [20, 40, 100]
        },
        'source_sha256': sources,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, help='New output file; refuses overwrite')
    args = parser.parse_args()
    result = json.dumps(audit(args.root.resolve()), indent=2, sort_keys=True) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(result)
    else:
        print(result, end='')
