"""
Register 5-config sweeps for EuroSAT and DFC2020 (S1→S2 direction only).

Each sweep fixes one config's discrete hyperparameters and searches over the
continuous HP space defined in the per-dataset YAML.

Usage (from repo root):
    python sweep/create_sweep_multidataset.py [--dry-run]
    python sweep/create_sweep_multidataset.py --dry-run --datasets eurosat dfc2020
"""

import argparse
import copy
import os
import yaml
import wandb

_SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
_YAML_DIR  = os.path.join(_SWEEP_DIR, 'sweep_yaml')

# 5 configs shared across all datasets
CONFIGS = [
    {
        'name': 'delulu',
        'fixed': {
            'protect_lrm': 'False',
            'use_mask_token': 'False',
            'latent_masked_only': 'True',
            'labeled_start_fraction': '0.0',
        },
        'active_losses': ['latent', 'prefusion', 'distill', 'ce'],
    },
    {
        'name': 'mask-token',
        'fixed': {
            'protect_lrm': 'False',
            'use_mask_token': 'True',
            'latent_masked_only': 'True',
            'labeled_start_fraction': '0.0',
        },
        'active_losses': ['latent', 'distill', 'ce'],
    },
    {
        # lsf=1.0 + lf=0: labeled batches never appear → ce never active
        'name': 'no-batch-mixing',
        'fixed': {
            'protect_lrm': 'False',
            'use_mask_token': 'False',
            'latent_masked_only': 'True',
            'labeled_start_fraction': '1.0',
            'labeled_frequency': '0',
        },
        'active_losses': ['latent', 'prefusion', 'distill'],
    },
    {
        'name': 'no-prefusion',
        'fixed': {
            'protect_lrm': 'False',
            'use_mask_token': 'False',
            'latent_masked_only': 'True',
            'labeled_start_fraction': '0.0',
        },
        'active_losses': ['latent', 'distill', 'ce'],
    },
    {
        'name': 'no-latent',
        'fixed': {
            'protect_lrm': 'False',
            'use_mask_token': 'False',
            'latent_masked_only': 'True',
            'labeled_start_fraction': '0.0',
        },
        'active_losses': ['distill', 'prefusion', 'ce'],
    },
]

DATASETS = {
    'eurosat': {
        'yaml': 'sweep_eurosat_final.yaml',
        'dataset_arg': 'eurosat',
        'directions': [
            {
                'slug': 'nirrgb',
                'stage0': 'checkpoints/sft_evan_base_eurosat_nir_fft_lr0.001_20260422_043429.pt',
                'new_mod': 'rgb',
            },
        ],
        'n_jobs': 40,
    },
    'dfc2020': {
        'yaml': 'sweep_dfc2020_final.yaml',
        'dataset_arg': 'dfc2020',
        'directions': [
            {
                'slug': 's1s2',
                'stage0': 'checkpoints/sft_evan_base_dfc2020_s1_fft_lr0.0005_20260418_080704.pt',
                'new_mod': 's2',
            },
        ],
        'n_jobs': 40,
    },
    'benv2': {
        'yaml': 'sweep_benv2_final.yaml',
        'dataset_arg': 'benv2',
        'directions': [
            {
                'slug': 's1s2',
                'stage0': 'checkpoints/sft_evan_base_benv2_s1_fft_lr0.0005_20260418_064233.pt',
                'new_mod': 's2',
            },
            {
                'slug': 's2s1',
                'stage0': 'checkpoints/sft_evan_base_benv2_s2_fft_lr0.001_20260418_112953.pt',
                'new_mod': 's1',
            },
        ],
        'n_jobs': 40,
    },
}


def _load_merged_config(yaml_file: str) -> dict:
    with open(os.path.join(_YAML_DIR, 'base.yaml')) as f:
        base = yaml.safe_load(f)
    with open(os.path.join(_YAML_DIR, yaml_file)) as f:
        override = yaml.safe_load(f)
    merged = override.copy()
    merged['parameters'] = {
        **base.get('parameters', {}),
        **override.get('parameters', {}),
    }
    return merged


def _build_sweep(cfg: dict, direction: dict, dataset_arg: str, base_config: dict) -> tuple:
    config = copy.deepcopy(base_config)

    extra_args = [
        '--dataset', dataset_arg,
        '--stage0_checkpoint', direction['stage0'],
        '--new_mod_group', direction['new_mod'],
    ]
    for k, v in cfg['fixed'].items():
        extra_args.extend([f'--{k}', v])
    for loss in cfg['active_losses']:
        extra_args.extend(['--active_losses', loss])

    config['command'] = config.get('command', []) + extra_args

    if 'labeled_frequency' in cfg['fixed']:
        config['parameters'].pop('labeled_frequency', None)

    ds_short = dataset_arg  # e.g. 'eurosat', 'dfc2020'
    project = f"delulu-{ds_short}-final-{cfg['name']}-{direction['slug']}"
    config['project'] = project
    return config, project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()),
                        help='Which datasets to register sweeps for')
    args = parser.parse_args()

    registry_path = os.path.join(_SWEEP_DIR, 'sweep_registry.txt')

    all_registered = []  # list of (full_id, n_jobs)

    for ds_name in args.datasets:
        ds = DATASETS[ds_name]
        base_config = _load_merged_config(ds['yaml'])

        for direction in ds['directions']:
            for cfg in CONFIGS:
                config, project = _build_sweep(
                    cfg, direction, ds['dataset_arg'], base_config
                )

                print(f'\n{"="*60}')
                print(f'Dataset: {ds_name}  Config: {cfg["name"]}  Direction: {direction["slug"]}')
                print(f'Project: {project}')
                print(f'Fixed args: {cfg["fixed"]}')
                print(f'stage0: {direction["stage0"]}')

                if args.dry_run:
                    print('[dry-run] skipping W&B registration')
                    continue

                sweep_id = wandb.sweep(sweep=config, project=project)
                entity = wandb.Api().default_entity
                full_id = f'{entity}/{project}/{sweep_id}'

                print(f'\nSweep registered: {full_id}')
                print(f'Submit {ds["n_jobs"]} jobs:')
                print(f"  for i in $(seq 1 {ds['n_jobs']}); do sbatch sweep/run_sweep.sh '{full_id}'; done")

                all_registered.append((full_id, ds['n_jobs']))

                with open(registry_path, 'a') as f:
                    f.write(f'{project} {full_id} {direction["stage0"]} {direction["new_mod"]}\n')

    if all_registered:
        print(f'\n{"="*60}')
        print('All sbatch commands (copy-paste):')
        for full_id, n_jobs in all_registered:
            print(f"for i in $(seq 1 {n_jobs}); do sbatch sweep/run_sweep.sh '{full_id}'; done")


if __name__ == '__main__':
    main()
