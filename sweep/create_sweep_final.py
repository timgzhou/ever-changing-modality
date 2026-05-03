"""
Register 8 W&B sweeps: 4 named configs × 2 modality directions (s2→s1, s1→s2).

Each sweep fixes one config's discrete hyperparameters and searches over the
continuous HP space defined in sweep_yaml/sweep_benv2_final.yaml.

Usage (from repo root):
    python sweep/create_sweep_final.py [--dry-run]

--dry-run: print command args and config without registering with W&B.
"""

import argparse
import copy
import os
import yaml
import wandb

_SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
_YAML_DIR = os.path.join(_SWEEP_DIR, 'sweep_yaml')

DIRECTIONS = [
    {
        'slug': 's2s1',
        'stage0': 'checkpoints/sft_evan_base_benv2_s2_fft_lr0.001_20260418_112953.pt',
        'new_mod': 's1',
    },
    {
        'slug': 's1s2',
        'stage0': 'checkpoints/sft_evan_base_benv2_s1_fft_lr0.0005_20260418_064233.pt',
        'new_mod': 's2',
    },
]

# Each config: fixed CLI args appended to the sweep command.
# Keys map directly to sweep_shot.py argparse arguments.
CONFIGS = [
    {
        'name': 'delulu',
        'fixed': {
            'protect_lrm': 'False',
            'use_mask_token': 'False',
            'latent_masked_only': 'True',
            'labeled_start_fraction': '0.5',
        },
    },
    {
        'name': 'mask-token',
        'fixed': {
            'protect_lrm': 'False',
            'use_mask_token': 'True',
            'latent_masked_only': 'True',
            'labeled_start_fraction': '0.5',
            'use_prefusion': 'False',
        },
    },
    {
        # lsf=1.0 + lf=0: labeled batches never appear → ce never active
        # → active_losses always distill+latent+prefusion
        'name': 'no-batch-mixing',
        'fixed': {
            'protect_lrm': 'False',
            'use_mask_token': 'False',
            'latent_masked_only': 'True',
            'labeled_start_fraction': '1.0',
            'labeled_frequency': '0',
        },
    },
    {
        'name': 'no-prefusion',
        'fixed': {
            'protect_lrm': 'False',
            'use_mask_token': 'False',
            'latent_masked_only': 'True',
            'labeled_start_fraction': '0.5',
            'use_prefusion': 'False',
        },
    },
    {
        'name': 'no-latent',
        'fixed': {
            'protect_lrm': 'False',
            'use_mask_token': 'False',
            'latent_masked_only': 'True',
            'labeled_start_fraction': '0.5',
            'no_latent': 'True',
        },
    },
]


def _load_merged_config() -> dict:
    with open(os.path.join(_YAML_DIR, 'base.yaml')) as f:
        base = yaml.safe_load(f)
    with open(os.path.join(_YAML_DIR, 'sweep_benv2_final.yaml')) as f:
        override = yaml.safe_load(f)
    merged = override.copy()
    merged['parameters'] = {
        **base.get('parameters', {}),
        **override.get('parameters', {}),
    }
    return merged


def _build_sweep(cfg: dict, direction: dict, base_config: dict) -> dict:
    config = copy.deepcopy(base_config)

    # Fixed args that vary per (config, direction)
    extra_args = [
        '--dataset', 'benv2',
        '--stage0_checkpoint', direction['stage0'],
        '--new_mod_group', direction['new_mod'],
    ]
    for k, v in cfg['fixed'].items():
        extra_args.extend([f'--{k}', v])

    config['command'] = config.get('command', []) + extra_args

    # If labeled_frequency is fixed to 0, remove it from the swept parameters
    if 'labeled_frequency' in cfg['fixed']:
        config['parameters'].pop('labeled_frequency', None)

    project = f"delulu-benv2-final-{cfg['name']}-{direction['slug']}"
    config['project'] = project
    return config, project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configs without registering with W&B')
    args = parser.parse_args()

    base_config = _load_merged_config()
    registry_path = os.path.join(_SWEEP_DIR, 'sweep_registry.txt')

    all_full_ids = []

    for direction in DIRECTIONS:
        for cfg in CONFIGS:
            config, project = _build_sweep(cfg, direction, base_config)

            print(f'\n{"="*60}')
            print(f'Config: {cfg["name"]}  Direction: {direction["slug"]}')
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
            print(f'Submit 136 jobs:')
            print(f"  for i in $(seq 1 136); do sbatch sweep/run_sweep.sh '{full_id}'; done")

            all_full_ids.append(full_id)

            with open(registry_path, 'a') as f:
                f.write(f'{project} {full_id} {direction["stage0"]} {direction["new_mod"]}\n')

    if all_full_ids:
        print(f'\n{"="*60}')
        print('All sbatch commands (copy-paste):')
        for full_id in all_full_ids:
            print(f"for i in $(seq 1 136); do sbatch sweep/run_sweep.sh '{full_id}'; done")


if __name__ == '__main__':
    main()
