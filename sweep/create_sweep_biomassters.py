"""
Register W&B sweeps for BioMassters SHOT: 2 modality directions (s2->s1, s1->s2).

Each sweep fixes the discrete config and searches the continuous HP space in
sweep_yaml/sweep_biomassters.yaml (+ base.yaml). Mirrors create_sweep_final.py
but for biomassters: T=12 teachers, regression metric (val/addition_score,
maximized because it is negative RMSE), modality protection disabled.

Usage (from repo root):
    python sweep/create_sweep_biomassters.py [--dry-run]
"""

import argparse
import copy
import os
import yaml
import wandb

_SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
_YAML_DIR = os.path.join(_SWEEP_DIR, 'sweep_yaml')

# Best-val (lowest RMSE) T=12 UperNet+relu teachers, from res/train_sft/biomassters.csv.
DIRECTIONS = [
    {
        'slug': 's2s1',
        'stage0': 'checkpoints/sft_evan_base_biomassters_s2_fft_lr0.0005_20260725_075836.pt',
        'new_mod': 's1',
    },
    {
        'slug': 's1s2',
        'stage0': 'checkpoints/sft_evan_base_biomassters_s1_fft_lr0.0005_20260725_075947.pt',
        'new_mod': 's2',
    },
]

# Single config mirroring the tuned biomassters runs: modality protection off
# (unprotect_starting_mod), labeled mixing from the start, all four losses.
CONFIGS = [
    {
        'name': 'delulu',
        'fixed': {
            'protect_lrm': '0.0',
            'use_mask_token': 'False',
            'latent_masked_only': 'True',
            'unprotect_starting_mod': 'True',
            'labeled_start_fraction': '0',
        },
        'active_losses': ['latent', 'prefusion', 'distill', 'ce'],
    },
]

NUM_TIME_STEPS = '12'
BATCH_SIZE = '16'          # T=12: bs=32 OOMs, bs=8 fits; try 16 (verify headroom).
N_JOBS = 64


def _load_merged_config() -> dict:
    with open(os.path.join(_YAML_DIR, 'base.yaml')) as f:
        base = yaml.safe_load(f)
    with open(os.path.join(_YAML_DIR, 'sweep_biomassters.yaml')) as f:
        override = yaml.safe_load(f)
    merged = override.copy()
    merged['parameters'] = {
        **base.get('parameters', {}),
        **override.get('parameters', {}),
    }
    return merged


def _build_sweep(cfg, direction, base_config):
    config = copy.deepcopy(base_config)
    extra_args = [
        '--dataset', 'biomassters',
        '--stage0_checkpoint', direction['stage0'],
        '--new_mod_group', direction['new_mod'],
        '--num_time_steps', NUM_TIME_STEPS,
        '--batch_size', BATCH_SIZE,
        '--results_csv', f"res/delulu-sweep/sweep_results_biomassters_{direction['slug']}.csv",
    ]
    for k, v in cfg['fixed'].items():
        extra_args.extend([f'--{k}', v])
    for loss in cfg['active_losses']:
        extra_args.extend(['--active_losses', loss])

    config['command'] = config.get('command', []) + extra_args
    project = f"delulu-biomassters-{cfg['name']}-{direction['slug']}"
    config['project'] = project
    return config, project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
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
            print(f'Fixed args: {cfg["fixed"]}  active_losses={cfg["active_losses"]}')
            print(f'stage0: {direction["stage0"]}')

            if args.dry_run:
                print('[dry-run] skipping W&B registration')
                continue

            sweep_id = wandb.sweep(sweep=config, project=project)
            entity = wandb.Api().default_entity
            full_id = f'{entity}/{project}/{sweep_id}'
            print(f'\nSweep registered: {full_id}')
            print(f"  for i in $(seq 1 {N_JOBS}); do sbatch sweep/run_sweep.sh '{full_id}'; done")
            all_full_ids.append(full_id)
            with open(registry_path, 'a') as f:
                f.write(f'{project} {full_id} {direction["stage0"]} {direction["new_mod"]}\n')

    if all_full_ids:
        print(f'\n{"="*60}')
        print('All sbatch commands (copy-paste):')
        for full_id in all_full_ids:
            print(f"for i in $(seq 1 {N_JOBS}); do sbatch sweep/run_sweep.sh '{full_id}'; done")


if __name__ == '__main__':
    main()
