"""
Register W&B sweeps for BioMassters Delulu on the MEAN-POOLED inputs.

Why this exists
---------------
The pooled crossconfig runs (128 epochs, lr 5e-4, 2 configs per cell) lost 10 of
12 cells to their own teacher. Those configs were tuned on benv2/dfc2020 and
carried over, so the deficit could be a hyperparameter artefact rather than a
property of the method. Input-pooling made a Delulu epoch ~4x cheaper (62s vs
245s), which finally makes a real search affordable: this sweeps the FULL
9-parameter space (lr, weight_decay, both modality dropouts, labeled_frequency,
token_mask_ratio, and the three loss weights) per direction.

Differences from create_sweep_biomassters.py (the T=12 version):
  - teachers come from the '/tpooled' registry keys, not hardcoded T=12 paths;
  - --num_time_steps is NEGATIVE, which makes the loader mean-pool at the input
    and read the on-disk cache (datasets/geoben2/biomassters_pooled);
  - separate W&B projects and separate results CSVs, so pooled and T=12 sweep
    rows can never be mixed;
  - all four directions from the paper table, not just the two S1/S2 ones.

Usage (from repo root):
    python sweep/create_sweep_biomassters_tpooled.py [--dry-run]
"""

import argparse
import copy
import json
import os
import yaml
import wandb

_SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SWEEP_DIR)
_YAML_DIR = os.path.join(_SWEEP_DIR, 'sweep_yaml')
_TEACHERS = os.path.join(_ROOT, 'artifacts', 'sft_teachers.json')

# The four table directions, as (start, new).
DIRECTIONS = [
    ('s1', 's2'),
    ('s2', 's1'),
    ('s2_rgb', 's2_norgb'),
    ('s2_norgb', 's2_rgb'),
]

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

# Negative => mean-pool the 12 timesteps at the input and use the cache.
NUM_TIME_STEPS = '-12'
# The pooled sample is [C,1,H,W], so the memory that forced bs=16 at T=12 is
# gone; 32 matches the other non-temporal datasets.
BATCH_SIZE = '32'
EPOCHS = '128'
# 128 runs total across 4 directions.
N_JOBS = 32


def _teacher_for(mod: str) -> str:
    with open(_TEACHERS) as f:
        reg = json.load(f)
    key = f'biomassters/{mod}/evan_base/upernet+relu/split1/tpooled'
    entry = reg.get(key)
    if entry is None:
        raise SystemExit(
            f"No pooled teacher for {mod!r} (key {key}).\n"
            f"Run:  python res/train_sft/sft_best.py   after the pooled stage-0 "
            f"runs have landed in res/train_sft/biomassters.csv")
    ckpt = entry['checkpoint']
    if not os.path.isfile(os.path.join(_ROOT, ckpt)):
        raise SystemExit(f"Teacher checkpoint missing on disk: {ckpt}")
    return ckpt


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


def _build_sweep(cfg, start, new, base_config):
    config = copy.deepcopy(base_config)
    slug = f'{start}to{new}'
    extra_args = [
        '--dataset', 'biomassters',
        '--stage0_checkpoint', _teacher_for(start),
        '--new_mod_group', new,
        '--num_time_steps', NUM_TIME_STEPS,
        '--batch_size', BATCH_SIZE,
        '--epochs', EPOCHS,
        '--results_csv',
        f'res/delulu-sweep/sweep_results_biomassters_tpooled_{slug}.csv',
    ]
    for k, v in cfg['fixed'].items():
        extra_args.extend([f'--{k}', v])
    for loss in cfg['active_losses']:
        extra_args.extend(['--active_losses', loss])

    config['command'] = config.get('command', []) + extra_args
    project = f"delulu-biomassters-tpooled-{cfg['name']}-{slug}"
    config['project'] = project
    return config, project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    base_config = _load_merged_config()
    registry_path = os.path.join(_SWEEP_DIR, 'sweep_registry.txt')
    all_full_ids = []

    for start, new in DIRECTIONS:
        for cfg in CONFIGS:
            config, project = _build_sweep(cfg, start, new, base_config)
            print(f'\n{"="*60}')
            print(f'Config: {cfg["name"]}  Direction: {start} -> {new}')
            print(f'Project: {project}')
            print(f'stage0: {_teacher_for(start)}')
            print(f'T={NUM_TIME_STEPS} (mean-pooled), bs={BATCH_SIZE}, epochs={EPOCHS}')

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
                f.write(f'{project} {full_id} {_teacher_for(start)} {new}\n')

    if all_full_ids:
        print(f'\n{"="*60}')
        print(f'All sbatch commands ({N_JOBS} agents each, {N_JOBS*len(all_full_ids)} runs total):')
        for full_id in all_full_ids:
            print(f"for i in $(seq 1 {N_JOBS}); do sbatch sweep/run_sweep.sh '{full_id}'; done")


if __name__ == '__main__':
    main()
