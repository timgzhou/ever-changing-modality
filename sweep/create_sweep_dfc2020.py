"""
Register W&B sweeps for DFC2020 SHOT on the Copernicus-Bench split.

Three directions, one per starting modality, each the best-performing addition
from that teacher in the 128-epoch manual runs:

    s1       -> +s2_norgb   teacher 52.55, addition 57.21 (+4.66)  high headroom
    s2_rgb   -> +s2_norgb   teacher 62.05, addition 62.84 (+0.79)  marginal
    s2_norgb -> +s2_rgb     teacher 66.74, addition 65.45 (-1.29)  strongest teacher

Teachers are the best-by-val upernet checkpoints from
artifacts/sft_teachers.json (key: dfc2020_cobench/<mod>/evan_base/upernet).
The decoder is restored from the checkpoint config, so the student is upernet
too -- do NOT mix in a linear teacher.

Usage (from repo root):
    python sweep/create_sweep_dfc2020.py [--dry-run]
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

DECODER = 'upernet'
# split1: stage 1 uses train2 as the unlabeled pool, so a `full` teacher has
# already seen it with labels (see res/train_sft/README_dfc2020.md).
TEACHER_SPLIT = 'split1'
MODEL = 'evan_base'
# 128 epochs: the 64-epoch runs were still climbing (tail slope +0.23 mIoU/10ep);
# at 128 the slope flattens to +0.03, so this is the converged budget.
EPOCHS = '128'
BATCH_SIZE = '8'
N_JOBS = 24          # trials per sweep

DIRECTIONS = [
    {'slug': 's1_to_s2norgb',      'start': 's1',       'new_mod': 's2_norgb'},
    {'slug': 's2rgb_to_s2norgb',   'start': 's2_rgb',   'new_mod': 's2_norgb'},
    {'slug': 's2norgb_to_s2rgb',   'start': 's2_norgb', 'new_mod': 's2_rgb'},
]

# Matches the manual runs: modality protection off, labeled mixing from step 0,
# all four losses, masked-only latent reconstruction.
CONFIG = {
    'name': 'delulu',
    'fixed': {
        'protect_lrm': '0.0',
        'use_mask_token': 'False',
        'latent_masked_only': 'True',
        'unprotect_starting_mod': 'True',
        'labeled_start_fraction': '0',
    },
    'active_losses': ['latent', 'prefusion', 'distill', 'ce'],
}

# Automatic loss balancing replaces the three swept lambdas with four learned
# log-variance scalars (Kendall et al. 2018). See loss_balancing.py.
LOSS_BALANCE = os.environ.get('LOSS_BALANCE', 'uncertainty')
# v2 of the balanced sweep: CE is anchored at --lambda_ce (not learned), and
# only latent/prefusion/distill get learned log-variances. See loss_balancing.py
# for why -- CE never co-occurs with the other three in a batch.
SWEEP_TAG = os.environ.get('SWEEP_TAG', 'uncbal2')


def _teacher_for(modality: str) -> str:
    with open(_TEACHERS) as f:
        reg = json.load(f)
    key = f'dfc2020_cobench/{modality}/{MODEL}/{DECODER}/{TEACHER_SPLIT}'
    if key not in reg:
        raise SystemExit(f'no teacher registered under {key!r}; '
                         'run python res/train_sft/sft_best.py')
    ck = reg[key]['checkpoint']
    if not os.path.exists(os.path.join(_ROOT, ck)):
        raise SystemExit(f'teacher checkpoint missing: {ck}')
    return ck


def _load_merged_config() -> dict:
    with open(os.path.join(_YAML_DIR, 'base.yaml')) as f:
        base = yaml.safe_load(f)
    # control arm (--loss_balance none) sweeps the three lambdas itself; the
    # balanced arm drops them because the balancer learns them.
    _yaml = ('sweep_dfc2020_cobench_nobal.yaml' if LOSS_BALANCE == 'none'
             else 'sweep_dfc2020_cobench.yaml')
    with open(os.path.join(_YAML_DIR, _yaml)) as f:
        override = yaml.safe_load(f)
    merged = override.copy()
    merged['parameters'] = {**base.get('parameters', {}),
                            **override.get('parameters', {})}
    return merged


def _build_sweep(direction, base_config):
    config = copy.deepcopy(base_config)
    extra = [
        '--dataset', 'dfc2020',
        '--stage0_checkpoint', _teacher_for(direction['start']),
        '--new_mod_group', direction['new_mod'],
        '--epochs', EPOCHS,
        '--batch_size', BATCH_SIZE,
        '--results_csv',
        f"res/delulu-sweep/sweep_results_dfc2020_cobench_{direction['slug']}_{SWEEP_TAG}.csv",
    ]
    extra.extend(['--loss_balance', LOSS_BALANCE])
    for k, v in CONFIG['fixed'].items():
        extra.extend([f'--{k}', v])
    for loss in CONFIG['active_losses']:
        extra.extend(['--active_losses', loss])
    config['command'] = config.get('command', []) + extra
    # v2: the v1 sweeps (registered 2026-08-21, cancelled) used `full`
    # teachers that had already seen the train2 unlabeled pool. New project
    # names so the leaked trials never mix with these.
    project = f"delulu-dfc2020-cobench-{direction['slug']}-{SWEEP_TAG}"
    config['project'] = project
    return config, project


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    base_config = _load_merged_config()
    registry_path = os.path.join(_SWEEP_DIR, 'sweep_registry.txt')
    ids = []

    for d in DIRECTIONS:
        config, project = _build_sweep(d, base_config)
        print(f'\n{"="*62}')
        print(f'Direction: {d["start"]} -> +{d["new_mod"]}   project: {project}')
        print(f'teacher : {_teacher_for(d["start"])}')
        print(f'epochs={EPOCHS} bs={BATCH_SIZE} decoder={DECODER}')
        if args.dry_run:
            print('[dry-run] not registering')
            continue
        sweep_id = wandb.sweep(sweep=config, project=project)
        entity = wandb.Api().default_entity
        full = f'{entity}/{project}/{sweep_id}'
        print(f'registered: {full}')
        print(f"  for i in $(seq 1 {N_JOBS}); do sbatch sweep/run_sweep.sh '{full}'; done")
        ids.append(full)
        with open(registry_path, 'a') as f:
            f.write(f'{project} {full} {_teacher_for(d["start"])} {d["new_mod"]}\n')

    if ids:
        print(f'\n{"="*62}\nAll sbatch loops:')
        for full in ids:
            print(f"for i in $(seq 1 {N_JOBS}); do sbatch sweep/run_sweep.sh '{full}'; done")


if __name__ == '__main__':
    main()
