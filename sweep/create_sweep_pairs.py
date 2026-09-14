"""Register the two deep DeluluNet sweeps: 128 trials each, one per dataset.

Rather than a shallow sweep over many directions, this searches two directions
hard and tests whether the resulting config generalises:

  dfc2020  s1 -> +s2_norgb   cross-sensor, high headroom (SAR teacher gains optical)
  benv2    s2_rgb -> +s2_norgb   within-sensor (optical teacher gains the other bands)

128 trials is chosen from the noise floor, not convenience: seed-to-seed sd on
DFC2020 transfer is ~1.09 mIoU (max 1.85, measured over 9 repeated-config groups
in res/delulu/dfc2020_cobench_upernet.csv), while the earlier 16-trial DFC2020
sweeps spanned only sd 0.98-1.70. At n=16 the "best" config was therefore
indistinguishable from a lucky seed. n=128 puts the top of the sampled
distribution clearly above that floor.

Usage:
  python sweep/create_sweep_pairs.py --dry-run
  python sweep/create_sweep_pairs.py
"""
import argparse
import copy
import json
import os

import wandb
import yaml

_SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SWEEP_DIR)
_YAML_DIR = os.path.join(_SWEEP_DIR, 'sweep_yaml')
_TEACHERS = os.path.join(_ROOT, 'artifacts', 'sft_teachers.json')

MODEL = 'evan_base'
# split1: stage 1 uses train2 as its unlabeled pool, so a `full` teacher has
# already seen that pool with labels and the comparison leaks.
TEACHER_SPLIT = 'split1'
N_TRIALS = 128
# weight_decay is inert (|rho| <= 0.04 on all three objectives across 8 sweeps),
# so it is pinned here instead of consuming a search dimension. 2e-4 is the
# top-20% median from those sweeps.
WEIGHT_DECAY = '2e-4'

DIRECTIONS = [
    {
        'dataset': 'dfc2020', 'registry_prefix': 'dfc2020_cobench', 'decoder': 'upernet',
        'start': 's1', 'new_mod': 's2_norgb', 'slug': 'dfc2020_s1_to_s2norgb',
        # 128 epochs: the 64-epoch runs were still climbing (+0.23 mIoU/10ep);
        # at 128 the tail slope flattens to +0.03.
        'epochs': '128', 'batch_size': '8',
    },
    {
        'dataset': 'benv2', 'registry_prefix': 'benv2', 'decoder': 'cls',
        'start': 's2_rgb', 'new_mod': 's2_norgb', 'slug': 'benv2_s2rgb_to_s2norgb',
        'epochs': '64', 'batch_size': '32',
    },
]

# Matches the manual runs and the earlier sweeps: modality protection off,
# labeled mixing from step 0, all four losses, masked-only latent reconstruction.
FIXED = {
    'protect_lrm': '0.0',
    'latent_masked_only': 'True',
    'unprotect_starting_mod': 'True',
    'labeled_start_fraction': '0',
    'weight_decay': WEIGHT_DECAY,
}
ACTIVE_LOSSES = ['latent', 'prefusion', 'distill', 'ce']


def _teacher_for(direction) -> str:
    with open(_TEACHERS) as f:
        reg = json.load(f)
    key = (f"{direction['registry_prefix']}/{direction['start']}/"
           f"{MODEL}/{direction['decoder']}/{TEACHER_SPLIT}")
    if key not in reg:
        raise SystemExit(f'no teacher registered under {key!r}; '
                         'run python res/train_sft/sft_best.py')
    ck = reg[key]['checkpoint']
    if not os.path.exists(os.path.join(_ROOT, ck)):
        raise SystemExit(f'teacher checkpoint missing: {ck}')
    return ck


def _load_config() -> dict:
    # base.yaml also defines weight_decay; it is dropped here because this sweep
    # pins it on the command line instead of searching it.
    with open(os.path.join(_YAML_DIR, 'base.yaml')) as f:
        base = yaml.safe_load(f)
    with open(os.path.join(_YAML_DIR, 'sweep_pairs_nobal.yaml')) as f:
        override = yaml.safe_load(f)
    merged = override.copy()
    params = {**base.get('parameters', {}), **override.get('parameters', {})}
    params.pop('weight_decay', None)
    merged['parameters'] = params
    return merged


def _build(direction, base_config):
    config = copy.deepcopy(base_config)
    extra = [
        '--dataset', direction['dataset'],
        '--stage0_checkpoint', _teacher_for(direction),
        '--new_mod_group', direction['new_mod'],
        '--epochs', direction['epochs'],
        '--batch_size', direction['batch_size'],
        '--loss_balance', 'none',
        '--results_csv',
        f"res/delulu-sweep/sweep_results_{direction['slug']}_deep.csv",
    ]
    for k, v in FIXED.items():
        extra.extend([f'--{k}', v])
    for loss in ACTIVE_LOSSES:
        extra.extend(['--active_losses', loss])
    config['command'] = config.get('command', []) + extra
    project = f"delulu-deep-{direction['slug']}"
    config['project'] = project
    return config, project


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    base_config = _load_config()
    registry_path = os.path.join(_SWEEP_DIR, 'sweep_registry.txt')

    for d in DIRECTIONS:
        config, project = _build(d, base_config)
        print(f'\n{"=" * 66}')
        print(f"{d['dataset']}  {d['start']} -> +{d['new_mod']}    project: {project}")
        print(f"teacher: {_teacher_for(d)}")
        print(f"epochs={d['epochs']} bs={d['batch_size']} trials={N_TRIALS} "
              f"swept={len(config['parameters'])} params (weight_decay pinned {WEIGHT_DECAY})")
        if args.dry_run:
            print('[dry-run] not registering')
            continue
        sweep_id = wandb.sweep(sweep=config, project=project)
        entity = wandb.Api().default_entity
        full = f'{entity}/{project}/{sweep_id}'
        print(f'registered: {full}')
        print(f"  for i in $(seq 1 {N_TRIALS}); do sbatch sweep/run_sweep.sh '{full}'; done")
        with open(registry_path, 'a') as f:
            f.write(f"{project} {full} {_teacher_for(d)} {d['new_mod']}\n")


if __name__ == '__main__':
    main()
