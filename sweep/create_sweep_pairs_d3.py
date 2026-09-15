"""Register a depth-3 DeluluNet sweep on DFC2020 (128 trials).

Companion to create_sweep_pairs.py, which swept the same space at the default
projector depth of 2. The depth ablation (res/delulu/dfc2020_projector_depth.csv)
found depth 3 slightly ahead on addition (+0.38, t=2.33, 5/6 paired wins) while
running hyperparameters tuned AT depth 2 -- a handicap. This sweep removes that
handicap by tuning at depth 3, so the two 128-trial sweeps are directly
comparable and the question becomes whether a depth-3 model, properly tuned,
beats the best depth-2 model.

  dfc2020  s1 -> +s2_norgb   same direction as the depth-2 sweep

128 trials is chosen from the noise floor, not convenience: seed-to-seed sd on
DFC2020 transfer is ~1.09 mIoU (max 1.85, measured over 9 repeated-config groups
in res/delulu/dfc2020_cobench_upernet.csv), while the earlier 16-trial DFC2020
sweeps spanned only sd 0.98-1.70. At n=16 the "best" config was therefore
indistinguishable from a lucky seed. n=128 puts the top of the sampled
distribution clearly above that floor.

Usage:
  python sweep/create_sweep_pairs_d3.py --dry-run
  python sweep/create_sweep_pairs_d3.py
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
# Projector depth: (PROJ_DEPTH-1) self-attention blocks + 1 cross-attention block.
PROJ_DEPTH = 3
# weight_decay is inert (|rho| <= 0.04 on all three objectives across 8 sweeps),
# so it is pinned here instead of consuming a search dimension. 2e-4 is the
# top-20% median from those sweeps.
WEIGHT_DECAY = '2e-4'

DIRECTIONS = [
    {
        'dataset': 'dfc2020', 'registry_prefix': 'dfc2020_cobench', 'decoder': 'upernet',
        'start': 's1', 'new_mod': 's2_norgb', 'slug': 'dfc2020_s1_to_s2norgb',
        # 128 epochs: matches the depth-2 sweep so the two are comparable.
        'epochs': '128', 'batch_size': '8',
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
        '--intermediate_projector_num_layers', str(PROJ_DEPTH),
        '--results_csv',
        f"res/delulu-sweep/sweep_results_{direction['slug']}_deep_d{PROJ_DEPTH}.csv",
    ]
    for k, v in FIXED.items():
        extra.extend([f'--{k}', v])
    for loss in ACTIVE_LOSSES:
        extra.extend(['--active_losses', loss])
    config['command'] = config.get('command', []) + extra
    project = f"delulu-deep-{direction['slug']}-d{PROJ_DEPTH}"
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
              f"swept={len(config['parameters'])} params (weight_decay pinned {WEIGHT_DECAY}, "
              f"projector depth {PROJ_DEPTH})")
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
