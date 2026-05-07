"""
Register one W&B sweep: single config, s1→s2 direction, 64 runs.

Sweeps per-modality dropout, protect_lrm schedule, unprotect_starting_mod,
labeled_frequency, loss weights, and mae_mask_ratio.

Usage (from repo root):
    python sweep/create_sweep_masking.py [--dry-run]

--dry-run: print command args and config without registering with W&B.
"""

import argparse
import copy
import os
import yaml
import wandb

_SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
_YAML_DIR = os.path.join(_SWEEP_DIR, 'sweep_yaml')

DIRECTION = {
    'slug': 's1s2',
    'stage0': 'checkpoints/sft_evan_base_benv2_s1_fft_lr0.0005_20260418_064233.pt',
    'new_mod': 's2',
}

# Fixed discrete args injected into the sweep command.
# protect_lrm and unprotect_starting_mod are swept via the YAML values fields.
FIXED = {
    'latent_masked_only': 'True',
    'use_mask_token': 'False',
    'labeled_start_fraction': '0',
}


def _load_config() -> dict:
    with open(os.path.join(_YAML_DIR, 'sweep_benv2_masking.yaml')) as f:
        return yaml.safe_load(f)


def _build_sweep(base_config: dict) -> tuple[dict, str]:
    config = copy.deepcopy(base_config)

    extra_args = [
        '--dataset', 'benv2',
        '--stage0_checkpoint', DIRECTION['stage0'],
        '--new_mod_group', DIRECTION['new_mod'],
    ]
    for k, v in FIXED.items():
        extra_args.extend([f'--{k}', v])

    # active_losses requires repeated flags (action='append' in sweep_shot.py)
    extra_args += [
        '--active_losses', 'latent',
        '--active_losses', 'prefusion',
        '--active_losses', 'distill',
        '--active_losses', 'ce',
    ]

    config['command'] = config.get('command', []) + extra_args

    project = f"delulu-benv2-masking-{DIRECTION['slug']}"
    config['project'] = project
    return config, project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Print config without registering with W&B')
    args = parser.parse_args()

    base_config = _load_config()
    config, project = _build_sweep(base_config)

    print(f'\n{"="*60}')
    print(f'Project: {project}')
    print(f'Fixed args: {FIXED}')
    print(f'Direction: {DIRECTION["slug"]}  stage0: {DIRECTION["stage0"]}')
    print(f'Swept parameters: {list(config["parameters"].keys())}')

    if args.dry_run:
        import json
        print('\n[dry-run] Full config:')
        print(json.dumps(config, indent=2))
        return

    sweep_id = wandb.sweep(sweep=config, project=project)
    entity = wandb.Api().default_entity
    full_id = f'{entity}/{project}/{sweep_id}'

    print(f'\nSweep registered: {full_id}')
    print(f'Submit 64 jobs:')
    print(f"  for i in $(seq 1 64); do sbatch sweep/run_sweep.sh '{full_id}'; done")

    registry_path = os.path.join(_SWEEP_DIR, 'sweep_registry.txt')
    with open(registry_path, 'a') as f:
        f.write(f'{project} {full_id} {DIRECTION["stage0"]} {DIRECTION["new_mod"]}\n')


if __name__ == '__main__':
    main()
