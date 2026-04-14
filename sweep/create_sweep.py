#!/usr/bin/env python
"""
Create a wandb sweep for a given script and modality/dataset combination.

Usage:
    # Original SHOT sweeps (EuroSAT only):
    python create_sweep.py --script shot --starting nir --newmod rgb
    python create_sweep.py --script shot --starting rgb --newmod nir --dino

    # Stage-0 SFT:
    python create_sweep.py --script sft --dataset benv2 --modalities s2_rgb
    python create_sweep.py --script sft --dataset eurosat --modalities rgb

    # Baseline distillation:
    python create_sweep.py --script baseline_distill --dataset benv2 --teacher_checkpoint checkpoints/benv2_s2_s0.pt --modality s2_rgb

For sft and baseline_distill the merged config (base.yaml + script yaml) is
registered and the fixed args are appended to the command automatically.
"""

import argparse
import os
import yaml
import wandb

# ---------------------------------------------------------------------------
# SHOT sweep
# ---------------------------------------------------------------------------

CHECKPOINT_MAP = {
    "nir":  "checkpoints/nir_randinit_fft.pt",
    "rgb":  "checkpoints/rgb_randinit_fft.pt",
    "vre":  "checkpoints/vre_randinit_fft.pt",
    "swir": "checkpoints/swir_randinit_fft.pt",
}

CHECKPOINT_MAP_DINO = {
    "rgb": "checkpoints/rgb_dinoinit_fft.pt",
}


def _create_shot_sweep(starting_mod: str, new_mod: str, dino: bool = False,
                        dataset: str = 'eurosat', stage0_checkpoint: str = None,
                        script: str = 'shot') -> str:
    config = _load_merged_config(script)

    if stage0_checkpoint is None:
        cpt_map = CHECKPOINT_MAP_DINO if dino else CHECKPOINT_MAP
        if starting_mod not in cpt_map:
            raise ValueError(f"Unknown starting modality: {starting_mod}. Options: {list(cpt_map.keys())}")
        stage0_checkpoint = cpt_map[starting_mod]

    project_name = f"delulu-{script}-{dataset}-{starting_mod}-{new_mod}"
    if dino:
        project_name += "-dino"
    config['project'] = project_name

    config['command'].extend([
        '--dataset', dataset,
        '--stage0_checkpoint', stage0_checkpoint,
        '--new_mod_group', new_mod,
    ])

    sweep_id = wandb.sweep(sweep=config, project=project_name)
    full_sweep_id = f"{wandb.Api().default_entity}/{project_name}/{sweep_id}"
    print(f"\nSweep created! Add this line to sweep_registry.txt:")
    print(f"{project_name} {full_sweep_id} {stage0_checkpoint} {new_mod}")
    return sweep_id


# ---------------------------------------------------------------------------
# Shared helper: merge base.yaml + per-script override yaml
# ---------------------------------------------------------------------------

_SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_merged_config(script: str) -> dict:
    """Deep-merge base.yaml into sweep_{script}.yaml (script params win)."""
    with open(os.path.join(_SWEEP_DIR, "sweep_yaml", "base.yaml"), 'r') as f:
        base = yaml.safe_load(f)
    with open(os.path.join(_SWEEP_DIR, "sweep_yaml", f"sweep_{script}.yaml"), 'r') as f:
        override = yaml.safe_load(f)

    merged = override.copy()
    merged['parameters'] = {
        **base.get('parameters', {}),
        **override.get('parameters', {}),
    }
    return merged


# ---------------------------------------------------------------------------
# SFT sweep
# ---------------------------------------------------------------------------

def _create_sft_sweep(dataset: str, modalities: list[str], init: str = 'random') -> str:
    config = _load_merged_config('sft')
    modalities_str = '+'.join(modalities)
    project_name = config.get('project', 'evan-sweep-sft')
    config['project'] = project_name

    config['command'] = (config.get('command', [])
                         + ['--dataset', dataset]
                         + ['--modalities'] + modalities
                         + ['--init', init])

    sweep_id = wandb.sweep(sweep=config, project=project_name)
    full_sweep_id = f"{wandb.Api().default_entity}/{project_name}/{sweep_id}"
    print(f"\nSFT sweep created for dataset={dataset} modalities={modalities}!")
    print(f"Sweep ID: {full_sweep_id}")
    print(f"\nLaunch with:")
    print(f"  wandb agent {full_sweep_id}")
    return sweep_id


# ---------------------------------------------------------------------------
# Baseline distillation sweep
# ---------------------------------------------------------------------------

def _create_baseline_distill_sweep(dataset: str, teacher_checkpoint: str, modality: str) -> str:
    config = _load_merged_config('baseline_distill')
    project_name = config.get('project', 'evan-sweep-baseline-distill')
    config['project'] = project_name

    config['command'] = config.get('command', []) + [
        '--dataset', dataset,
        '--teacher_checkpoint', teacher_checkpoint,
        '--modality', modality,
    ]

    sweep_id = wandb.sweep(sweep=config, project=project_name)
    full_sweep_id = f"{wandb.Api().default_entity}/{project_name}/{sweep_id}"
    print(f"\nBaseline distill sweep created for dataset={dataset} {teacher_checkpoint} -> {modality}!")
    print(f"Sweep ID: {full_sweep_id}")
    print(f"\nLaunch with:")
    print(f"  wandb agent {full_sweep_id}")
    return sweep_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Create a W&B sweep.')
    parser.add_argument('--script', type=str, required=True,
                        choices=['shot', 'pldc', 'sft', 'baseline_distill'],
                        help='Which training script to sweep')

    # SHOT-specific
    parser.add_argument('--starting', type=str, help='[shot] Starting modality')
    parser.add_argument('--newmod', type=str, help='[shot] New modality to add')
    parser.add_argument('--dino', action='store_true', help='[shot] Use DINO checkpoint map (EuroSAT only)')
    parser.add_argument('--stage0_checkpoint', type=str,
                        help='[shot] Stage-0 checkpoint (required for non-EuroSAT datasets)')

    # Shared / SFT-specific
    parser.add_argument('--dataset', type=str,
                        choices=['eurosat', 'benv2', 'pastis', 'dfc2020'],
                        help='Dataset')
    parser.add_argument('--modalities', type=str, nargs='+',
                        help='[sft] Modalities to train on (first is primary)')
    parser.add_argument('--init', type=str, default='random',
                        choices=['random', 'dino', 's2dino'],
                        help='[sft] Weight initialization strategy (default: random)')

    # Baseline distill-specific
    parser.add_argument('--teacher_checkpoint', type=str,
                        help='[baseline_distill] Path to teacher checkpoint')
    parser.add_argument('--modality', type=str,
                        help='[baseline_distill] Student modality')

    args = parser.parse_args()

    if args.script == 'shot' or args.script == 'pldc':
        if not args.starting or not args.newmod:
            parser.error(f'--script {args.script} requires --starting and --newmod')
        if args.starting == args.newmod:
            parser.error('--starting and --newmod must be different')
        dataset = args.dataset or 'eurosat'
        _create_shot_sweep(args.starting, args.newmod, args.dino,
                           dataset=dataset, stage0_checkpoint=args.stage0_checkpoint,
                           script=args.script)

    elif args.script == 'sft':
        if not args.dataset or not args.modalities:
            parser.error('--script sft requires --dataset and --modalities')
        _create_sft_sweep(args.dataset, args.modalities, args.init)

    elif args.script == 'baseline_distill':
        if not args.dataset or not args.teacher_checkpoint or not args.modality:
            parser.error('--script baseline_distill requires --dataset, --teacher_checkpoint, --modality')
        _create_baseline_distill_sweep(args.dataset, args.teacher_checkpoint, args.modality)


if __name__ == '__main__':
    main()


# Examples
# --------
# SHOT (EuroSAT):
#   for m1 in rgb nir vre swir; do for m2 in rgb nir vre swir; do
#     [ "$m1" != "$m2" ] && python create_sweep.py --script shot --starting "$m1" --newmod "$m2"
#   done; done
#
# SFT:
#   python create_sweep.py --script sft --dataset benv2 --modalities s2_rgb
#   python create_sweep.py --script sft --dataset eurosat --modalities rgb
#
# Baseline distill:
#   python create_sweep.py --script baseline_distill --dataset benv2 \
#     --teacher_checkpoint checkpoints/benv2_s2_s0.pt --modality s2_rgb
