#!/usr/bin/env python
"""
Create a wandb sweep for a specific modality combination.

Usage:
    python create_sweep.py --starting nir --newmod rgb

This loads the shared sweep_config.yaml, adds the modality-specific args,
creates the sweep, and prints a line to add to sweep_registry.txt
"""

import argparse
import yaml
import wandb

BASE_CONFIG_PATH = "sweep_yaml/sweep_config.yaml"

# Map modality names to checkpoint paths
CHECKPOINT_MAP = {
    "nir": "checkpoints/nir_randinit_fft.pt",
    "rgb": "checkpoints/rgb_randinit_fft.pt",
    "vre": "checkpoints/vre_randinit_fft.pt",
    "swir": "checkpoints/swir_randinit_fft.pt",
}


def create_sweep(starting_mod: str, new_mod: str) -> str:
    # Load base config
    with open(BASE_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Get checkpoint path
    if starting_mod not in CHECKPOINT_MAP:
        raise ValueError(f"Unknown starting modality: {starting_mod}. Options: {list(CHECKPOINT_MAP.keys())}")
    stage0_checkpoint = CHECKPOINT_MAP[starting_mod]

    # Set project name based on modality combo
    project_name = f"delulu-sweep-{starting_mod}-{new_mod}"
    config['project'] = project_name

    # Add modality args to command
    config['command'].extend([
        '--stage0_checkpoint', stage0_checkpoint,
        '--new_mod_group', new_mod,
    ])

    # Create the sweep
    sweep_id = wandb.sweep(sweep=config, project=project_name)

    # Print registry line
    full_sweep_id = f"{wandb.Api().default_entity}/{project_name}/{sweep_id}"
    print(f"\nSweep created! Add this line to sweep_registry.txt:")
    print(f"{project_name} {full_sweep_id} {stage0_checkpoint} {new_mod}")

    return sweep_id


def main():
    parser = argparse.ArgumentParser(description='Create a sweep for a modality combination')
    parser.add_argument('--starting', type=str, required=True,
                        choices=['nir', 'rgb', 'vre', 'swir'],
                        help='Starting modality')
    parser.add_argument('--newmod', type=str, required=True,
                        choices=['nir', 'rgb', 'vre', 'swir'],
                        help='New modality to add')
    args = parser.parse_args()

    if args.starting == args.newmod:
        print("Error: starting and newmod must be different")
        return

    create_sweep(args.starting, args.newmod)


if __name__ == '__main__':
    main()


# for m1 in rgb nir vre swir; do for m2 in rgb nir vre swir; do if [ "$m1" != "$m2" ]; then python create_sweep.py --starting "$m1" --newmod "$m2"; fi; done; done