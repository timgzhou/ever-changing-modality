#!/usr/bin/env python
"""
Launch sbatch jobs for all sweeps in sweep_registry.txt

Usage:
    python run_all_sweeps.py           # Submit all sweeps
    python run_all_sweeps.py --dry-run # Show commands without running
"""

import argparse
import subprocess
import sys

REGISTRY_PATH = "sweep_registry.txt"
SBATCH_SCRIPT = "sh/run_sweep.sh"


def parse_registry():
    """Parse sweep_registry.txt and return list of (project, sweep_id, checkpoint, newmod)"""
    sweeps = []
    with open(REGISTRY_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                project_name = parts[0]
                sweep_id = parts[1]
                # Optional: checkpoint and newmod for reference
                checkpoint = parts[2] if len(parts) > 2 else ""
                newmod = parts[3] if len(parts) > 3 else ""
                sweeps.append((project_name, sweep_id, checkpoint, newmod))
    return sweeps


def run_sweeps(dry_run: bool = False):
    sweeps = parse_registry()

    if not sweeps:
        print("No sweeps found in sweep_registry.txt")
        print("Create sweeps first with: python create_sweep.py --starting <mod> --newmod <mod>")
        return

    print(f"Found {len(sweeps)} sweep(s) in registry:\n")

    for project, sweep_id, checkpoint, newmod in sweeps:
        cmd = f"sbatch {SBATCH_SCRIPT} {sweep_id}"
        print(f"  {project}: {cmd}")

        if not dry_run:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"    -> {result.stdout.strip()}")
            else:
                print(f"    -> Error: {result.stderr.strip()}")

    if dry_run:
        print("\n(Dry run - no jobs submitted. Remove --dry-run to submit)")


def main():
    parser = argparse.ArgumentParser(description='Run all sweeps from registry')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show commands without running')
    args = parser.parse_args()

    run_sweeps(dry_run=args.dry_run)


if __name__ == '__main__':
    main()


# python run_all_sweeps.py 