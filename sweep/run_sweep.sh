#!/bin/bash
#SBATCH --job-name=shot-sweep
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/sweep/%j.out
#SBATCH --account=aip-gpleiss
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL

# One agent per job (--count 1), submit N times for N parallel jobs:
#   for i in $(seq 1 128); do sbatch sweep/run_sweep.sh <sweep-id>; done

SWEEP_ID="${1:?Usage: sbatch sweep/run_sweep.sh <entity/project/sweep_id>}"

REPO_ROOT="$HOME/scratch/ever-changing-modalities"
cd "$REPO_ROOT"
source sh/env.sh
export TQDM_DISABLE=1
export WANDB_DIR="$HOME/wandb"
mkdir -p "$WANDB_DIR"

echo "Starting sweep agent: $SWEEP_ID"
wandb agent --count 1 "$SWEEP_ID"
echo "Agent completed."
