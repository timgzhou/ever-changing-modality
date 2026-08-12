#!/bin/bash
#SBATCH --job-name=sweep-register
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/sweep/register_%j.out
#SBATCH --account=aip-gpleiss
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL

# Register the BioMassters sweeps AND run one smoke-test trial, all inside one
# SLURM job so registration and the agent share the same (working) wandb env.
# The login node cannot import wandb cleanly, so registration must happen here.
#
# Usage:  sbatch sweep/register_and_smoketest.sh
#
# After it completes, read logs/sweep/register_<jobid>.out for the sweep IDs it
# printed, then submit the rest of the agents:
#   for i in $(seq 1 63); do sbatch sweep/run_sweep.sh '<s2s1-id>'; done
#   for i in $(seq 1 64); do sbatch sweep/run_sweep.sh '<s1s2-id>'; done

REPO_ROOT="$HOME/scratch/ever-changing-modalities"
cd "$REPO_ROOT"
source sh/env.sh
export TQDM_DISABLE=1
export WANDB_DIR="$HOME/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WANDB_DIR" logs/sweep

echo "=== registering biomassters sweeps ==="
# Capture stdout so we can grep the sweep IDs it prints.
python sweep/create_sweep_biomassters.py | tee /tmp/bm_sweeps_$$.txt

# Grab the first registered sweep id from the "Sweep registered:" line.
# (Do NOT match the "Sweep URL:" line — its path contains a literal '/sweeps/'
# segment before the real id, which a loose regex grabs by mistake.)
SMOKE_ID=$(grep -oP "Sweep registered: \Ktgz/\S+" /tmp/bm_sweeps_$$.txt | head -1)

echo ""
echo "=== smoke-testing ONE trial on: ${SMOKE_ID} ==="
if [ -z "${SMOKE_ID}" ]; then
    echo "[error] could not parse a sweep id from registration output"
    exit 1
fi

wandb agent --count 1 "${SMOKE_ID}"
echo "=== smoke test done ==="
