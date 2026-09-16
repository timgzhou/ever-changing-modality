#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/viz_biomassters/%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# kn101 has failing GPU memory: every job scheduled there since 2026-09-15 died
# with "CUDA error: uncorrectable ECC error encountered", while jobs on every
# other node succeeded. Slurm still lists it as healthy. Remove if repaired.
#SBATCH --exclude=kn101
# Visualize BioMassters samples (per-timestep, per-modality bands + AGB target)
# for one train / val / test sample. CPU-only.
#
# Usage:  sbatch sh/viz_biomassters.sh
#         NUM_TIME_STEPS=4 INDEX=3 sbatch sh/viz_biomassters.sh

source sh/env.sh
mkdir -p logs/viz_biomassters figs

python -u viz/viz_biomassters.py \
    --num_time_steps "${NUM_TIME_STEPS:-12}" \
    --index "${INDEX:-0}" \
    --out figs/biomassters
