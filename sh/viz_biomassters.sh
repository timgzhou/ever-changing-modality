#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/viz_biomassters/%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Visualize BioMassters samples (per-timestep, per-modality bands + AGB target)
# for one train / val / test sample. CPU-only.
#
# Usage:  sbatch sh/viz_biomassters.sh
#         NUM_TIME_STEPS=4 INDEX=3 sbatch sh/viz_biomassters.sh

source sh/env.sh
mkdir -p logs/viz_biomassters figs

python -u viz_biomassters.py \
    --num_time_steps "${NUM_TIME_STEPS:-12}" \
    --index "${INDEX:-0}" \
    --out figs/biomassters
