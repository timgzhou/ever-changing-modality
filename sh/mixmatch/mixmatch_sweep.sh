#!/bin/bash
# Launcher: one sbatch job per dataset+modality combo.
#
# Usage: bash sh/mixmatch/mixmatch_sweep.sh

declare -A MODALITY_CONFIGS
MODALITY_CONFIGS['eurosat']='rgb vre nir swir'
MODALITY_CONFIGS['benv2']='s2 s1 s2_rgb'
MODALITY_CONFIGS['dfc2020']='s2 s1 s2_rgb'

DATASETS=('eurosat' 'benv2' 'dfc2020')

mkdir -p logs/mixmatch

for DATASET in "${DATASETS[@]}"; do
    for MODALITY in ${MODALITY_CONFIGS[$DATASET]}; do
        echo "[submit] ${DATASET} | modality=${MODALITY}"
        sbatch sh/mixmatch/mixmatch_job.sh "$DATASET" "$MODALITY"
    done
done

# bash sh/mixmatch/mixmatch_sweep.sh