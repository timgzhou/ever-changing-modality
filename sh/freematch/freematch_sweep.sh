#!/bin/bash
# Launcher: one sbatch job per dataset+modality combo.
#
# Usage: bash sh/freematch/freematch_sweep.sh
#        bash sh/freematch/freematch_sweep.sh eurosat        # single dataset

declare -A MODALITY_CONFIGS
MODALITY_CONFIGS['eurosat']='rgb vre nir swir'
MODALITY_CONFIGS['benv2']='s2 s1 s2_rgb'

if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
else
    DATASETS=('eurosat' 'benv2')
fi

mkdir -p logs/freematch

for DATASET in "${DATASETS[@]}"; do
    for MODALITY in ${MODALITY_CONFIGS[$DATASET]}; do
        echo "[submit] ${DATASET} | modality=${MODALITY}"
        sbatch sh/freematch/freematch_job.sh "$DATASET" "$MODALITY"
    done
done

# bash sh/freematch/freematch_sweep.sh
