#!/bin/bash
# Augmentation-parity ablation: re-run the FreeMatch sweep with the strong branch
# degraded to weak-equivalent (flips only).
#
# Motivation: FreeMatch's unsupervised signal is "weak-view prediction survives
# strong corruption", whereas SHOT uses no augmentation at all. This ablation
# measures how much of FreeMatch's benefit comes from augmentation strength
# rather than from self-adaptive thresholding, which is the control needed to
# compare the two families fairly.
#
# Rows land in the same CSV as the main sweep, distinguished by the
# no_strong_aug column (True here, False for the main sweep). The resume guard
# keys on that column, so the two arms never skip each other's rows.
#
# Usage: bash sh/freematch/freematch_ablation_aug.sh
#        bash sh/freematch/freematch_ablation_aug.sh eurosat

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
        echo "[submit] ${DATASET} | modality=${MODALITY} | no_strong_aug"
        sbatch sh/freematch/freematch_job.sh "$DATASET" "$MODALITY" no_strong_aug
    done
done

# bash sh/freematch/freematch_ablation_aug.sh
