#!/bin/bash
# Launcher: submits one SLURM job per model+dataset+modality combo.
# Usage: bash sh/rsfm_sft/rsfm_sft_all.sh

declare -A MODALITY_CONFIGS
# Single modalities
# MODALITY_CONFIGS['eurosat']='s2 rgb vre nir swir'
# MODALITY_CONFIGS['benv2']='s2 s1 s2s1 s2_rgb'
# MODALITY_CONFIGS['dfc2020']='s2 s1 s2s1 s2_rgb'

# Combined modalities (oracle upper-bound for Addition table)
# MODALITY_CONFIGS['eurosat']='rgb+nir rgb+vre swir+nir swir+rgb swir+vre vre+nir vre+rgb'
# MODALITY_CONFIGS['benv2']='s2_rgb+s1 s2_rgb+s2_norgb s1+s2 s2+s1 s2_norgb'
# MODALITY_CONFIGS['dfc2020']='s2_rgb+s1 s2_rgb+s2_norgb s1+s2 s2+s1 s2_norgb'

MODALITY_CONFIGS['eurosat']='rgb+nir rgb+vre rgb+swir'
MODALITY_CONFIGS['benv2']='s2_norgb'
MODALITY_CONFIGS['dfc2020']='s2_norgb'

MODELS=('panopticon' 'olmoearth-base')
DATASETS=('benv2' 'dfc2020')
TRAIN_MODES=('fft')

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for TRAIN_MODE in "${TRAIN_MODES[@]}"; do
            for MODALITY in ${MODALITY_CONFIGS[$DATASET]}; do
                echo "Submitting: model=${MODEL} dataset=${DATASET} train_mode=${TRAIN_MODE} modality=${MODALITY}"
                sbatch --export=ALL,MODEL="${MODEL}",DATASET="${DATASET}",TRAIN_MODE="${TRAIN_MODE}",MODALITY="${MODALITY}" \
                    sh/rsfm_sft/rsfm_sft_job.sh
            done
        done
    done
done

# bash sh/rsfm_sft/rsfm_sft_all.sh