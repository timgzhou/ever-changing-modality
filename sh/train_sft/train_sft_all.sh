#!/bin/bash
# Launcher: submits one SLURM job per dataset+model+modality+lr+wd combo.
# Usage: bash sh/train_sft/train_sft_all.sh

declare -A MODALITY_CONFIGS
# Single modalities (stage 0 oracle)
# MODALITY_CONFIGS['eurosat']='s2 rgb vre nir swir'
# MODALITY_CONFIGS['benv2']='s2 s1 s2_rgb s2_norgb'
# MODALITY_CONFIGS['dfc2020']='s2 s1 s2_rgb s2_norgb'

# Combined modalities: use '+' separator — job script splits into --modalities args
# (oracle upper-bound for Addition table: DINO-init SFT on start+new combined)
# MODALITY_CONFIGS['eurosat']='rgb+nir rgb+vre rgb+swir swir+nir swir+rgb swir+vre vre+nir vre+rgb'
# MODALITY_CONFIGS['benv2']='s2_norgb s2_rgb+s1 s2_rgb+s2_norgb s1+s2 s2+s1'
# MODALITY_CONFIGS['dfc2020']='s2_norgb s2_rgb+s1 s2_rgb+s2_norgb s1+s2 s2+s1'

MODALITY_CONFIGS['eurosat']='rgb+nir rgb+vre rgb+swir'
MODALITY_CONFIGS['benv2']='s2_norgb s2'
MODALITY_CONFIGS['dfc2020']='s2_norgb'

# DATASETS=('eurosat' 'benv2' 'dfc2020')
# MODELS=('evan_small' 'evan_large' 'evan_base' )
DATASETS=('benv2' 'dfc2020')
MODELS=('evan_base')
TRAIN_MODES=('fft')
LRS=('0.0005' '0.0001')
WDS=('0.01' '0.0')

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for TRAIN_MODE in "${TRAIN_MODES[@]}"; do
            for MODALITY_ENTRY in ${MODALITY_CONFIGS[$DATASET]}; do
                for LR in "${LRS[@]}"; do
                    for WD in "${WDS[@]}"; do
                        MODALITY_KEY="${MODALITY_ENTRY}"
                        RESULTS_CSV="res/train_sft/${DATASET}.csv"
                        DINO_PAT="^${DATASET},${MODEL},${MODALITY_KEY},${TRAIN_MODE},[^,]+,[^,]+,${LR},${WD},[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+"
                        grep -qP "${DINO_PAT},True" "${RESULTS_CSV}" 2>/dev/null && DINO_TRUE=1 || DINO_TRUE=0
                        grep -qP "${DINO_PAT},False" "${RESULTS_CSV}" 2>/dev/null && DINO_FALSE=1 || DINO_FALSE=0
                        if [ "$DINO_TRUE" -ge 1 ] && [ "$DINO_FALSE" -ge 1 ]; then
                            echo "Skipping (both dino variants done): dataset=${DATASET} model=${MODEL} train_mode=${TRAIN_MODE} modality=${MODALITY_ENTRY} lr=${LR} wd=${WD}"
                            continue
                        fi
                        echo "Submitting: dataset=${DATASET} model=${MODEL} train_mode=${TRAIN_MODE} modality=${MODALITY_ENTRY} lr=${LR} wd=${WD}"
                        sbatch --export=ALL,DATASET="${DATASET}",MODEL="${MODEL}",TRAIN_MODE="${TRAIN_MODE}",MODALITY_ENTRY="${MODALITY_ENTRY}",LR="${LR}",WD="${WD}" \
                            sh/train_sft/train_sft_job.sh
                    done
                done
            done
        done
    done
done


# bash sh/train_sft/train_sft_all.sh
