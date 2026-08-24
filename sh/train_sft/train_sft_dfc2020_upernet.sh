#!/bin/bash
# UPerNet-decoder arm for DFC2020 on the Copernicus-Bench split.
#
# Why: Copernicus-Bench trains "a UPerNet decoder with an auxiliary FCN decoder"
# on top of the encoder for segmentation (arXiv:2503.11849, benchmark setup).
# Our default head is linear -- a 1x1 conv over the 16x16 token grid plus 16x
# bilinear upsample -- which cannot recover the fine spatial structure that
# segmentation depends on. The linear sweep landed at 49.6 (s1) / 58.1 (s2_norgb)
# test mIoU vs their 50.8 (S1) / 66.2 (full S2): S1 nearly matches, optical does
# not, which is what a missing multi-scale decoder predicts.
#
# Scope: all 6 modality entries at the config that won every val-selected
# comparison in the linear sweep (dino_init=True, train_split=full, lr=5e-4),
# both weight decays. 12 jobs, not a full 48-job re-sweep.
#
# Usage: bash sh/train_sft/train_sft_dfc2020_upernet.sh

export DFC2020_SPLIT=cobench
export DECODER=upernet

DATASET=dfc2020
MODEL=evan_base
TRAIN_MODE=fft
LR=0.0005
# split1 = train1 only. REQUIRED for any teacher used by stage-1 methods
# (shot_ete, distillation, MKE): they treat train2 as the UNLABELED pool, so a
# `full` teacher has already been supervised on that pool and the comparison
# leaks. Override with TRAIN_SPLIT=full for the supervised upper bound.
TRAIN_SPLIT="${TRAIN_SPLIT:-split1}"
TRAIN_AUG=none

# Override from the shell, e.g. to re-run only the combined arms:
#   MODALITIES='s2_rgb+s1 s2_norgb+s1 s2_rgb+s2_norgb' bash <this script>
MODALITIES="${MODALITIES:-s2_rgb s2_norgb s1 s2_rgb+s1 s2_norgb+s1 s2_rgb+s2_norgb}"
WDS='0.01 0.0'

n=0
for MODALITY_ENTRY in ${MODALITIES}; do
    for WD in ${WDS}; do
        echo "Submitting: upernet ${MODALITY_ENTRY} lr=${LR} wd=${WD} split=${TRAIN_SPLIT}"
        sbatch --export=ALL,DATASET="${DATASET}",MODEL="${MODEL}",TRAIN_MODE="${TRAIN_MODE}",MODALITY_ENTRY="${MODALITY_ENTRY}",LR="${LR}",WD="${WD}",TRAIN_AUG="${TRAIN_AUG}",TRAIN_SPLIT="${TRAIN_SPLIT}",DFC2020_SPLIT="${DFC2020_SPLIT}",DECODER="${DECODER}" \
            sh/train_sft/train_sft_job.sh
        n=$((n+1))
    done
done
echo "submitted ${n} upernet jobs (each runs both dino variants)"
