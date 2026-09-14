#!/bin/bash
# Stage-0 (train_sft) unimodal launch: all 4 datasets, lr 1e-4, wd 0.01, split1.
# Each submitted job internally runs both dino-init and random-init.
# DRYRUN=1 prints without submitting.
set -u
# Unimodal stage-0 teachers, one per (dataset, modality).
#   eurosat: no 's2' -- the dataset only ships S2, so 's2' is the whole thing
#            rather than a distinct modality to add.
#   benv2 / dfc2020: the narrow S2 sub-groups (s2_vre/s2_nir/s2_swir/s2_aw) are
#            deliberately excluded; only the full sensors and the rgb/norgb split.
#   biomassters: s2_norgb is the 7-band complement of s2_rgb within the 10-band
#            S2 stack (added 2026-09-13).
declare -A UNI=(
  [eurosat]='rgb vre nir swir aw'
  [benv2]='s2 s1 s2_rgb s2_norgb'
  [dfc2020]='s2 s1 s2_rgb s2_norgb'
  [biomassters]='s2 s1 s2_rgb s2_norgb'
)
MODEL=evan_base; TRAIN_MODE=fft; LR=0.0001; WD=0.01; TRAIN_SPLIT=split1; TRAIN_AUG=none
n=0
for DATASET in eurosat benv2 dfc2020 biomassters; do
  EXPORTS="ALL,DATASET=${DATASET},MODEL=${MODEL},TRAIN_MODE=${TRAIN_MODE},LR=${LR},WD=${WD},TRAIN_SPLIT=${TRAIN_SPLIT},TRAIN_AUG=${TRAIN_AUG}"
  # DFC2020 needs the UPerNet decoder to match the Copernicus-Bench setup.
  [ "${DATASET}" = "dfc2020" ] && EXPORTS="${EXPORTS},DECODER=upernet"
  for M in ${UNI[$DATASET]}; do
    n=$((n+1))
    if [ "${DRYRUN:-0}" = "1" ]; then
      echo "[$n] sbatch --export=${EXPORTS},MODALITY_ENTRY=${M} sh/train_sft/train_sft_job.sh"
    else
      jid=$(sbatch --parsable --export="${EXPORTS},MODALITY_ENTRY=${M}" sh/train_sft/train_sft_job.sh)
      echo "[$n] submitted ${jid}  ${DATASET}/${M}"
    fi
  done
done
echo "total: $n jobs"
