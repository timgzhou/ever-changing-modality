#!/bin/bash
# Can DINO init be made non-harmful under UPerNet on DFC2020?
#
# The gap: under upernet/split1, random init beats DINO by +3.5 to +4.7 mIoU on
# the multi-band modalities, and it does NOT close with training -- projecting
# the observed 24->64 epoch closing rate per cell gives 150-1300 epochs for
# s2/s2_norgb/s2+s1, and s2_rgb/s2_rgb+s1 are already flat or diverging. Only s1
# closes (2.84 -> 0.08 by 64 epochs). So more epochs is not the fix.
#
# The candidate mechanism is the discriminative lr in train_sft.py: in fft mode
# the backbone trains at lr/10 while the head trains at lr. Under a linear head
# that head is ~6K params, so nearly the whole model is discounted and a RANDOM
# backbone cannot catch up -- which is exactly where DINO wins (+5.5, 47/48
# pairs). Under upernet the head is 39M params at full lr against an 85M
# backbone at lr/10, so the decoder can adapt around a frozen-ish backbone and
# the DINO prior is never corrected. Raising the multiplier should help DINO
# specifically, and should NOT help random init (which is already the one being
# under-trained by the discount under linear, not upernet).
#
# Arms: backbone_lr_mult in {0.3, 1.0} x {dino, random} on the three modalities
# with the largest gap, at the best-known lr/wd/epochs. 0.1 is the historical
# default and is already measured, so it is not re-run.
#
# Rows land in res/train_sft/dfc2020_cobench_bblr.csv -- a SEPARATE file,
# because backbone_lr_mult is a new column and appending a 19-field row under
# the existing 18-field header silently shifts every field left by one.
#
# Usage:
#   bash sh/train_sft/train_sft_dfc2020_bblr.sh          # dry run
#   SUBMIT=1 bash sh/train_sft/train_sft_dfc2020_bblr.sh

set -u
SUBMIT="${SUBMIT:-0}"
DATASET=dfc2020
MODEL=evan_base
TRAIN_MODE=fft
LR="${LR:-0.0005}"
WD="${WD:-0.0}"
EPOCHS="${EPOCHS:-64}"
TRAIN_SPLIT="${TRAIN_SPLIT:-split1}"
DECODER=upernet
TRAIN_AUG=none

# Largest DINO deficits at 64ep (random - dino): s2_norgb+s1 +4.53,
# s2+s1 +4.09, s2_norgb +4.03, s2 +4.05. s2_rgb is the control -- DINO already
# WINS there (-1.28), so a real fix must not break it.
MODALITIES="${MODALITIES:-s2_norgb+s1 s2_norgb s2 s2_rgb}"
MULTS="${MULTS:-0.3 1.0}"
# train_sft_job.sh runs BOTH init arms per job via DINO_ARMS, so one job per
# (modality, mult) covers the dino-vs-random pair with identical everything else.
n=0
for MODALITY_ENTRY in ${MODALITIES}; do
  for MULT in ${MULTS}; do
    echo "  [$((++n))] ${MODALITY_ENTRY} bblr=${MULT} (dino+random) lr=${LR} wd=${WD} ep=${EPOCHS}"
    [ "${SUBMIT}" = "1" ] && \
      sbatch --export=ALL,DATASET="${DATASET}",MODEL="${MODEL}",TRAIN_MODE="${TRAIN_MODE}",MODALITY_ENTRY="${MODALITY_ENTRY}",LR="${LR}",WD="${WD}",TRAIN_AUG="${TRAIN_AUG}",TRAIN_SPLIT="${TRAIN_SPLIT}",DECODER="${DECODER}",EPOCHS="${EPOCHS}",BACKBONE_LR_MULT="${MULT}",DINO_ARMS="1 0" \
        sh/train_sft/train_sft_job.sh >/dev/null
  done
done
echo
echo "total: ${n} jobs x 2 init arms = $((n*2)) runs (SUBMIT=${SUBMIT})"
