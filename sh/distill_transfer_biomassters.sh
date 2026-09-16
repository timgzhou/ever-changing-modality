#!/bin/bash
# Distillation (KD / TTM) baseline for the MODALITY TRANSFER setting on BioMassters.
# Counterpart of sh/distill_transfer_dfc2020.sh; see that file for the rationale.
#
#   teacher : UNIMODAL, split1 only
#   student : UNIMODAL, on the NEW modality only  (--modalities <NEW>, one arg)
#   data    : train2, unlabeled (pure distillation)
#   test    : NEW modality only
#
# The existing res/baselines/biomassters_distillation_upernet.csv rows are NOT
# this: they pass `--modalities <teacher> <new>`, giving a BIMODAL student. That
# is the ADDITION setting. The transfer column of the paper table has no
# biomassters numbers at all, which is what this script fills.
#
# METRIC IS RMSE — LOWER IS BETTER. Teachers are best-by-LOWEST val RMSE.
#
# WALLTIME: past biomassters baselines ran 64 epochs in 9:08-9:56 against an
# 11:59 limit, and distillation adds a teacher forward pass on top of that. With
# 46 TIMEOUTs already in this account's history, 64 epochs is not worth the risk:
# this uses 48 epochs (matching the stage-0 rerun) AND requests 23:59.
#
# Usage:
#   bash sh/distill_transfer_biomassters.sh            # dry run
#   SUBMIT=1 bash sh/distill_transfer_biomassters.sh

set -u
MODEL="${MODEL:-evan_base}"
EPOCHS="${EPOCHS:-48}"
LRS="${LRS:-0.0005 0.0001}"
KL_TYPES="${KL_TYPES:-kd ttm}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WALLTIME="${WALLTIME:-23:59:00}"
SUBMIT="${SUBMIT:-0}"
RESULTS_CSV="${RESULTS_CSV:-res/baselines/biomassters_distill_transfer_upernet.csv}"

# split1 upernet+relu teachers, best-by-val (LOWEST RMSE).
#   s2       val 42.75 / test 43.59
#   s1       val 46.58 / test 47.68
#   s2_rgb   val 41.01 / test 41.96   (2026-09-15 stage-0 rerun, lr 5e-4)
#   s2_norgb val 41.01 / test 41.88   (same)
TEACHER_s2="checkpoints/sft_evan_base_biomassters_s2_fft_lr0.0005_20260725_075836.pt"
TEACHER_s1="checkpoints/sft_evan_base_biomassters_s1_fft_lr0.0005_20260725_075947.pt"
TEACHER_s2_rgb="checkpoints/sft_evan_base_biomassters_s2_rgb_fft_lr0.0005_20260915_005742.pt"
TEACHER_s2_norgb="checkpoints/sft_evan_base_biomassters_s2_norgb_fft_lr0.0005_20260915_011435.pt"

# The four pairs the paper table reports.
PAIRS="${PAIRS:-s2_rgb:s1 s2_rgb:s2_norgb s1:s2 s2:s1}"

n=0
for P in ${PAIRS}; do
    START="${P%%:*}"; NEW="${P##*:}"
    eval "TEACHER=\${TEACHER_${START}:-}"
    if [ -z "${TEACHER}" ] || [ ! -f "${TEACHER}" ]; then
        echo "  [skip] no teacher for ${START}"; continue
    fi
    for LR in ${LRS}; do
      for KL in ${KL_TYPES}; do
        TAG="bm_distill_transfer_${KL}_${START}_to_${NEW}_lr${LR}"
        ARGS="baseline/baseline_distillation.py --dataset biomassters --modalities ${NEW}"
        ARGS="${ARGS} --teacher_checkpoint ${TEACHER} --decoder_type upernet --relu_output"
        ARGS="${ARGS} --model ${MODEL} --num_time_steps 12 --batch_size ${BATCH_SIZE}"
        ARGS="${ARGS} --epochs ${EPOCHS} --lr ${LR} --kl_type ${KL}"
        ARGS="${ARGS} --results_csv ${RESULTS_CSV}"
        if [ "$SUBMIT" = "1" ]; then
            sbatch --time="${WALLTIME}" \
                --export=ALL,BASELINE_ARGS="${ARGS}",RUN_TAG="${TAG}" \
                sh/baselines_biomassters_job.sh >/dev/null
        fi
        n=$((n+1))
        echo "  [$n] ${TAG}   (teacher=${START} split1 -> student=${NEW} unimodal, kl=${KL})"
      done
    done
done
echo
echo "total: ${n} jobs, ${EPOCHS} epochs, --time=${WALLTIME} (SUBMIT=${SUBMIT})"
echo "  -> ${RESULTS_CSV}"
