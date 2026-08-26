#!/bin/bash
# Distillation baseline for the MODALITY TRANSFER setting on DFC2020/cobench.
#
# This is the correct analogue of DeluluNet's `transfer` metric:
#   teacher : UNIMODAL, trained on split1 only (train1)
#   student : UNIMODAL, on the NEW modality only
#   data    : train2 only, unlabeled (alpha=1.0 -> pure distillation,
#             the student never sees train2's ground-truth labels)
#   test    : NEW modality only
#
# The earlier distillation runs (res/baselines/dfc2020_cobench_distillation_*)
# passed `--modalities <teacher> <new>`, producing a BIMODAL student evaluated
# on both modalities. That is the ADDITION setting, not transfer, which is why
# those numbers were not comparable to delulu's transfer column.
#
# Usage:
#   bash sh/distill_transfer_dfc2020.sh            # dry run
#   SUBMIT=1 bash sh/distill_transfer_dfc2020.sh

set -u
DECODER="${DECODER:-upernet}"
MODEL="${MODEL:-evan_base}"
TEACHER_SPLIT="${TEACHER_SPLIT:-split1}"
EPOCHS="${EPOCHS:-64}"
LRS="${LRS:-0.0005 0.0001}"
# Two KD variants, reported as separate baselines (cf. res/results_table.py):
#   kd  — standard KD: both logits temperature-scaled, loss x T^2
#   ttm — transformed teacher matching: teacher scaled, student NOT, no T^2
KL_TYPES="${KL_TYPES:-kd ttm}"
SUBMIT="${SUBMIT:-0}"
# INIT_FROM_TEACHER=1 initialises the student from the teacher's backbone instead
# of random init. Until now the teacher-init arm silently built a LINEAR head
# (init_student_from_teacher did not forward --decoder_type), so "random beats
# teacher init" was confounded with "upernet beats linear". Now matched.
INIT_FROM_TEACHER="${INIT_FROM_TEACHER:-0}"
TEACHERS_JSON="artifacts/sft_teachers.json"

# teacher:new — the same six directions as the delulu runs
PAIRS="s1:s2_rgb s1:s2_norgb s2_rgb:s1 s2_rgb:s2_norgb s2_norgb:s1 s2_norgb:s2_rgb"

n=0
for P in ${PAIRS}; do
    START="${P%%:*}"; NEW="${P##*:}"
    TEACHER=$(jq -r ".\"dfc2020_cobench/${START}/${MODEL}/${DECODER}/${TEACHER_SPLIT}\".checkpoint // empty" "${TEACHERS_JSON}")
    if [ -z "${TEACHER}" ] || [ ! -f "${TEACHER}" ]; then
        echo "  [skip] no ${DECODER}/${TEACHER_SPLIT} teacher for ${START}"; continue
    fi
    for LR in ${LRS}; do
      for KL in ${KL_TYPES}; do
        TAG="distill_transfer_${KL}_${START}_to_${NEW}_lr${LR}"
        [ "$INIT_FROM_TEACHER" = "1" ] && TAG="${TAG}_initteacher"
        ARGS="baseline/baseline_distillation.py --dataset dfc2020 --modalities ${NEW}"
        ARGS="${ARGS} --teacher_checkpoint ${TEACHER} --decoder_type ${DECODER} --model ${MODEL}"
        ARGS="${ARGS} --epochs ${EPOCHS} --lr ${LR} --kl_type ${KL}"
        ARGS="${ARGS} --results_csv res/baselines/dfc2020_cobench_distill_transfer_${DECODER}.csv"
        [ "$INIT_FROM_TEACHER" = "1" ] && ARGS="${ARGS} --init_from_teacher"
        if [ "$SUBMIT" = "1" ]; then
            sbatch --export=ALL,BASELINE_ARGS="${ARGS}",RUN_TAG="${TAG}",DECODER="${DECODER}" \
                sh/baselines_dfc2020_job.sh >/dev/null
        fi
        n=$((n+1))
        echo "  [$n] ${TAG}   (teacher=${START} split1 -> student=${NEW} unimodal, kl=${KL})"
      done
    done
done
echo
echo "total: ${n} jobs (SUBMIT=${SUBMIT})"
