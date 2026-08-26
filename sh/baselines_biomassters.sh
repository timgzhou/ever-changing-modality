#!/bin/bash
# All BioMassters baselines (T=12 temporal AGB regression, metric = RMSE, LOWER IS BETTER).
#
# Families
#   distillation  teacher-based, multimodal student  (kd / ttm variants)
#   mke           teacher-based, multimodal student  (multimodal knowledge expansion)
#   mixmatch      semi-supervised, single modality   (no teacher)
#
# FreeMatch is deliberately ABSENT. Its mechanism is self-adaptive thresholding on
# max(softmax) plus a class histogram -- both are K-way classification constructs
# with no regression analogue (arXiv 2205.07246 never mentions regression). It was
# never ported to biomassters and should not be forced.
#
# Teachers: best-by-val (lowest RMSE) split1 checkpoints from res/train_sft/biomassters.csv.
# split1 matters -- stage 1 uses train2 as its unlabeled pool, so a `full` teacher
# has already seen that pool with labels and the comparison leaks.
#
# The decoder MUST match the teacher: these teachers are upernet+relu.
#   s2 upernet+relu  val 42.75 / test 43.59
#   s1 upernet+relu  val 46.58 / test 47.68
#
# Usage:
#   bash sh/baselines_biomassters.sh            # dry run, prints the plan
#   SUBMIT=1 bash sh/baselines_biomassters.sh   # actually sbatch

set -u
MODEL="${MODEL:-evan_base}"
EPOCHS="${EPOCHS:-64}"
LRS="${LRS:-0.0005 0.0001}"
KL_TYPES="${KL_TYPES:-kd ttm}"
BATCH_SIZE="${BATCH_SIZE:-8}"          # T=12 folds time into batch; 8 is the safe point
SUBMIT="${SUBMIT:-0}"

# best-by-val split1 upernet+relu teachers
TEACHER_s2="checkpoints/sft_evan_base_biomassters_s2_fft_lr0.0005_20260725_075836.pt"
TEACHER_s1="checkpoints/sft_evan_base_biomassters_s1_fft_lr0.0005_20260725_075947.pt"

COMMON="--decoder_type upernet --relu_output --model ${MODEL} --num_time_steps 12"
COMMON="${COMMON} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS}"

n=0
submit () {  # $1=tag  $2...=args
    local tag="$1"; shift
    if [ "$SUBMIT" = "1" ]; then
        sbatch --export=ALL,BASELINE_ARGS="$*",RUN_TAG="${tag}" \
            sh/baselines_biomassters_job.sh >/dev/null
    fi
    echo "  [$((++n))] ${tag}"
}

echo "=== teacher-based: distillation + mke (2 directions) ==="
for P in s2:s1 s1:s2; do
    START="${P%%:*}"; NEW="${P##*:}"
    eval "TEACHER=\$TEACHER_${START}"
    if [ ! -f "${TEACHER}" ]; then
        echo "  [error] teacher not found: ${TEACHER}"; exit 1
    fi
    for LR in ${LRS}; do
        for KL in ${KL_TYPES}; do
            submit "bm_distill_${KL}_${START}_to_${NEW}_lr${LR}" \
                baseline/baseline_distillation.py --dataset biomassters \
                --modalities "${START}" "${NEW}" --teacher_checkpoint "${TEACHER}" \
                ${COMMON} --lr "${LR}" --kl_type "${KL}" \
                --results_csv res/baselines/biomassters_distillation_upernet.csv
        done
        submit "bm_mke_${START}_to_${NEW}_lr${LR}" \
            baseline/baseline_mke.py --dataset biomassters \
            --modalities "${START}" "${NEW}" --teacher_checkpoint "${TEACHER}" \
            ${COMMON} --lr "${LR}" \
            --results_csv res/baselines/biomassters_mke_upernet.csv
    done
done

echo
echo "=== semi-supervised: mixmatch (no teacher, single modality) ==="
# lambda_u=75 from the paper assumes an L2/Brier consistency loss; this code uses
# a plain MSE/CE-scale term, so the paper value over-weights the unlabeled branch
# by ~2.4x. Swept low, as on dfc2020 (where 75 collapsed to 5.3 mIoU vs 59.4 at 0.5).
LAMBDA_US="${LAMBDA_US:-0.5 1.0}"
for MOD in s2 s1; do
    for LR in ${LRS}; do
        for LU in ${LAMBDA_US}; do
            submit "bm_mixmatch_${MOD}_lr${LR}_lu${LU}" \
                baseline/baseline_mixmatch.py --dataset biomassters \
                --modality "${MOD}" --use_dino_weights \
                ${COMMON} --lr "${LR}" --lambda_u "${LU}" \
                --results_csv res/baselines/biomassters_mixmatch_upernet.csv
        done
    done
done

echo
echo "total: ${n} jobs (SUBMIT=${SUBMIT})"
