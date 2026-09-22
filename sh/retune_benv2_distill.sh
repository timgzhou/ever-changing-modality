#!/bin/bash
# Re-tune the reBEN KD/TTM baselines under the FIXED multilabel distillation loss.
#
# WHY
# ---
# Until 2026-09-20 the multilabel branch of train_utils.distillation_loss()
# ignored kl_type entirely: it always scaled BOTH teacher and student logits by
# T, so 'kd' and 'ttm' were bit-identical on reBEN and the tables carried two
# baseline rows that were the same computation. The fix gives TTM its real
# definition (soften the TEACHER only) and brings KD in line with the softmax
# arm's Hinton T**2 gradient rescale.
#
# That T**2 is NOT a no-op: with alpha=1.0 and T=2.0 the distillation term is
# now 4x heavier against the CE term than the config was originally selected
# under. Re-selecting the config is therefore the honest move -- otherwise the
# reported reBEN KD row is a config tuned for one loss scale, evaluated at
# another.
#
# WHAT IT SWEEPS
# --------------
# The same grid the original reBEN distillation sweep used, per (teacher,
# student, kl_type): lr x temperature. Temperature matters most here -- it is
# exactly the knob the T**2 rescale interacts with. init_from_teacher is pinned
# True to match Delulu's student_init (see _flat_frames in res/results_BL.py).
#
# Results land in their own CSV so the pre-fix rows stay readable for comparison.
#
# Usage:
#   bash sh/retune_benv2_distill.sh            # dry run
#   SUBMIT=1 bash sh/retune_benv2_distill.sh   # actually sbatch

set -u
MODEL="${MODEL:-evan_base}"
EPOCHS="${EPOCHS:-20}"
LRS="${LRS:-0.0003 0.001}"
TEMPS="${TEMPS:-0.5 1.0 2.0}"
SUBMIT="${SUBMIT:-0}"
WALLTIME="${WALLTIME:-6:00:00}"
TEACHERS_JSON="artifacts/sft_teachers.json"
RESULTS_CSV="${RESULTS_CSV:-res/baselines/distillation_benv2_transfer_retuned.csv}"
PAIRS="${PAIRS:-s2_rgb:s1 s2_rgb:s2_norgb s1:s2 s2:s1}"

n=0
for P in ${PAIRS}; do
    START="${P%%:*}"; NEW="${P##*:}"
    TEACHER=$(jq -r ".\"benv2/${START}/${MODEL}/cls/split1\".checkpoint // empty" "${TEACHERS_JSON}")
    if [ -z "${TEACHER}" ] || [ ! -f "${TEACHER}" ]; then
        echo "  [skip] no teacher for benv2/${START}"; continue
    fi
    for KL in kd ttm; do
        for LR in ${LRS}; do
            for T in ${TEMPS}; do
                TAG="retune_${KL}_${START}_to_${NEW}_lr${LR}_T${T}"
                ARGS="baseline/baseline_distillation.py --dataset benv2 --modalities ${NEW}"
                ARGS="${ARGS} --teacher_checkpoint ${TEACHER} --model ${MODEL}"
                ARGS="${ARGS} --epochs ${EPOCHS} --lr ${LR} --kl_type ${KL} --temperature ${T}"
                ARGS="${ARGS} --init_from_teacher --num_workers 2"
                ARGS="${ARGS} --results_csv ${RESULTS_CSV}"
                if [ "$SUBMIT" = "1" ]; then
                    sbatch --time="${WALLTIME}" \
                        --export=ALL,BASELINE_ARGS="${ARGS}",RUN_TAG="${TAG}" \
                        sh/baselines_seeds_job.sh >/dev/null
                fi
                echo "  [$((++n))] ${TAG}"
            done
        done
    done
done

echo
echo "total: ${n} jobs (SUBMIT=${SUBMIT})"
echo "after this completes: pick winners by val, then run 3 seeds of each"
