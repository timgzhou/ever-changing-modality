#!/bin/bash
# Random HP sweep for DISTILL-ONLY DeluluNet on DFC2020/cobench, all 6 directions.
#
# Motivation: the head analysis showed the transfer path's weakness is the
# HALLUCINATED head (48.9 vs 51.3 for the real head, distill_only). Modality
# dropout is what trains that path -- dropping the START modality during
# training is exactly what forces the model to learn to hallucinate it. The
# current rates (start 0.332 / new 0.171) are carried over from an untuned
# biomassters config and were never tuned for DFC2020, so they are the first
# suspect. token_mask_ratio routes the projectors on non-dropout steps and is
# swept alongside; lr/wd are included since they were also inherited.
#
# Sampling (log-uniform for lr/wd, uniform for the rest):
#   modality_dropout_startmod  0.10 .. 0.70   <- the transfer-critical knob
#   modality_dropout_newmod    0.05 .. 0.50
#   token_mask_ratio           0.15 .. 0.75
#   lr                         5e-5 .. 5e-4
#   weight_decay               1e-6 .. 1e-3
#
# Usage:
#   bash sh/distillonly_sweep_dfc2020.sh              # dry run
#   SUBMIT=1 bash sh/distillonly_sweep_dfc2020.sh     # 6 x N_TRIALS jobs
#   N_TRIALS=8 SUBMIT=1 bash ...                      # smaller first pass

set -u
DECODER="${DECODER:-upernet}"
MODEL="${MODEL:-evan_base}"
TEACHER_SPLIT="${TEACHER_SPLIT:-split1}"
EPOCHS="${EPOCHS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_TRIALS="${N_TRIALS:-32}"
SEED_BASE="${SEED_BASE:-1000}"
SUBMIT="${SUBMIT:-0}"
TEACHERS_JSON="artifacts/sft_teachers.json"
PAIRS="${PAIRS:-s1:s2_norgb s1:s2_rgb s2_norgb:s1 s2_norgb:s2_rgb s2_rgb:s1 s2_rgb:s2_norgb}"

n=0
for P in ${PAIRS}; do
    START="${P%%:*}"; NEW="${P##*:}"
    TEACHER=$(jq -r ".\"dfc2020_cobench/${START}/${MODEL}/${DECODER}/${TEACHER_SPLIT}\".checkpoint // empty" "${TEACHERS_JSON}")
    if [ -z "${TEACHER}" ] || [ ! -f "${TEACHER}" ]; then
        echo "[error] no ${DECODER}/${TEACHER_SPLIT} teacher for ${START}"; exit 1
    fi
    for t in $(seq 0 $((N_TRIALS - 1))); do
        # deterministic per (direction, trial) so the sweep is reproducible
        read MDS MDN TMR LR WD < <(python3 - "$START" "$NEW" "$t" "$SEED_BASE" <<'PY'
import sys, random, hashlib
start, new, trial, base = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
# hash() is salted per process (PYTHONHASHSEED), so use a stable digest instead
key = f"{start}|{new}|{trial}|{base}".encode()
random.seed(int(hashlib.sha256(key).hexdigest()[:16], 16))
lu = lambda a, b: 10 ** random.uniform(__import__('math').log10(a), __import__('math').log10(b))
print(f"{random.uniform(0.10,0.70):.4f} {random.uniform(0.05,0.50):.4f} "
      f"{random.uniform(0.15,0.75):.4f} {lu(5e-5,5e-4):.6g} {lu(1e-6,1e-3):.6g}")
PY
)
        TRIAL_TAG="t${t}"
        if [ "$SUBMIT" = "1" ]; then
            sbatch --export=ALL,START="${START}",NEW="${NEW}",TEACHER="${TEACHER}",\
DECODER="${DECODER}",EPOCHS="${EPOCHS}",BATCH_SIZE="${BATCH_SIZE}",DISTILL_ONLY=1,\
MODALITY_DROPOUT_STARTMOD="${MDS}",MODALITY_DROPOUT_NEWMOD="${MDN}",\
TOKEN_MASK_RATIO="${TMR}",LR="${LR}",WEIGHT_DECAY="${WD}",\
TRIAL_TAG="${TRIAL_TAG}",SEED=0,\
RESULTS_CSV="res/delulu/dfc2020_cobench_distillonly_sweep_${DECODER}.csv" \
                sh/shot_ete_dfc2020_job.sh >/dev/null
        fi
        n=$((n+1))
        [ "$t" -lt 2 ] && echo "  [$n] ${START}->+${NEW} ${TRIAL_TAG}: mds=${MDS} mdn=${MDN} tmr=${TMR} lr=${LR} wd=${WD}"
    done
    echo "  ... ${N_TRIALS} trials for ${START}->+${NEW}"
done
echo
echo "total: ${n} jobs (SUBMIT=${SUBMIT})"
