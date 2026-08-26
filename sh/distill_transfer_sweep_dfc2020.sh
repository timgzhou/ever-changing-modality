#!/bin/bash
# Hyperparameter sweep for the TRANSFER distillation baseline on DFC2020/cobench.
#
# Transfer is DeluluNet's weakest setting, and the baseline it is measured
# against had only 2 LRs x 2 kl_types (4 configs) versus delulu's 16-trial
# sweep. This levels the comparison.
#
# Hyperparameters that actually matter for pure distillation (alpha=1.0, so the
# student sees NO labels -- only the teacher's soft outputs on train2):
#
#   lr                learning rate
#   weight_decay      AdamW decay. Was previously hardcoded to PyTorch's 0.01
#                     default and not tunable; --weight_decay added 2026-08-22.
#   temperature       softens the teacher distribution. THE key KD knob: it
#                     controls how much dark knowledge (relative probabilities
#                     among non-argmax classes) the student sees.
#   kl_type           kd (both logits scaled, loss x T^2) vs ttm (teacher only,
#                     no T^2). These interact strongly with temperature -- KD's
#                     T^2 makes its effective LR scale with T^2, TTM's does not.
#   init_from_teacher start the student from teacher weights instead of DINO.
#                     Only legal for unimodal students, i.e. exactly this
#                     setting. Plausibly a large effect and never tested.
#
# Deliberately NOT swept: alpha (1.0 defines "pure distillation"; anything less
# leaks train2 labels and breaks the protocol), distillation_mode (feature-MSE
# is a different method, not a hyperparameter), epochs, batch_size.
#
# Grid: 3 lr x 2 wd x 3 T x 2 kl x 2 init = 72 configs per direction. Too many,
# so this uses random search over the grid, N_TRIALS per direction.
#
# Usage:
#   bash sh/distill_transfer_sweep_dfc2020.sh              # dry run
#   SUBMIT=1 bash sh/distill_transfer_sweep_dfc2020.sh
#   DIRECTIONS='s1:s2_norgb' N_TRIALS=24 SUBMIT=1 bash ...

set -u
DECODER="${DECODER:-upernet}"
MODEL="${MODEL:-evan_base}"
TEACHER_SPLIT="${TEACHER_SPLIT:-split1}"
EPOCHS="${EPOCHS:-64}"
SUBMIT="${SUBMIT:-0}"
N_TRIALS="${N_TRIALS:-16}"
SEED="${SEED:-0}"
TEACHERS_JSON="artifacts/sft_teachers.json"

# the three directions the delulu sweep covers, so the comparison is matched
DIRECTIONS="${DIRECTIONS:-s1:s2_norgb s2_rgb:s2_norgb s2_norgb:s2_rgb}"

LRS="5e-5 1e-4 5e-4"
WDS="0.0 0.01"
TEMPS="1.0 2.0 4.0"
KLS="kd ttm"
INITS="0 1"

# Draw the whole grid once with python (proper RNG); the shell strides it.
# The hand-rolled modular "pick" this replaces aliased badly -- with 6 distinct
# strides over 2-3 option lists, kl_type stayed on 'kd' for every trial and the
# configs repeated with period 6.
CONFIGS=$(python3 - "$N_TRIALS" "$SEED" <<'PYEOF'
import random, sys
n, seed = int(sys.argv[1]), int(sys.argv[2])
rng = random.Random(seed)
LRS=['5e-5','1e-4','5e-4']; WDS=['0.0','0.01']; TEMPS=['1.0','2.0','4.0']
KLS=['kd','ttm']; INITS=['0','1']
seen=set()
while len(seen) < n:
    c=(rng.choice(LRS),rng.choice(WDS),rng.choice(TEMPS),rng.choice(KLS),rng.choice(INITS))
    seen.add(c)                      # sample WITHOUT replacement from the 72-point grid
for c in list(seen)[:n]:
    print(' '.join(c))
PYEOF
)

n=0
for P in ${DIRECTIONS}; do
    START="${P%%:*}"; NEW="${P##*:}"
    TEACHER=$(jq -r ".\"dfc2020_cobench/${START}/${MODEL}/${DECODER}/${TEACHER_SPLIT}\".checkpoint // empty" "${TEACHERS_JSON}")
    if [ -z "${TEACHER}" ] || [ ! -f "${TEACHER}" ]; then
        echo "  [skip] no ${DECODER}/${TEACHER_SPLIT} teacher for ${START}"; continue
    fi
    while read -r LR WD T KL IT; do
        [ -z "${LR:-}" ] && continue
        [ "$IT" = "1" ] && INIT_FLAG="--init_from_teacher" || INIT_FLAG=""
        TAG="dts_${START}_to_${NEW}_lr${LR}_wd${WD}_T${T}_${KL}_init${IT}"
        ARGS="baseline/baseline_distillation.py --dataset dfc2020 --modalities ${NEW}"
        ARGS="${ARGS} --teacher_checkpoint ${TEACHER} --decoder_type ${DECODER} --model ${MODEL}"
        ARGS="${ARGS} --epochs ${EPOCHS} --lr ${LR} --weight_decay ${WD}"
        ARGS="${ARGS} --temperature ${T} --kl_type ${KL} ${INIT_FLAG}"
        ARGS="${ARGS} --results_csv res/baselines/dfc2020_cobench_distill_transfer_sweep_${DECODER}.csv"
        if [ "$SUBMIT" = "1" ]; then
            sbatch --export=ALL,BASELINE_ARGS="${ARGS}",RUN_TAG="${TAG}",DECODER="${DECODER}" \
                sh/baselines_dfc2020_job.sh >/dev/null
        fi
        n=$((n+1))
        echo "  [$n] ${TAG}"
    done <<< "${CONFIGS}"
done
echo
echo "total: ${n} jobs (SUBMIT=${SUBMIT})"
