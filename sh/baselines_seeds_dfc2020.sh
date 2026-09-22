#!/bin/bash
# Seed replicates for the DFC2020 baselines, so the tables can report a real
# seed std instead of a hyperparameter spread.
#
# Why this exists
# ---------------
# Every +/- printed for a baseline in res/latex/BL_tables*.tex is currently the
# spread across DIFFERENT CONFIGURATIONS, not across seeds. None of the four
# baseline families had a --seed flag at all, and `global_rep` -- which looks
# like a seed column -- is the pooling mode ('clstoken' in every row of every
# baseline CSV). Grouping any baseline family by its full hyperparameter tuple
# gives exactly one row per config.
#
# Concretely, for KD/TTM on dfc2020 each cell had only TWO candidate rows (lr
# 5e-4 and lr 1e-4), so "top-2 by val" consumed the entire pool and the error
# bar was the gap between two learning rates:
#     (s2_rgb, s2_norgb, kd)  tests [59.00, 58.45] -> +/-0.39
# Delulu meanwhile has real seeds (seed in {0,1,2} on dfc2020), so the two
# columns' error bars did not mean the same thing.
#
# What this runs
# --------------
# ONE config per cell -- the one that currently wins by val, so the mean stays
# comparable to what the tables already report -- times SEEDS. Seeded runs write
# to *_seeds.csv: the existing CSVs have a fixed-width header and csv.writer
# only emits a header for a NEW file, so an appended column would make pandas
# raise ParserError and results_BL.py would silently SKIP those rows.
#
# Winning configs were read off the current CSVs; all match the script defaults
# except mixmatch lambda_u (0.5, not the 75.0 default that collapses training)
# and the s2_rgb->s2_norgb TTM cell (lr 5e-4).
#
# Usage:
#   bash sh/baselines_seeds_dfc2020.sh            # dry run, prints the plan
#   SUBMIT=1 bash sh/baselines_seeds_dfc2020.sh   # actually sbatch
#   SEEDS="1 2" SUBMIT=1 bash ...                 # subset of seeds

set -u
DECODER="${DECODER:-upernet}"
MODEL="${MODEL:-evan_base}"
EPOCHS="${EPOCHS:-64}"
# seed 0 is run too: the existing rows predate the --seed flag, so their seed is
# whatever torch defaulted to, not a recorded 0. Re-running it keeps all three
# replicates on one footing.
SEEDS="${SEEDS:-0 1 2}"
SUBMIT="${SUBMIT:-0}"
# Overrides the job script's 11:59:00. These runs measured 00:47-01:43 over the
# last month (sacct, baselines_dfc2020_job.sh), so 3h is ~1.7x the worst case
# and queues far sooner than a 12h ask.
WALLTIME="${WALLTIME:-3:00:00}"
TEACHERS_JSON="artifacts/sft_teachers.json"
TEACHER_SPLIT="${TEACHER_SPLIT:-split1}"
FAMILIES="${FAMILIES:-distillation mke freematch mixmatch}"

# teacher-based cells: START:NEW, matching the pairs the tables report
PAIRS="${PAIRS:-s1:s2 s2:s1 s2_rgb:s1 s2_rgb:s2_norgb}"
# semi-supervised cells: single modalities
SINGLES="${SINGLES:-s1 s2 s2_rgb s2_norgb}"

# Tags to skip, one per line, e.g. the runs already in flight after a partial
# submission. SKIP_FILE=<path> to use.
SKIP_FILE="${SKIP_FILE:-}"

n=0
skipped=0
submit () {  # $1=tag  $2...=args
    local tag="$1"; shift
    if [ -n "${SKIP_FILE}" ] && grep -qxF "${tag}" "${SKIP_FILE}" 2>/dev/null; then
        skipped=$((skipped+1)); return
    fi
    if [ "$SUBMIT" = "1" ]; then
        sbatch --time="${WALLTIME}" \
            --export=ALL,BASELINE_ARGS="$*",RUN_TAG="${tag}",DECODER="${DECODER}" \
            sh/baselines_dfc2020_job.sh >/dev/null
    fi
    echo "  [$((++n))] ${tag}"
}

has () { case " ${FAMILIES} " in *" $1 "*) return 0;; *) return 1;; esac; }

for S in ${SEEDS}; do
echo "=== seed ${S} ==="

if has distillation || has mke; then
for P in ${PAIRS}; do
    START="${P%%:*}"; NEW="${P##*:}"
    TEACHER=$(jq -r ".\"dfc2020_cobench/${START}/${MODEL}/${DECODER}/${TEACHER_SPLIT}\".checkpoint // empty" "${TEACHERS_JSON}")
    if [ -z "${TEACHER}" ] || [ ! -f "${TEACHER}" ]; then
        echo "  [skip] no ${DECODER}/${TEACHER_SPLIT} teacher for ${START}"; continue
    fi

    if has distillation; then
    for KL in kd ttm; do
        # Only this one cell wants 5e-4; every other KD/TTM cell wins at 1e-4.
        LR=0.0001
        if [ "${START}:${NEW}:${KL}" = "s2_rgb:s2_norgb:ttm" ]; then LR=0.0005; fi
        submit "seed${S}_distill_${KL}_${START}_to_${NEW}" \
            "baseline/baseline_distillation.py --dataset dfc2020 --modalities ${NEW}" \
            "--teacher_checkpoint ${TEACHER} --decoder_type ${DECODER} --model ${MODEL}" \
            "--epochs ${EPOCHS} --lr ${LR} --kl_type ${KL} --init_from_teacher --seed ${S}" \
            "--results_csv res/baselines/dfc2020_cobench_distill_transfer_${DECODER}.csv"
    done
    fi

    if has mke; then
    submit "seed${S}_mke_${START}_to_${NEW}" \
        "baseline/baseline_mke.py --dataset dfc2020 --modalities ${START} ${NEW}" \
        "--teacher_checkpoint ${TEACHER} --decoder_type ${DECODER} --model ${MODEL}" \
        "--epochs ${EPOCHS} --lr 0.0001 --init_from_teacher --seed ${S}" \
        "--results_csv res/baselines/dfc2020_cobench_mke_${DECODER}.csv"
    fi
done
fi

for M in ${SINGLES}; do
    if has freematch; then
    submit "seed${S}_freematch_${M}" \
        "baseline/baseline_freematch.py --dataset dfc2020 --modality ${M}" \
        "--decoder_type ${DECODER} --model ${MODEL} --use_dino_weights" \
        "--epochs ${EPOCHS} --lr 0.0001 --seed ${S}" \
        "--results_csv res/baselines/dfc2020_cobench_freematch_${DECODER}.csv"
    fi
    if has mixmatch; then
    submit "seed${S}_mixmatch_${M}" \
        "baseline/baseline_mixmatch.py --dataset dfc2020 --modality ${M}" \
        "--decoder_type ${DECODER} --model ${MODEL} --use_dino_weights" \
        "--epochs ${EPOCHS} --lr 0.0001 --lambda_u 0.5 --seed ${S}" \
        "--results_csv res/baselines/dfc2020_cobench_mixmatch_${DECODER}.csv"
    fi
done
done

echo
echo "total: ${n} jobs submitted, ${skipped} skipped (SUBMIT=${SUBMIT}, seeds='${SEEDS}')"
