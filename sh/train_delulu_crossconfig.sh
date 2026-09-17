#!/bin/bash
# Cross-config generalisation run: do hyperparameters tuned on one dataset
# transfer to another?
#
#   6 configs   = {dfc2020, benv2} tuned x {transfer, peeking, addition}
#   2 datasets  = dfc2020, biomassters
#   4 directions= s1->s2, s2->s1, s2_rgb->s2_norgb, s2_norgb->s2_rgb
#   = 48 jobs
#
# Note biomassters is NOT one of the tuned datasets, so every biomassters job is
# an out-of-domain test of both config families; dfc2020 jobs test its own
# configs in-domain and the benv2 configs out-of-domain.
#
# EPOCHS: biomassters is capped at 64 regardless of the config's own epoch count.
# Measured throughput is ~9.9 min/epoch (10h for 64 epochs), so the 128-epoch
# dfc2020 configs would need ~20h and be killed at the 11:59 walltime. The cap
# means the 3 dfc2020-tuned configs run at a different epoch count on
# biomassters than in their source sweep -- a confound to keep in mind when
# comparing config families there, but it is the only way every job completes.
#
# Usage:  bash sh/train_delulu_crossconfig.sh
#         DRYRUN=1 bash sh/train_delulu_crossconfig.sh
#         DATASETS="dfc2020" bash sh/train_delulu_crossconfig.sh
#         WALLTIME=23:59:00 DATASETS=biomassters bash sh/train_delulu_crossconfig.sh
#
# WALLTIME overrides the job script's #SBATCH --time. biomassters measured
# 9.4-10.9 min/epoch in practice (not the 9.9 projected), so 64 epochs lands at
# 10.0-11.6h against an 11:59 limit. A job killed at the wall writes NO results
# row -- train_delulu.py appends only after the final epoch -- so a retry should
# use WALLTIME=23:59:00.
set -uo pipefail

TEACHERS_JSON="artifacts/sft_teachers.json"
DATASETS="${DATASETS:-dfc2020 biomassters}"
SELECTORS="${SELECTORS:-transfer peeking addition}"
CONFIG_FAMILIES="${CONFIG_FAMILIES:-dfc2020 benv2}"
LEDGER="logs/train_delulu/submitted_crossconfig.tsv"
mkdir -p logs/train_delulu res/delulu; touch "${LEDGER}"

prefix_for () { [ "$1" = "dfc2020" ] && echo "dfc2020_cobench" || echo "$1"; }
decoder_for () { [ "$1" = "dfc2020" ] && echo "upernet" || echo "upernet+relu"; }
batch_for ()   { [ "$1" = "dfc2020" ] && echo "8" || echo "16"; }

# Skip (dataset,direction,config,selector) combos already submitted and still
# queued/running, so re-running the launcher is safe.
ACTIVE=" $(squeue -u "$USER" -h -o '%i' 2>/dev/null | tr '\n' ' ') "
INFLIGHT=""
while read -r jid rest; do
    case "${ACTIVE}" in *" ${jid} "*) INFLIGHT="${INFLIGHT}${rest}"$'\n' ;; esac
done < "${LEDGER}"

n=0; dup=0; miss=0
for DATASET in ${DATASETS}; do
    PRE=$(prefix_for "${DATASET}"); DEC=$(decoder_for "${DATASET}"); BS=$(batch_for "${DATASET}")
    # PAIRS is a space-separated list of START:NEW, overriding the default,
    # e.g. PAIRS="s2_rgb:s1" to fill a single missing table cell. s2_rgb->s1 is
    # in the paper table but was never in the default list.
    for pair in ${PAIRS:-s1:s2 s2:s1 s2_rgb:s2_norgb s2_norgb:s2_rgb}; do
        START="${pair%%:*}"; NEW="${pair##*:}"
        KEY="${PRE}/${START}/evan_base/${DEC}/split1"
        TEACHER=$(jq -r ".\"${KEY}\".checkpoint // empty" "${TEACHERS_JSON}")
        if [ -z "${TEACHER}" ] || [ ! -f "${TEACHER}" ]; then
            echo "  [skip] ${DATASET} ${START}->+${NEW}: no teacher (${KEY})"; miss=$((miss+1)); continue
        fi
        for FAM in ${CONFIG_FAMILIES}; do
            CONFIG="configs/delulu_best_${FAM}.yaml"
            for SEL in ${SELECTORS}; do
                TAG="${DATASET} ${START} ${NEW} ${FAM} ${SEL}${LR:+ lr${LR}}${STUDENT_INIT:+ init${STUDENT_INIT}}${SEED:+ s${SEED}}"
                if printf '%s' "${INFLIGHT}" | grep -qxF "${TAG}"; then
                    echo "  [have] ${TAG}: in flight"; dup=$((dup+1)); continue
                fi
                n=$((n+1))
                EX="ALL,DATASET=${DATASET},START=${START},NEW=${NEW},TEACHER=${TEACHER}"
                EX="${EX},CONFIG=${CONFIG},SELECT_BY=${SEL},BATCH_SIZE=${BS}"
                EX="${EX},RESULTS_CSV=res/delulu/${DATASET}_crossconfig.csv"
                # LR override: the source config's lr can be catastrophically
                # wrong for the target dataset. On biomassters every config with
                # lr <= 1e-4 collapsed (dfc2020_peeking 5/5, benv2_addition 4/4;
                # RMSE 70-86 against a ~42-48 teacher) while every config with
                # lr >= 3.9e-4 survived (0/14). Set LR=<value> to rescue those
                # families; the label records it so the rows stay distinguishable.
                LBL="${FAM}_${SEL}"
                if [ -n "${LR:-}" ]; then
                    EX="${EX},LR=${LR}"
                    LBL="${LBL}_lr${LR}"
                fi
                # STUDENT_INIT=random re-rolls the student's weights while
                # KEEPING the frozen teacher, so supervision is unchanged and
                # only the starting point moves (train_delulu.py:255). The
                # earlier 6-pair probe at lr 1.57e-4 predates the per-selector
                # tuned configs and had no config_label, so it is not comparable
                # to the current table; this reruns it properly. The label
                # records the arm so teacher- and random-init rows never pool.
                if [ -n "${STUDENT_INIT:-}" ] && [ "${STUDENT_INIT}" != "teacher" ]; then
                    EX="${EX},STUDENT_INIT=${STUDENT_INIT}"
                    LBL="${LBL}_init${STUDENT_INIT}"
                fi
                # SEED: replicates. The teacher-init rows show within-pair sd
                # 0.77-0.99 mIoU, so a single run per cell cannot resolve the
                # ~0.5 effect this ablation is chasing.
                [ -n "${SEED:-}" ] && EX="${EX},SEED=${SEED}" && LBL="${LBL}_s${SEED}"
                EX="${EX},CONFIG_LABEL=${LBL}"
                # biomassters: pin 64 epochs, overriding the config's value
                [ "${DATASET}" = "biomassters" ] && EX="${EX},EPOCHS=64"
                if [ "${DRYRUN:-0}" = "1" ]; then
                    echo "[$n] ${DATASET} ${START}->+${NEW}  cfg=${FAM}/${SEL}"
                else
                    TIME_ARG=""
                    [ -n "${WALLTIME:-}" ] && TIME_ARG="--time=${WALLTIME}"
                    jid=$(sbatch --parsable ${TIME_ARG} --export="${EX}" sh/train_delulu_job.sh)
                    printf '%s\t%s\n' "${jid}" "${TAG}" >> "${LEDGER}"
                    echo "[$n] ${jid}  ${DATASET} ${START}->+${NEW}  cfg=${FAM}/${SEL}"
                fi
            done
        done
    done
done
echo "total: ${n} submitted; ${dup} already in flight; ${miss} skipped (no teacher)"
