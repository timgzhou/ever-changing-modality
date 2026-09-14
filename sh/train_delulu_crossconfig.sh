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
    for pair in "s1 s2" "s2 s1" "s2_rgb s2_norgb" "s2_norgb s2_rgb"; do
        set -- $pair; START="$1"; NEW="$2"
        KEY="${PRE}/${START}/evan_base/${DEC}/split1"
        TEACHER=$(jq -r ".\"${KEY}\".checkpoint // empty" "${TEACHERS_JSON}")
        if [ -z "${TEACHER}" ] || [ ! -f "${TEACHER}" ]; then
            echo "  [skip] ${DATASET} ${START}->+${NEW}: no teacher (${KEY})"; miss=$((miss+1)); continue
        fi
        for FAM in ${CONFIG_FAMILIES}; do
            CONFIG="configs/delulu_best_${FAM}.yaml"
            for SEL in ${SELECTORS}; do
                TAG="${DATASET} ${START} ${NEW} ${FAM} ${SEL}"
                if printf '%s' "${INFLIGHT}" | grep -qxF "${TAG}"; then
                    echo "  [have] ${TAG}: in flight"; dup=$((dup+1)); continue
                fi
                n=$((n+1))
                EX="ALL,DATASET=${DATASET},START=${START},NEW=${NEW},TEACHER=${TEACHER}"
                EX="${EX},CONFIG=${CONFIG},SELECT_BY=${SEL},BATCH_SIZE=${BS}"
                EX="${EX},RESULTS_CSV=res/delulu/${DATASET}_crossconfig.csv"
                EX="${EX},CONFIG_LABEL=${FAM}_${SEL}"
                # biomassters: pin 64 epochs, overriding the config's value
                [ "${DATASET}" = "biomassters" ] && EX="${EX},EPOCHS=64"
                if [ "${DRYRUN:-0}" = "1" ]; then
                    echo "[$n] ${DATASET} ${START}->+${NEW}  cfg=${FAM}/${SEL}"
                else
                    jid=$(sbatch --parsable --export="${EX}" sh/train_delulu_job.sh)
                    printf '%s\t%s\n' "${jid}" "${TAG}" >> "${LEDGER}"
                    echo "[$n] ${jid}  ${DATASET} ${START}->+${NEW}  cfg=${FAM}/${SEL}"
                fi
            done
        done
    done
done
echo "total: ${n} submitted; ${dup} already in flight; ${miss} skipped (no teacher)"
