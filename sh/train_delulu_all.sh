#!/bin/bash
# Stage-1 (train_delulu) launcher: all requested unimodal pair directions,
# across every dataset, using the split1 stage-0 teachers.
#
# Directions (teacher -> added modality):
#   eurosat                        pairwise over rgb / vre / nir      (6)
#   benv2 / dfc2020 / biomassters  pairwise over s1 / s2,             (4 each)
#                                  plus pairwise over s2_norgb / s2_rgb
#
# TEACHER_SPLIT=split1 is required: stage 1 uses train2 as its UNLABELED pool,
# so a `full` teacher -- supervised on train1+train2 -- has already seen that
# pool with labels and the comparison leaks.
#
# Teachers come from artifacts/sft_teachers.json; regenerate it with
#   python res/train_sft/sft_best.py
# after the stage-0 jobs land, or run sh/train_delulu_after_sft.sh which does
# that and then calls this script.
#
# Usage:
#   bash sh/train_delulu_all.sh                  # submit
#   DRYRUN=1 bash sh/train_delulu_all.sh         # print only
#   DATASETS="dfc2020" bash sh/train_delulu_all.sh
set -uo pipefail

TEACHERS_JSON="artifacts/sft_teachers.json"
TEACHER_SPLIT="${TEACHER_SPLIT:-split1}"
MODEL="${MODEL:-evan_base}"
DATASETS="${DATASETS:-eurosat benv2 dfc2020 biomassters}"
EPOCHS="${EPOCHS:-64}"

if [ ! -f "${TEACHERS_JSON}" ]; then
    echo "[error] ${TEACHERS_JSON} missing — run: python res/train_sft/sft_best.py"; exit 1
fi

# Registry prefix and decoder tag differ per dataset.
#   dfc2020 rows are registered under the cobench split name.
#   decoder tag must match what stage 0 wrote: cls for classification/multilabel,
#   upernet for dfc2020 segmentation, upernet+relu for biomassters regression.
prefix_for () { [ "$1" = "dfc2020" ] && echo "dfc2020_cobench" || echo "$1"; }
decoder_for () {
    case "$1" in
        eurosat|benv2) echo "cls" ;;
        dfc2020)       echo "${DECODER:-upernet}" ;;
        biomassters)   echo "upernet+relu" ;;
    esac
}
mods_for () {
    case "$1" in
        eurosat) echo "rgb vre nir" ;;
        *)       echo "s1 s2|s2_norgb s2_rgb" ;;   # '|' separates independent groups
    esac
}
# Per-dataset batch size. dfc2020 is 256x256 segmentation; biomassters is
# temporal (T=12 frames per sample, so ~12x the activations of a still image)
# and OOMs at 32 on an L40S (44 GB) -- sh/train_delulu_biomassters_best_job.sh
# has always used 16.
batch_for () {
    case "$1" in
        dfc2020)     echo "8"  ;;
        biomassters) echo "16" ;;
        *)           echo "32" ;;
    esac
}

# Skip directions that are already finished (a row in the results CSV) or
# already queued/running (the job echoes "<start> -> +<new>" into its log).
# Without this, re-running the launcher -- which the chain job does once the
# slower teachers land -- would submit a duplicate of every direction.
# Submission ledger: "<jobid> <dataset> <start> <new>" appended at submit time.
# A queued job has not written a log yet, so scraping logs misses exactly the
# jobs we most need to detect. The ledger is filtered against squeue so entries
# for finished/cancelled jobs do not block a legitimate resubmission.
LEDGER="logs/train_delulu/submitted.tsv"
mkdir -p logs/train_delulu; touch "${LEDGER}"
INFLIGHT=""
if [ -z "${SKIP_INFLIGHT_CHECK:-}" ] && command -v squeue >/dev/null 2>&1; then
    ACTIVE=" $(squeue -u "$USER" -h -o '%i' 2>/dev/null | tr '\n' ' ') "
    while read -r jid ds st nw; do
        case "${ACTIVE}" in *" ${jid} "*) INFLIGHT="${INFLIGHT}${ds} ${st} ${nw}"$'\n' ;; esac
    done < "${LEDGER}"
fi
already_done () {   # $1=dataset $2=start $3=new
    local csv="res/delulu/$1_unimodal_pairs.csv"
    [ -f "${csv}" ] && awk -F, -v s="$2" -v n="$3" 'NR>1 && $4==s && $5==n {found=1} END{exit !found}' "${csv}" && return 0
    printf '%s' "${INFLIGHT}" | grep -qxF "$1 $2 $3" && return 0
    return 1
}

n=0; miss=0; dup=0
for DATASET in ${DATASETS}; do
    PREFIX=$(prefix_for "${DATASET}")
    DEC=$(decoder_for "${DATASET}")
    BS=$(batch_for "${DATASET}")
    # NB: the array is MOD_GROUPS, not GROUPS -- bash pre-populates GROUPS with
    # the caller's unix group ids and silently ignores assignments to it, which
    # made every direction collapse to START == NEW and get skipped.
    # IFS is restored right after the split: the inner `for START in ${GROUP}`
    # needs normal whitespace word-splitting.
    IFS='|' read -ra MOD_GROUPS <<< "$(mods_for "${DATASET}")"
    unset IFS
    for GROUP in "${MOD_GROUPS[@]}"; do
        for START in ${GROUP}; do
            for NEW in ${GROUP}; do
                [ "${START}" = "${NEW}" ] && continue
                KEY="${PREFIX}/${START}/${MODEL}/${DEC}/${TEACHER_SPLIT}"
                TEACHER=$(jq -r ".\"${KEY}\".checkpoint // empty" "${TEACHERS_JSON}")
                if [ -z "${TEACHER}" ] || [ ! -f "${TEACHER}" ]; then
                    echo "  [skip] ${DATASET} ${START}->+${NEW}: no teacher for ${KEY}"
                    miss=$((miss+1)); continue
                fi
                if already_done "${DATASET}" "${START}" "${NEW}"; then
                    echo "  [have] ${DATASET} ${START}->+${NEW}: finished or in flight, skipping"
                    dup=$((dup+1)); continue
                fi
                n=$((n+1))
                EXPORTS="ALL,DATASET=${DATASET},START=${START},NEW=${NEW},TEACHER=${TEACHER},EPOCHS=${EPOCHS},BATCH_SIZE=${BS},RESULTS_CSV=res/delulu/${DATASET}_unimodal_pairs.csv"
                if [ "${DRYRUN:-0}" = "1" ]; then
                    echo "[$n] ${DATASET}  ${START} -> +${NEW}   teacher=$(basename "${TEACHER}")"
                else
                    jid=$(sbatch --parsable --export="${EXPORTS}" sh/train_delulu_job.sh)
                    printf '%s\t%s\t%s\t%s\n' "${jid}" "${DATASET}" "${START}" "${NEW}" >> "${LEDGER}"
                    echo "[$n] submitted ${jid}  ${DATASET}  ${START} -> +${NEW}"
                fi
            done
        done
    done
done
echo "total: ${n} train_delulu jobs; ${miss} skipped (teacher missing); ${dup} skipped (already done/in flight)"
