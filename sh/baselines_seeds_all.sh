#!/bin/bash
# Seed replicates for the reBEN and EuroSAT baselines, so every dataset in the
# paper reports a real seed std rather than a hyperparameter spread.
#
# dfc2020 was done separately (sh/baselines_seeds_dfc2020.sh, 60 runs, landed
# 2026-09-19). This covers the other two datasets on the same protocol:
#   pick the val-winning config per cell, then run 3 seeds of exactly that one.
#
# The baselines need NO new tuning -- reBEN and EuroSAT already have 3-20
# configs swept per cell. The winning config is read out of the existing CSVs by
# sweep/pick_baseline_winners.py rather than hardcoded here, so this script
# cannot drift from the data it claims to replicate.
#
# Seeded runs write to *_seeds.csv. That separate file is required, not
# cosmetic: the CSV writers only emit a header for a NEW file, so appending a
# column to an existing CSV yields wider rows under an old header, pandas raises
# ParserError, and res/results_BL.py silently SKIPS those rows.
#
# Usage:
#   bash sh/baselines_seeds_all.sh                     # dry run, prints the plan
#   SUBMIT=1 bash sh/baselines_seeds_all.sh            # actually sbatch
#   DATASETS=eurosat SUBMIT=1 bash ...                 # one dataset
#   SEEDS="1 2" SUBMIT=1 bash ...                      # subset of seeds

set -u
SEEDS="${SEEDS:-0 1 2}"
SUBMIT="${SUBMIT:-0}"
DATASETS="${DATASETS:-benv2 eurosat}"
WALLTIME="${WALLTIME:-3:00:00}"
PLAN="${PLAN:-}"

# Build the per-(dataset,cell) winning-config plan.
if [ -z "${PLAN}" ]; then
    PLAN=$(mktemp)
    python sweep/pick_baseline_winners.py --datasets ${DATASETS} > "${PLAN}" || {
        echo "[error] could not build the winner plan"; exit 1; }
fi

n=0
while IFS=$'\t' read -r TAG ARGS; do
    [ -z "${TAG}" ] && continue
    case "${TAG}" in \#*) continue;; esac
    for S in ${SEEDS}; do
        FULL="${ARGS} --seed ${S}"
        if [ "$SUBMIT" = "1" ]; then
            sbatch --time="${WALLTIME}" \
                --export=ALL,BASELINE_ARGS="${FULL}",RUN_TAG="seed${S}_${TAG}" \
                sh/baselines_seeds_job.sh >/dev/null
        fi
        echo "  [$((++n))] seed${S}_${TAG}"
    done
done < "${PLAN}"

echo
echo "total: ${n} jobs (SUBMIT=${SUBMIT}, datasets='${DATASETS}', seeds='${SEEDS}')"
