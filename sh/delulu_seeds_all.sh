#!/bin/bash
# Three seeds of each Delulu per-(dataset, direction, selector) sweep winner.
#
# Step 3 of the paper protocol:
#   1. structural knobs (mask ratio, both dropouts, labeled_frequency,
#      lambda_distill) tuned once on DFC2020 and TRANSFERRED to every dataset
#      -- that transferability is a reported result, not a confound;
#   2. lr + lambda_latent + lambda_prefusion swept per (dataset, direction,
#      selector)  -> sweep/create_sweep_paper_losswts.py, 27 sweeps x 12 trials;
#   3. THIS SCRIPT: re-run each sweep's val-winning config with 3 seeds. Those
#      three runs are what the table's mean +/- std reports.
#
# Winners come from res/delulu-sweep/paper_losswts_winners.json, generated from
# the sweep CSVs -- never hardcoded, so the seeds always reproduce the config
# the tables claim.
#
# Usage:
#   bash sh/delulu_seeds_all.sh            # dry run
#   SUBMIT=1 bash sh/delulu_seeds_all.sh   # actually sbatch
#   DATASETS=eurosat SUBMIT=1 bash ...     # one dataset

set -u
WINNERS="${WINNERS:-res/delulu-sweep/paper_losswts_winners.json}"
SEEDS="${SEEDS:-0 1 2}"
SUBMIT="${SUBMIT:-0}"
DATASETS="${DATASETS:-dfc2020 benv2 eurosat}"
RESULTS_CSV="${RESULTS_CSV:-res/delulu/paper_seeds.csv}"
# Measured 2026-09-20: a 64-epoch Delulu trial runs ~2h on eurosat (113s/epoch)
# and up to ~5h on dfc2020. 8h gives headroom without over-asking.
WALLTIME="${WALLTIME:-8:00:00}"

n=0
for S in ${SEEDS}; do
for DS in ${DATASETS}; do
    COUNT=$(jq "[.[] | select(.dataset==\"${DS}\")] | length" "${WINNERS}")
    for i in $(seq 0 $((COUNT-1))); do
        E=$(jq -c "[.[] | select(.dataset==\"${DS}\")] | .[${i}]" "${WINNERS}")
        START=$(echo "$E"   | jq -r '.start')
        NEW=$(echo "$E"     | jq -r '.new')
        SEL=$(echo "$E"     | jq -r '.select_by')
        CKPT=$(echo "$E"    | jq -r '.stage0_checkpoint')
        if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
            echo "  [skip] ${DS} ${START}->${NEW} ${SEL}: stage0 checkpoint missing"; continue
        fi
        # dfc2020 is dense segmentation and needs the smaller batch.
        BS=32; [ "${DS}" = "dfc2020" ] && BS=8
        ARGS="--dataset ${DS} --new_mod_group ${NEW} --stage0_checkpoint ${CKPT}"
        ARGS="${ARGS} --select_by ${SEL} --seed ${S} --batch_size ${BS} --num_workers 2"
        for K in lr weight_decay epochs lambda_latent lambda_prefusion lambda_distill \
                 labeled_frequency modality_dropout_startmod modality_dropout_newmod; do
            V=$(echo "$E" | jq -r ".${K}")
            [ "$V" != "null" ] && [ -n "$V" ] && ARGS="${ARGS} --${K} ${V}"
        done
        # mae_mask_ratio is spelled --token_mask_ratio on train_delulu.py's CLI.
        MR=$(echo "$E" | jq -r '.mae_mask_ratio')
        [ "$MR" != "null" ] && ARGS="${ARGS} --token_mask_ratio ${MR}"
        ARGS="${ARGS} --latent_masked_only --active_losses latent prefusion distill ce"
        ARGS="${ARGS} --config_label paperseed_${DS}_${START}_to_${NEW}_${SEL}"
        ARGS="${ARGS} --results_csv ${RESULTS_CSV}"
        TAG="seed${S}_${DS}_${START}_to_${NEW}_${SEL}"
        if [ "$SUBMIT" = "1" ]; then
            sbatch --time="${WALLTIME}" --job-name="dseed_${DS}_${SEL}" \
                --export=ALL,DELULU_ARGS="${ARGS}",RUN_TAG="${TAG}" \
                sh/delulu_seeds_job.sh >/dev/null
        fi
        echo "  [$((++n))] ${TAG}"
    done
done
done

echo
echo "total: ${n} jobs (SUBMIT=${SUBMIT}, seeds='${SEEDS}', datasets='${DATASETS}')"
