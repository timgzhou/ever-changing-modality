#!/bin/bash
# Launcher: one SLURM job per model+dataset+modality combo. Each job sweeps 9
# lr/wd combos internally and SKIPS combos already present in
# res/rsfm/rsfm_results.csv, so re-running is safe and resumes.
#
# Usage:
#   bash sh/rsfm_sft/rsfm_sft_all.sh            # dry run, prints the plan
#   SUBMIT=1 bash sh/rsfm_sft/rsfm_sft_all.sh
#
# WALLTIME: 9 sweep combos x 20 epochs in one job. The #SBATCH default is 2h;
# WALLTIME here overrides it, and a job killed at the wall loses only the combos
# it had not yet written (the CSV is appended per combo, and the skip-check
# above makes a resubmit resume).
#
# The dfc2020 oracle rows were re-run on 2026-09-15: the previous ones predated
# the Copernicus-Bench split and were quarantined to res/ignore/. rsfm_sft.py
# loads train1/val1, i.e. split1, so these are the correct non-leaky oracles.
set -u
SUBMIT="${SUBMIT:-0}"
WALLTIME="${WALLTIME:-5:59:00}"

declare -A MODALITY_CONFIGS
# The modalities the paper tables need: the four single modalities for the
# Transfer/Peek oracle columns, plus the combined ones for Addition.
# s2s1 covers the s1<->s2 Addition rows; s2_rgb+s1 and s2_rgb+s2_norgb cover the
# two S2-RGB Addition rows. COMBINED_RSFM_ALIASES in results_BL.py maps them.
MODALITY_CONFIGS['dfc2020']="${MODALITIES:-s1 s2 s2_rgb s2_norgb s2s1 s2_rgb+s1 s2_rgb+s2_norgb}"
# biomassters (enabled 2026-09-17). It is temporal (T=12) and a REGRESSION task,
# and rsfm_sft.py previously supported neither. Both are now handled:
#   temporal   - neither wrapper is temporal (Panopticon's backbone has no time
#                axis; OlmoEarthWrapper hardcodes T=1 via .unsqueeze(3)), so
#                rsfm_sft.py folds T into the batch, runs the frozen backbone per
#                timestep, and mean-pools FEATURES over T -- the same shim EVAN
#                uses, so the oracles stay comparable to the models they bound.
#   regression - dense head with num_classes=1, MSELoss, RMSE metric, and
#                LOWER-IS-BETTER val selection (the sentinel and comparison flip).
# BATCH_SIZE must be small: T folds into the batch, so 4 -> 48 at the backbone.
MODALITY_CONFIGS['biomassters']="${MODALITIES:-s1 s2 s2_rgb s2_norgb s2s1 s2_rgb+s1 s2_rgb+s2_norgb}"
MODALITY_CONFIGS['benv2']="${MODALITIES:-s2_norgb}"
MODALITY_CONFIGS['eurosat']="${MODALITIES:-rgb+nir rgb+vre rgb+swir}"

MODELS="${MODELS:-panopticon olmoearth-base olmoearth-large}"
DATASETS="${DATASETS:-dfc2020}"
TRAIN_MODES="${TRAIN_MODES:-fft}"

n=0
for MODEL in ${MODELS}; do
    for DATASET in ${DATASETS}; do
        for TRAIN_MODE in ${TRAIN_MODES}; do
            for MODALITY in ${MODALITY_CONFIGS[$DATASET]}; do
                n=$((n+1))
                if [ "${SUBMIT}" = "1" ]; then
                    sbatch --time="${WALLTIME}" \
                        --export=ALL,MODEL="${MODEL}",DATASET="${DATASET}",TRAIN_MODE="${TRAIN_MODE}",MODALITY="${MODALITY}" \
                        sh/rsfm_sft/rsfm_sft_job.sh >/dev/null
                    echo "  [$n] submitted ${MODEL} ${DATASET} ${MODALITY}"
                else
                    echo "  [$n] ${MODEL} ${DATASET} ${TRAIN_MODE} ${MODALITY}"
                fi
            done
        done
    done
done
echo
echo "total: ${n} jobs, --time=${WALLTIME} (SUBMIT=${SUBMIT})"
