#!/bin/bash
# Launcher: SFT sweep for BioMassters (temporal S1/S2 AGB regression).
# Submits one SLURM job per modality+lr+wd combo; the job script itself sweeps
# use_dino in {1,0}. Mirrors sh/train_sft/train_sft_all.sh but scoped to
# biomassters, adding NUM_TIME_STEPS to the per-job export.
#
# Usage: bash sh/train_sft/train_sft_biomassters.sh

DATASET='biomassters'
MODEL='evan_base'
TRAIN_MODE='fft'

# Stage-0 oracles: single modality (s2, s1). Combined (s2+s1 / s1+s2) is the
# addition upper bound. '+' separator -> train_sft_job.sh splits into --modalities.
MODALITIES='s2 s1 s2+s1'

LRS=('0.0005' '0.0001')
WDS=('0.01' '0.0')

# Temporal pooling window (<=12). BioMassters mean-pools features over this many
# most-recent timesteps inside the model. Full-year signal (12) matches the
# published protocol; keep in sync with the shot_ete / e2e scripts.
NUM_TIME_STEPS="${NUM_TIME_STEPS:-12}"

RESULTS_CSV="res/train_sft/${DATASET}.csv"
mkdir -p logs/train_sft

for MODALITY_ENTRY in ${MODALITIES}; do
    for LR in "${LRS[@]}"; do
        for WD in "${WDS[@]}"; do
            MODALITY_KEY="${MODALITY_ENTRY}"
            # Skip only when BOTH dino variants are already in the results CSV.
            # The key includes num_time_steps and the decoder tag, so runs at a
            # different T (or with the old linear head) are not treated as done.
            # Must stay in sync with the pattern in train_sft_job.sh.
            DINO_PAT="^${DATASET},${MODEL},${MODALITY_KEY},${TRAIN_MODE},[^,]+,[^,]+,${LR},${WD},([^,]+,){7}"
            TAIL="${NUM_TIME_STEPS},\Qupernet+relu\E$"
            grep -qP "${DINO_PAT}True,${TAIL}"  "${RESULTS_CSV}" 2>/dev/null && DINO_TRUE=1  || DINO_TRUE=0
            grep -qP "${DINO_PAT}False,${TAIL}" "${RESULTS_CSV}" 2>/dev/null && DINO_FALSE=1 || DINO_FALSE=0
            if [ "$DINO_TRUE" -ge 1 ] && [ "$DINO_FALSE" -ge 1 ]; then
                echo "Skipping (both dino variants done): modality=${MODALITY_ENTRY} lr=${LR} wd=${WD}"
                continue
            fi
            echo "Submitting: dataset=${DATASET} model=${MODEL} train_mode=${TRAIN_MODE} modality=${MODALITY_ENTRY} lr=${LR} wd=${WD} T=${NUM_TIME_STEPS}"
            sbatch --export=ALL,DATASET="${DATASET}",MODEL="${MODEL}",TRAIN_MODE="${TRAIN_MODE}",MODALITY_ENTRY="${MODALITY_ENTRY}",LR="${LR}",WD="${WD}",NUM_TIME_STEPS="${NUM_TIME_STEPS}" \
                sh/train_sft/train_sft_job.sh
        done
    done
done

# bash sh/train_sft/train_sft_biomassters.sh
# Override temporal window:  NUM_TIME_STEPS=12 bash sh/train_sft/train_sft_biomassters.sh
