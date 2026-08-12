#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/train_sft/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Expected env vars (set by train_sft_all.sh):
#   DATASET, MODEL, TRAIN_MODE, MODALITY_ENTRY, LR, WD, TRAIN_SPLIT

source sh/env.sh
export TQDM_DISABLE=1

MODALITIES="${MODALITY_ENTRY//+/ }"
MODALITY_KEY="${MODALITY_ENTRY}"
RESULTS_CSV="res/train_sft/${DATASET}.csv"

# BioMassters is temporal: pool features over this many timesteps (<=12).
# It is also regression on non-negative AGB, so use the PANGAEA RegUPerNet-style
# multi-scale decoder with a ReLU-clamped output.
# T and DECODER_TAG also form part of the results-CSV key below, so a run at a
# different T (or with a different head) is not suppressed by an older row.
EXTRA_ARGS=""
T=10                                    # train_sft.py --num_time_steps default
DECODER_TAG="linear"                    # dense-head default; "cls" for classification
case "${DATASET}" in
    biomassters)
        T="${NUM_TIME_STEPS:-12}"
        DECODER_TAG="upernet+relu"
        EXTRA_ARGS="--num_time_steps ${T} --decoder_type upernet --relu_output"
        ;;
    benv2|eurosat)
        DECODER_TAG="cls"
        ;;
esac

TRAIN_SPLIT="${TRAIN_SPLIT:-split1}"

echo "Running: model=${MODEL} dataset=${DATASET} train_mode=${TRAIN_MODE} modalities=${MODALITIES} lr=${LR} wd=${WD} train_split=${TRAIN_SPLIT}"

for USE_DINO in 1 0; do
    DINO_VAL="True"
    DINO_FLAG="--use_dino_weights"
    if [ "${USE_DINO}" = "0" ]; then
        DINO_VAL="False"
        DINO_FLAG=""
    fi

    # Key must include T, the decoder tag, train_aug and train_split, else an
    # older row (different temporal window / linear head / augmentation / data
    # split) wrongly suppresses this run. train_split matters most here: a
    # 'full' run is identical to its 'split1' counterpart in EVERY other
    # column, so without it the split1 row would always suppress the full run.
    # Every row carries train_split since the migration, so it is required (not
    # optional like the legacy trailing columns).
    TRAIN_AUG="${TRAIN_AUG:-none}"
    if grep -qP "^${DATASET},${MODEL},${MODALITY_KEY},${TRAIN_MODE},[^,]+,[^,]+,${LR},${WD},([^,]+,){7}${DINO_VAL},${T},\Q${DECODER_TAG}\E,${TRAIN_AUG},${TRAIN_SPLIT}\r?$" "${RESULTS_CSV}" 2>/dev/null; then
        echo "  → dino_init=${DINO_VAL} train_split=${TRAIN_SPLIT} already in results, skipping"
        continue
    fi

    echo "--- use_dino=${USE_DINO} ---"
    python -u train_sft.py \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --modalities ${MODALITIES} \
        --train_mode ${TRAIN_MODE} \
        --epochs 24 \
        --lr ${LR} \
        --weight_decay ${WD} \
        --train_aug ${TRAIN_AUG} \
        --train_split ${TRAIN_SPLIT} \
        ${EXTRA_ARGS} \
        ${DINO_FLAG}
done
