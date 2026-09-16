#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/train_sft/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# kn101 has failing GPU memory: every job scheduled there since 2026-09-15 died
# with "CUDA error: uncorrectable ECC error encountered", while jobs on every
# other node succeeded. Slurm still lists it as healthy. Remove if repaired.
#SBATCH --exclude=kn101

# Expected env vars (set by train_sft_all.sh):
#   DATASET, MODEL, TRAIN_MODE, MODALITY_ENTRY, LR, WD, TRAIN_SPLIT
#   EPOCHS      training epochs (default 24)
#   DINO_ARMS   which init arms to run: "1 0" (default, dino then random),
#               "1" for dino only, "0" for random only. Each arm is a full
#               training run, so "1 0" doubles the walltime -- biomassters at
#               ~14.4 min/epoch needs one arm per job beyond ~24 epochs.

source sh/env.sh
export TQDM_DISABLE=1

MODALITIES="${MODALITY_ENTRY//+/ }"
MODALITY_KEY="${MODALITY_ENTRY}"
# DFC2020 has two incompatible splits (roi = ROI-disjoint 10-class,
# cobench = Copernicus-Bench 3156/986/986 8-class). Keep their results in
# separate files so the dedup below never treats one as satisfying the other.
CSV_SUFFIX=""
if [ "${DATASET}" = "dfc2020" ]; then
    CSV_SUFFIX="_cobench"
fi
# BACKBONE_LR_MULT: multiplier on the backbone lr in fft mode (train_sft.py
# default 0.1 = the historical hardcoded lr/10; 1.0 = symmetric). Anything other
# than 0.1 adds a results column, so train_sft.py writes it to its own *_bblr
# file -- mirror that routing here or the dedup check reads the wrong file.
BBLR_ARG=""
if [ -n "${BACKBONE_LR_MULT:-}" ]; then
    BBLR_ARG="--backbone_lr_mult ${BACKBONE_LR_MULT}"
    [ "${BACKBONE_LR_MULT}" != "0.1" ] && CSV_SUFFIX="${CSV_SUFFIX}_bblr"
fi
RESULTS_CSV="res/train_sft/${DATASET}${CSV_SUFFIX}.csv"

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
    dfc2020)
        # Copernicus-Bench appends a UPerNet decoder (+ auxiliary FCN) for
        # segmentation; our default linear head is a 1x1 conv over the 16x16
        # token grid + 16x bilinear upsample, which loses the multi-scale
        # detail segmentation depends on. Set DECODER=upernet to match their
        # setup. DECODER_TAG is part of the results-CSV key, so upernet rows
        # never suppress the linear ones.
        if [ "${DECODER:-linear}" = "upernet" ]; then
            DECODER_TAG="upernet"
            EXTRA_ARGS="--decoder_type upernet"
        fi
        ;;
esac

TRAIN_SPLIT="${TRAIN_SPLIT:-split1}"

echo "Running: model=${MODEL} dataset=${DATASET} train_mode=${TRAIN_MODE} modalities=${MODALITIES} lr=${LR} wd=${WD} train_split=${TRAIN_SPLIT}"

for USE_DINO in ${DINO_ARMS:-1 0}; do
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
    # Historical rows carry two extra columns (tz_lora_rank,
    # tz_modality_specific_layer_augmenter) that were dropped when LoRA was
    # removed; the optional group matches both the old and new schema so
    # completed runs are still recognised and not re-submitted.
    # The epoch count is part of the key: a 24-epoch row must NOT suppress a
    # 48-epoch rerun of the same config. Layout after lr,wd is
    #   trainable_params, epoch, test, val, metric_name, checkpoint, global_rep
    # so epoch is field 2 of the 7 that used to be skipped wholesale.
    # BACKBONE_LR_MULT != 0.1 routes train_sft.py to a separate *_bblr.csv with one
    # extra trailing column, so the dedup pattern must not anchor on train_split$.
    _TAIL='\r?$'
    if [ -n "${BACKBONE_LR_MULT:-}" ] && [ "${BACKBONE_LR_MULT}" != "0.1" ]; then
        _TAIL=",${BACKBONE_LR_MULT}\r?$"
    fi
    if grep -qP "^${DATASET},${MODEL},${MODALITY_KEY},${TRAIN_MODE},([^,]*,[^,]*,)?${LR},${WD},[^,]+,${EPOCHS:-24},([^,]+,){5}${DINO_VAL},${T},\Q${DECODER_TAG}\E,${TRAIN_AUG},${TRAIN_SPLIT}${_TAIL}" "${RESULTS_CSV}" 2>/dev/null; then
        echo "  → dino_init=${DINO_VAL} train_split=${TRAIN_SPLIT} already in results, skipping"
        continue
    fi

    echo "--- use_dino=${USE_DINO} ---"
    python -u train_sft.py \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --modalities ${MODALITIES} \
        --train_mode ${TRAIN_MODE} \
        --epochs ${EPOCHS:-24} \
        --lr ${LR} \
        --weight_decay ${WD} \
        --train_aug ${TRAIN_AUG} \
        --train_split ${TRAIN_SPLIT} \
        ${BBLR_ARG} \
        ${EXTRA_ARGS} \
        ${DINO_FLAG}
done
