#!/bin/bash
#SBATCH --time=0:40:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/viz_biomassters_preds/%j.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Visualize BioMassters AGB predictions: 2 input panels + 5 prediction panels
# (2 SFT, 3 Delulu paths) + 1 target, for the first N test samples.
#
# Usage:  bash sh/viz_biomassters_preds.sh              # s2 -> s1
#         DIRECTION=s1s2 bash sh/viz_biomassters_preds.sh
#         DIRECTION=both bash sh/viz_biomassters_preds.sh
#         NUM_SAMPLES=6 bash sh/viz_biomassters_preds.sh
#         INDICES="10 11 12 13" bash sh/viz_biomassters_preds.sh

source sh/env.sh
export TQDM_DISABLE=1
# T=12 folds the time axis into the batch dim, so activations are heavy.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/viz_biomassters_preds figs

# Both directions share one stage-0 teacher per modality — every SHOT run in
# each family records the same stage0_checkpoint. Each SFT_* below is the exact
# teacher its SHOT runs were distilled from, so the peeking/transfer columns are
# compared against the model they actually started from.
# NOTE: a better S2 teacher exists (…065543, test RMSE 42.91 vs …075836's ~47.7),
# but swapping it in would misrepresent the SHOT runs' starting point.
SFT_S2="${SFT_S2:-checkpoints/sft_evan_base_biomassters_s2_fft_lr0.0005_20260725_075836.pt}"
SFT_S1="${SFT_S1:-checkpoints/sft_evan_base_biomassters_s1_fft_lr0.0005_20260725_075947.pt}"

# SHOT checkpoints, both evan_base / T=12 / upernet+relu, modality pairs verified
# by reading supported_modalities out of the weights.
#   s2s1: best all-round of 18 runs — best val peeking (-43.13), addition -11.40
#   s1s2: best all-round of 17 runs — best val addition (-8.71), transfer -22.30
DELULU_S2S1="${DELULU_S2S1:-checkpoints/delulunet_biomassters_s2s1_peeking_rank1_seed1.pt}"
DELULU_S1S2="${DELULU_S1S2:-checkpoints/delulunet_biomassters_s1s2_addition_rank2_seed2.pt}"

# MUST match what the checkpoints were trained with.
NUM_TIME_STEPS="${NUM_TIME_STEPS:-12}"
DIRECTION="${DIRECTION:-s2s1}"

run_direction() {
    local start="$1" new="$2" delulu="$3" sft_start="$4" sft_new="$5"

    for f in "${delulu}" "${sft_start}" "${sft_new}"; do
        if [ ! -f "${f}" ]; then
            echo "[error] checkpoint not found: ${f}"
            exit 1
        fi
    done

    echo ""
    echo "=== viz biomassters | ${start} -> ${new} | T=${NUM_TIME_STEPS} ==="
    echo "    delulu:    ${delulu}"
    echo "    sft_start: ${sft_start}"
    echo "    sft_new:   ${sft_new}"

    python -u viz_biomassters_preds.py \
        --starting_modality "${start}" \
        --new_modality      "${new}" \
        --delulu            "${delulu}" \
        --sft_start         "${sft_start}" \
        --sft_new           "${sft_new}" \
        --num_samples       "${NUM_SAMPLES:-8}" \
        --num_time_steps    "${NUM_TIME_STEPS}" \
        ${INDICES:+--indices ${INDICES}} \
        --num_workers 2 \
        --out "figs/biomassters_${start}_to_${new}.png"
}

case "${DIRECTION}" in
    s2s1) run_direction s2 s1 "${DELULU_S2S1}" "${SFT_S2}" "${SFT_S1}" ;;
    s1s2) run_direction s1 s2 "${DELULU_S1S2}" "${SFT_S1}" "${SFT_S2}" ;;
    both)
        run_direction s2 s1 "${DELULU_S2S1}" "${SFT_S2}" "${SFT_S1}"
        run_direction s1 s2 "${DELULU_S1S2}" "${SFT_S1}" "${SFT_S2}"
        ;;
    *)
        echo "[error] DIRECTION must be one of: s2s1, s1s2, both (got '${DIRECTION}')"
        exit 1
        ;;
esac
