#!/bin/bash
#SBATCH --time=0:40:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/viz_dfc2020/%j.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

# Visualize DFC2020 s2_rgb -> s2_norgb predictions for the first 4 test samples:
# 2 input panels + 5 prediction panels (2 SFT, 3 Delulu paths) + 1 target.
#
# Usage:  sbatch sh/viz_dfc2020.sh
#         NUM_SAMPLES=8 sbatch sh/viz_dfc2020.sh

source sh/env.sh
export TQDM_DISABLE=1
mkdir -p logs/viz_dfc2020 figs

# DFC2020 lives in a 10GB zip at the repo root and is not extracted anywhere yet.
# get_dfc2020_loaders defaults to this path; extract once into $SLURM_TMPDIR-backed
# local storage would be lost between jobs, so extract into datasets/ instead.
DATA_DIR="datasets/GFM-Bench/DFC2020"
if [ ! -f "${DATA_DIR}/DFC2020/metadata.csv" ]; then
    echo "=== Extracting DFC2020.zip -> ${DATA_DIR} (one-time, ~10GB) ==="
    mkdir -p "${DATA_DIR}"
    unzip -q -n DFC2020.zip -d "${DATA_DIR}"
fi

# NOTE: res/delulu/hptuned_apr21.csv is unreliable for identifying modality
# direction — timestamped checkpoint names collided across runs, so some rows
# claiming s2_rgb->s2_norgb point at models for entirely different pairs (the
# previously-chosen 0420_1027 is really an s1->s2 evan_large model). The
# checkpoints below were verified by reading evan_config.supported_modalities
# out of the weights. Of the 68 dfc2020 SHOT checkpoints on disk, exactly 8
# genuinely carry {s2_rgb, s2_norgb}, and all 8 are evan_base.
#
# Best mean-val across the three paths among those 8:
#   val transfer/peeking/addition = 52.28 / 57.82 / 55.99
#   test                          = 48.90 / 44.67 / 49.78
DELULU="${DELULU:-checkpoints/delulu-checkpoints/delulunet_dfc2020_0420_0752.pt}"

# Stage-0 SFT baselines, evan_base to match the SHOT model's arch.
# s2_rgb:   val 55.71 / test 48.76
# s2_norgb: val 56.22 / test 43.73 — picked over the 56.90-val run, whose test
#           mIoU drops to 39.87 and would understate the baseline in a figure.
SFT_START="${SFT_START:-checkpoints/delulu-checkpoints/sft_evan_base_dfc2020_s2_rgb_fft_lr0.0005_20260420_184627.pt}"
SFT_NEW="${SFT_NEW:-checkpoints/delulu-checkpoints/sft_evan_base_dfc2020_s2_norgb_fft_lr0.0005_20260424_002258.pt}"

for f in "${DELULU}" "${SFT_START}" "${SFT_NEW}"; do
    if [ ! -f "${f}" ]; then
        echo "[error] checkpoint not found: ${f}"
        exit 1
    fi
done

echo "=== viz dfc2020 | s2_rgb -> s2_norgb ==="
echo "    delulu:    ${DELULU}"
echo "    sft_start: ${SFT_START}"
echo "    sft_new:   ${SFT_NEW}"

# Sample choice matters a lot on DFC2020: ~50% of the average tile is masked
# (savanna), but tiles with NO masking are uniform single-class ones where every
# model scores mIoU 100. Selection therefore requires class diversity too.
# Set INDICES="0 1 2 3" to force specific tiles.
python -u viz_dfc2020.py \
    --delulu      "${DELULU}" \
    --sft_start   "${SFT_START}" \
    --sft_new     "${SFT_NEW}" \
    --num_samples "${NUM_SAMPLES:-4}" \
    --max_ignore  "${MAX_IGNORE:-0.35}" \
    --min_classes "${MIN_CLASSES:-3}" \
    --scan_limit  "${SCAN_LIMIT:-600}" \
    ${INDICES:+--indices ${INDICES}} \
    --num_workers 2 \
    --out figs/dfc2020_s2rgb_to_s2norgb.png

# source sh/vis_dfc2020.sh