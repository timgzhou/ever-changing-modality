#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/halluc_analyze/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# See sh/projector_probe_job.sh for why kn101 is excluded (dead GPU memory,
# still listed as healthy by Slurm).
#SBATCH --exclude=kn101

# One arm of the hallucination-quality analysis, as a batch job.
# sh/halluc_analyze_dfc2020.sh runs the same command inline, which only works
# from inside an existing allocation; this wrapper lets the arms be submitted
# and run in parallel.
#
# Required: CKPT, OUT_DIR
# Optional: DATASET N_BATCHES BATCH_SIZE N_VIS SEED
#
# 1h walltime is generous: the model runs under no_grad over N_BATCHES batches
# and the rank transform is a double-argsort on top. A 20-batch dfc2020 arm at
# batch_size 16 measured 39s end to end, figures included.
set -euo pipefail
source sh/env.sh
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/halluc_analyze

: "${CKPT:?}"; : "${OUT_DIR:?}"
[ -f "${CKPT}" ] || { echo "[error] checkpoint not found: ${CKPT}"; exit 1; }
mkdir -p "${OUT_DIR}"

python -u analysis/analyze_hallucination_correlation.py \
    --checkpoint "${CKPT}" \
    --dataset "${DATASET:-dfc2020}" \
    --n_batches "${N_BATCHES:-20}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --n_vis "${N_VIS:-4}" \
    --seed "${SEED:-0}" \
    --out_dir "${OUT_DIR}" 2>&1 | tee "${OUT_DIR}/analyze.log"
