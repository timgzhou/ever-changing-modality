#!/bin/bash
#SBATCH --time=5:59:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/shot_ete_default/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Test shot_ete on BioMassters with DEFAULT hyperparameters (the train_delulu
# submission defaults now baked into shot_ete.py's argparser) and modality
# protection DISABLED (--unprotect_starting_mod).
#
# Deliberately passes NO SHOT hparam flags (modality_dropout, token_mask_ratio,
# lambda_*, labeled_frequency, ...) so the argparser defaults apply:
#   token_mask_ratio=0.4  labeled_start_fraction=0  warmup_epochs=4
#   modality_dropout=0.3  labeled_frequency=0.3  lambda_*=1.0
#
# Usage:  sbatch sh/shot_ete_biomassters_default.sh
#         TEACHER=path/to/f0.pt NEW_MOD=s1 EPOCHS=64 sbatch sh/shot_ete_biomassters_default.sh

source sh/env.sh
export TQDM_DISABLE=1
# T=12 folds the time axis into the batch dim (effective batch = batch_size * T),
# so SHOT (student + frozen teacher + decoders + projectors) is memory-heavy.
# Reduce fragmentation and keep batch_size modest.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/shot_ete_default

# Best-val S2 teacher (UperNet+relu, T=12): val RMSE 42.91 t/ha.
TEACHER="${TEACHER:-checkpoints/sft_evan_base_biomassters_s2_fft_lr0.0005_20260725_065543.pt}"
NEW_MOD="${NEW_MOD:-s1}"
NUM_TIME_STEPS="${NUM_TIME_STEPS:-12}"   # MUST match the teacher's T
EPOCHS="${EPOCHS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"            # small: T=12 -> effective batch = BATCH_SIZE*T

if [ ! -f "${TEACHER}" ]; then
    echo "[error] teacher checkpoint not found: ${TEACHER}"
    exit 1
fi

echo "=== shot_ete DEFAULT params | biomassters -> ${NEW_MOD} | T=${NUM_TIME_STEPS} ==="
echo "    teacher: ${TEACHER}"

python -u shot_ete.py \
    --dataset biomassters \
    --new_mod_group "${NEW_MOD}" \
    --stage0_checkpoint "${TEACHER}" \
    --active_losses latent prefusion distill ce \
    --results_csv "res/delulu/biomassters_default_${NEW_MOD}.csv" \
    --wandb_project "delulu-biomassters-default" \
    --unprotect_starting_mod \
    --num_time_steps "${NUM_TIME_STEPS}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 4
