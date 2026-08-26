#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/shot_ete_dfc2020/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# One shot_ete direction on DFC2020 / Copernicus-Bench split.
# Expected env vars (set by sh/shot_ete_dfc2020_all.sh):
#   START, NEW, TEACHER, DECODER, EPOCHS, BATCH_SIZE

source sh/env.sh
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Copernicus-Bench official 3156/986/986 split, 8 classes. MUST match the split
# the teacher was trained on, or the head size and label mapping disagree.
export DFC2020_SPLIT=cobench

mkdir -p logs/shot_ete_dfc2020 checkpoints res/delulu

# SHOT hyperparameters from the biomassters s1->s2 best config (untuned here).
LR="${LR:-0.0001569391767106977}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3.351617951860976e-05}"
MODALITY_DROPOUT="0.3"
MODALITY_DROPOUT_STARTMOD="0.33189226742900324"
MODALITY_DROPOUT_NEWMOD="0.17068517311514753"
LABELED_FREQUENCY="0.23002477810989655"
LABELED_START_FRACTION="0"
# NOTE: the latent loss was inflated by embed_dim (768) until 2026-08-20 --
# see _compute_latent_loss in shot.py. Every lambda below was tuned against
# that broken scale, where latent was ~98% of the total signal. At the fixed
# scale the same value leaves latent at ~5%, so LAMBDA_LATENT is swept
# explicitly (override from the launcher).
LAMBDA_LATENT="${LAMBDA_LATENT:-0.3613664751387723}"
LAMBDA_PREFUSION="0.6430194633931678"
LAMBDA_DISTILL="0.15374988356364516"
TOKEN_MASK_RATIO="0.40414477259411485"
PROTECT_LRM="0.0"
SEED="${SEED:-0}"
# teacher (default) | random. NOTE: the distillation baseline's apparent
# 2.5-5.9 mIoU advantage for random init was a CONFOUND -- init_student_from_teacher
# did not forward --decoder_type, so its teacher-init arm silently ran a linear
# head while the random arm ran upernet (a 39,335,680-param difference). Delulu
# was never affected: EvanSegmenter.from_checkpoint restores decoder_type, and
# both arms share one model object. Measured delulu effect is ~null
# (transfer -0.03 mIoU, n=6).
STUDENT_INIT="${STUDENT_INIT:-teacher}"

# --- Ablation: distillation-only ------------------------------------------
# ACTIVE_LOSSES=distill isolates the distill term.
#
# IMPORTANT: do NOT also zero modality_dropout/token_mask_ratio. Masking is not
# a loss term, it is what routes the intermediate projectors into the forward
# pass: mask_input() builds masked_mod_features via
#     torch.where(full_mask_expanded, projected_seq, features)
# so with an all-False mask the projector output is computed but discarded and
# receives ZERO gradient. Training then never exercises the hallucination path,
# while `transfer` at eval time depends on it entirely -- transfer collapses to
# 7-31 mIoU while addition stays ~57-63. That is an artifact of the ablation,
# not a property of the distill loss.
ACTIVE_LOSSES="${ACTIVE_LOSSES:-latent prefusion distill ce}"
if [ "${DISTILL_ONLY:-0}" = "1" ]; then
    ACTIVE_LOSSES="distill"
    LABELED_FREQUENCY="0"        # no labeled batch mixing -> no CE path at all
    LAMBDA_DISTILL="1.0"
    # modality_dropout / token_mask_ratio intentionally left at their defaults
fi

RUN_TAG="${START}_to_${NEW}_${DECODER}_ll${LAMBDA_LATENT}_init${STUDENT_INIT}_seed${SEED}"
[ "${DISTILL_ONLY:-0}" = "1" ] && RUN_TAG="${START}_to_${NEW}_${DECODER}_distillonly_init${STUDENT_INIT}_seed${SEED}"

echo "=== shot_ete | dfc2020 (cobench) ${START} -> +${NEW} | decoder=${DECODER} ==="
echo "    teacher: ${TEACHER}"
echo "    lr=${LR} wd=${WEIGHT_DECAY} epochs=${EPOCHS} bs=${BATCH_SIZE}"

python -u shot_ete.py \
    --dataset dfc2020 \
    --new_mod_group "${NEW}" \
    --stage0_checkpoint "${TEACHER}" \
    --active_losses ${ACTIVE_LOSSES} \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --modality_dropout "${MODALITY_DROPOUT}" \
    --modality_dropout_startmod "${MODALITY_DROPOUT_STARTMOD}" \
    --modality_dropout_newmod "${MODALITY_DROPOUT_NEWMOD}" \
    --labeled_frequency "${LABELED_FREQUENCY}" \
    --labeled_start_fraction "${LABELED_START_FRACTION}" \
    --lambda_latent "${LAMBDA_LATENT}" \
    --lambda_prefusion "${LAMBDA_PREFUSION}" \
    --lambda_distill "${LAMBDA_DISTILL}" \
    --token_mask_ratio "${TOKEN_MASK_RATIO}" \
    --protect_lrm "${PROTECT_LRM}" \
    --latent_masked_only \
    --unprotect_starting_mod \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 4 \
    --seed "${SEED}" \
    --student_init "${STUDENT_INIT}" \
    --save_checkpoint \
    --checkpoint_dir checkpoints \
    --checkpoint_name "delulunet_dfc2020_${RUN_TAG}" \
    --results_csv "res/delulu/dfc2020_cobench_${DECODER}.csv" \
    --wandb_project "delulu-dfc2020-cobench"
