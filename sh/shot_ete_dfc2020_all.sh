#!/bin/bash
# DeluluNet (shot_ete) on DFC2020, Copernicus-Bench split — all 6 transfer directions.
#
# Each job takes a single-modality stage-0 teacher and adds a second modality
# using UNLABELED data (transfer / peeking / addition). The teachers are the
# best-by-val single-modality SFT checkpoints from
# res/train_sft/dfc2020_cobench.csv; val is a reliable selector on this split
# (test-vs-val Spearman rho = 0.99).
#
# Directions (teacher -> added modality):
#   s1       -> +s2_rgb     the high-headroom case: SAR teacher (52.5 test mIoU)
#   s1       -> +s2_norgb   gains optical (66.7 alone) -- most useful addition
#   s2_rgb   -> +s1         optical teacher gains SAR -- expected near-null
#   s2_rgb   -> +s2_norgb   optical gains the other optical bands
#   s2_norgb -> +s1         strongest teacher gains SAR -- expected near-null
#   s2_norgb -> +s2_rgb     strongest teacher gains RGB
#
# The DECODER must match the teacher: EvanSegmenter.from_checkpoint restores
# decoder_type from the checkpoint config, so pointing at an upernet teacher
# gives an upernet student automatically.
#
# SHOT hyperparameters are carried over from the biomassters s1->s2 best config
# and are NOT tuned for DFC2020. Treat this as the first pass, then sweep.
#
# Usage:
#   bash sh/shot_ete_dfc2020_all.sh                 # upernet teachers (default)
#   DECODER=linear bash sh/shot_ete_dfc2020_all.sh  # linear teachers
#   EPOCHS=40 bash sh/shot_ete_dfc2020_all.sh

DECODER="${DECODER:-upernet}"

# Teachers resolved from artifacts/sft_teachers.json (regenerate with
# python res/train_sft/sft_best.py). TEACHER_SPLIT=split1 is required: stage 1
# uses train2 as its UNLABELED pool, so a `full` teacher -- supervised on
# train1+train2 -- has already seen that pool with labels and the comparison
# leaks. See res/train_sft/README_dfc2020.md.
TEACHERS_JSON="artifacts/sft_teachers.json"
TEACHER_SPLIT="${TEACHER_SPLIT:-split1}"
MODEL="${MODEL:-evan_base}"
teacher_for () {
    jq -r ".\"dfc2020_cobench/$1/${MODEL}/${DECODER}/${TEACHER_SPLIT}\".checkpoint // empty" \
        "${TEACHERS_JSON}"
}

EPOCHS="${EPOCHS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# lambda_latent sweep. The latent loss was inflated by embed_dim (768) until
# 2026-08-20, so the carried-over 0.361 was tuned at a scale where latent was
# ~98% of the signal; at the fixed scale it contributes ~5%. These three points
# bracket "latent ~5% / ~25% / ~35% of total" at the corrected scale.
# Set LAMBDA_LATENTS="0.36" to pin a single value.
LAMBDA_LATENTS="${LAMBDA_LATENTS:-0.36 2.1 3.5}"

n=0
for START in s1 s2_rgb s2_norgb; do
  for NEW in s1 s2_rgb s2_norgb; do
    [ "${START}" = "${NEW}" ] && continue
    TEACHER=$(teacher_for "${START}")
    if [ ! -f "${TEACHER}" ]; then
        echo "[error] teacher checkpoint not found: ${TEACHER}"; exit 1
    fi
    for LL in ${LAMBDA_LATENTS}; do
      echo "Submitting: ${START} -> +${NEW}  (${DECODER}, lambda_latent=${LL})"
      sbatch --export=ALL,START="${START}",NEW="${NEW}",TEACHER="${TEACHER}",DECODER="${DECODER}",EPOCHS="${EPOCHS}",BATCH_SIZE="${BATCH_SIZE}",LAMBDA_LATENT="${LL}" \
          sh/shot_ete_dfc2020_job.sh
      n=$((n+1))
    done
  done
done
echo "submitted ${n} shot_ete jobs (${DECODER} teachers)"
