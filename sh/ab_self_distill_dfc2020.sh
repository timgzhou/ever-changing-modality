#!/bin/bash
# Paired A/B: does --self_distill_addition close the addition gap on DFC2020?
#
# Hypothesis (see the addition-gap analysis): on unlabeled token-masking batches
# the only task-shaped target is the frozen UNIMODAL teacher, whose answer IS the
# peeking answer -- so distillation penalises the fused two-modality prediction
# wherever it correctly disagrees. The flag instead distils the new-modality
# heads against the student's own addition path (stop-grad, no labels).
#
# Design: everything except the flag is held fixed at the config that produced
# res/delulu/dfc2020_unimodal_pairs.csv, so the arms are directly comparable to
# those numbers AND to each other. 3 seeds per arm, because seed-to-seed sd on
# DFC2020 transfer is ~1.09 mIoU (max 1.85) -- a single pair could not separate
# a real effect from seed noise.
#
# 4 directions x 2 arms x 3 seeds = 24 jobs.
#
# Directions are ordered so the two cases where the gap is actually visible run
# first: s2->s1 (peeking 61.81 > addition 61.59) and s2_norgb->s2_rgb
# (63.27 > 62.34). s1->s2 and s2_rgb->s2_norgb are controls where addition
# already wins; the flag should not hurt them.
#
# Usage:  bash sh/ab_self_distill_dfc2020.sh          # submit
#         DRYRUN=1 bash sh/ab_self_distill_dfc2020.sh # print only
set -uo pipefail

TEACHERS_JSON="artifacts/sft_teachers.json"
RESULTS_CSV="res/delulu/dfc2020_self_distill_ab.csv"
SEEDS="${SEEDS:-0 1 2}"

# Exact config behind res/delulu/dfc2020_unimodal_pairs.csv.
export LR=0.0001569391767106977
export WEIGHT_DECAY=3.351617951860976e-05
export MODALITY_DROPOUT=0.3
export MODALITY_DROPOUT_STARTMOD=0.33189226742900324
export MODALITY_DROPOUT_NEWMOD=0.17068517311514753
export LABELED_FREQUENCY=0.23002477810989655
export LABELED_START_FRACTION=0
export LAMBDA_LATENT=0.3613664751387723
export LAMBDA_PREFUSION=0.6430194633931678
export LAMBDA_DISTILL=0.15374988356364516
export TOKEN_MASK_RATIO=0.40414477259411485
export PROTECT_LRM=0.0
export EPOCHS=64
export BATCH_SIZE=8

n=0
for pair in "s2 s1" "s2_norgb s2_rgb" "s1 s2" "s2_rgb s2_norgb"; do
    set -- $pair; START="$1"; NEW="$2"
    KEY="dfc2020_cobench/${START}/evan_base/upernet/split1"
    TEACHER=$(jq -r ".\"${KEY}\".checkpoint // empty" "${TEACHERS_JSON}")
    if [ -z "${TEACHER}" ] || [ ! -f "${TEACHER}" ]; then
        echo "  [skip] ${START}->+${NEW}: no teacher for ${KEY}"; continue
    fi
    for SEED in ${SEEDS}; do
        for ARM in off on; do
            n=$((n+1))
            EXPORTS="ALL,DATASET=dfc2020,START=${START},NEW=${NEW},TEACHER=${TEACHER}"
            EXPORTS="${EXPORTS},EPOCHS=${EPOCHS},BATCH_SIZE=${BATCH_SIZE},SEED=${SEED}"
            EXPORTS="${EXPORTS},LR=${LR},WEIGHT_DECAY=${WEIGHT_DECAY}"
            EXPORTS="${EXPORTS},MODALITY_DROPOUT=${MODALITY_DROPOUT}"
            EXPORTS="${EXPORTS},MODALITY_DROPOUT_STARTMOD=${MODALITY_DROPOUT_STARTMOD}"
            EXPORTS="${EXPORTS},MODALITY_DROPOUT_NEWMOD=${MODALITY_DROPOUT_NEWMOD}"
            EXPORTS="${EXPORTS},LABELED_FREQUENCY=${LABELED_FREQUENCY}"
            EXPORTS="${EXPORTS},LABELED_START_FRACTION=${LABELED_START_FRACTION}"
            EXPORTS="${EXPORTS},LAMBDA_LATENT=${LAMBDA_LATENT}"
            EXPORTS="${EXPORTS},LAMBDA_PREFUSION=${LAMBDA_PREFUSION}"
            EXPORTS="${EXPORTS},LAMBDA_DISTILL=${LAMBDA_DISTILL}"
            EXPORTS="${EXPORTS},TOKEN_MASK_RATIO=${TOKEN_MASK_RATIO}"
            EXPORTS="${EXPORTS},PROTECT_LRM=${PROTECT_LRM}"
            EXPORTS="${EXPORTS},RESULTS_CSV=${RESULTS_CSV}"
            # Metrics-only ablation: 24 checkpoints would be ~10 G and scratch is
            # quota-bound. The CSV row carries everything this experiment needs.
            EXPORTS="${EXPORTS},SAVE_CHECKPOINT=0"
            [ "${ARM}" = "on" ] && EXPORTS="${EXPORTS},SELF_DISTILL_ADDITION=1"
            if [ "${DRYRUN:-0}" = "1" ]; then
                echo "[$n] ${START}->+${NEW}  seed=${SEED}  self_distill=${ARM}"
            else
                jid=$(sbatch --parsable --export="${EXPORTS}" sh/train_delulu_job.sh)
                echo "[$n] submitted ${jid}  ${START}->+${NEW}  seed=${SEED}  self_distill=${ARM}"
            fi
        done
    done
done
echo "total: ${n} jobs  ->  ${RESULTS_CSV}"
