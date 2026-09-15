#!/bin/bash
# Train checkpoints for the HALLUCINATION-QUALITY analysis (not a downstream A/B).
#
# The recon A/B (sh/ab_recon_dfc2020.sh) answered "does a cosine term improve
# mIoU" -- answer: no, and at weight 1.0 it actively hurts. It did NOT answer
# "does a cosine term hallucinate better", which is the actual claim the term
# was introduced for. Those are separate questions: a projector can reconstruct
# the target modality better while the fused task head fails to exploit it.
#
# So: retrain with SAVE_CHECKPOINT=1 and feed the checkpoints to
# analysis/analyze_hallucination_correlation.py, which reports
#   patch_pearson    -- per-patch r across feature dim (leaves shared geometry in)
#   batch_pearson    -- r across the SAMPLE axis (dataset mean removed; a
#                       collapsed conditional-mean projector scores ~0 here)
#   shuffled control -- hal from image i vs real from image j != i
# batch_pearson and the shuffled control are the ones that answer the question:
# they are exactly the metrics that a mean-collapsed projector cannot fake.
#
# Arms (1 seed each -- these metrics average over ~200 patches x thousands of
# samples, so they are far less seed-noisy than mIoU, and each .pt is ~865MB on
# a 92%-full scratch):
#   control   plain MSE                      the baseline being accused of collapse
#   ccos      centered cosine, prefusion w=0.01   the tuned-weight version
#   ccos_w1   centered cosine, prefusion w=1.0    strongest possible cosine pressure
#
# ccos_w1 is included precisely BECAUSE it was the worst arm downstream (-2.93
# peeking). If heavy cosine pressure improves hallucination while destroying
# mIoU, that is the cleanest possible demonstration that the two objectives
# come apart -- and the most useful result this experiment can produce.
set -uo pipefail

CONFIG="configs/delulu_best_dfc2020.yaml"
START="${START:-s1}"; NEW="${NEW:-s2_norgb}"
SELECTOR="${SELECTOR:-transfer}"
SEED="${SEED:-0}"
# 40 epochs, not the config's 128. make_scheduler() sets T_max from the epoch
# count, so this is a complete, fully-annealed 40-epoch run reaching eta_min --
# NOT a truncated 128-epoch run. The question here is whether the cosine term
# changes what the projector learns, which is visible well before convergence;
# spending 3x the GPU time to sharpen a number we are comparing ACROSS arms
# (all trained identically) buys nothing.
EPOCHS="${EPOCHS:-40}"
RESULTS_CSV="res/delulu/dfc2020_halluc_ckpt.csv"
LEDGER="logs/train_delulu/submitted_halluc.tsv"
mkdir -p logs/train_delulu res/delulu; touch "${LEDGER}"

TEACHER=$(jq -r ".\"dfc2020_cobench/${START}/evan_base/upernet/split1\".checkpoint // empty" artifacts/sft_teachers.json)
[ -n "${TEACHER}" ] && [ -f "${TEACHER}" ] || { echo "[error] no teacher for ${START}"; exit 1; }

ACTIVE=" $(squeue -u "$USER" -h -o '%i' 2>/dev/null | tr '\n' ' ') "
INFLIGHT=""
while read -r jid rest; do
    case "${ACTIVE}" in *" ${jid} "*) INFLIGHT="${INFLIGHT}${rest}"$'\n' ;; esac
done < "${LEDGER}"

n=0; dup=0; skip=0
for ARM in ${ARMS:-control ccos ccos_w1}; do
    TAG="${START} ${NEW} ${SELECTOR} ${ARM} s${SEED}"
    CKPT="halluc_dfc2020_${START}to${NEW}_${ARM}_s${SEED}.pt"

    # Unlike the recon A/B, dedup also checks the FILESYSTEM: these jobs exist to
    # produce a .pt, so an existing one means the work is already done. The recon
    # script only checked squeue, which silently re-ran completed configs.
    if [ -f "checkpoints/${CKPT}" ]; then
        echo "  [done] ${TAG}  -> checkpoints/${CKPT}"; skip=$((skip+1)); continue
    fi
    if printf '%s' "${INFLIGHT}" | grep -qxF "${TAG}"; then
        echo "  [have] ${TAG}"; dup=$((dup+1)); continue
    fi

    EX="ALL,DATASET=dfc2020,START=${START},NEW=${NEW},TEACHER=${TEACHER}"
    EX="${EX},CONFIG=${CONFIG},SELECT_BY=${SELECTOR},SEED=${SEED},BATCH_SIZE=8"
    EX="${EX},EPOCHS=${EPOCHS}"
    EX="${EX},RESULTS_CSV=${RESULTS_CSV},SAVE_CHECKPOINT=1,CKPT_NAME=${CKPT}"
    EX="${EX},CONFIG_LABEL=halluc_${ARM}"
    case "${ARM}" in
        ccos)    EX="${EX},RECON_LOSS=mse_ccos,RECON_COS_W_PREFUSION=0.01,RECON_COS_W_LATENT=1.0" ;;
        ccos_w1) EX="${EX},RECON_LOSS=mse_ccos" ;;
    esac

    n=$((n+1))
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "[$n] ${ARM} seed=${SEED} -> ${CKPT}"
    else
        jid=$(sbatch --parsable --export="${EX}" sh/train_delulu_job.sh)
        printf '%s\t%s\n' "${jid}" "${TAG}" >> "${LEDGER}"
        echo "[$n] ${jid}  ${ARM} seed=${SEED} -> ${CKPT}"
    fi
done
echo "total: ${n} submitted; ${dup} in flight; ${skip} already on disk"
echo "then: bash sh/halluc_analyze_dfc2020.sh"
