#!/bin/bash
# Run the hallucination-quality analysis over the arms trained by
# sh/halluc_ckpt_dfc2020.sh and collect the numbers into one place.
#
# What to read (see analysis/analyze_hallucination_correlation.py):
#   patch_pearson aligned vs shuffled -- the GAP is what matters. A high aligned
#     score with a small gap means the projector is reproducing shared latent
#     geometry, not this image's content.
#   batch_pearson [KEY] -- Pearson across the SAMPLE axis with the dataset mean
#     removed. A projector collapsed to the conditional mean scores ~0 here no
#     matter how good its patch_pearson looks. THIS is the number that decides
#     whether the cosine term improved hallucination.
#   patch_spearman / batch_spearman -- the same two metrics on ranks, each with
#     its own shuffled null. Read them as a DIFFERENCE against the Pearson row:
#     Pearson >> Spearman means the linear score is carried by a few
#     high-magnitude channels; Spearman >> Pearson means recovery is monotone
#     but at the wrong amplitude, which only the linear metric penalizes.
#
# The comparison of interest is control vs ccos vs ccos_w1 on batch_pearson and
# on the aligned-minus-shuffled gap -- NOT on mIoU, which the recon A/B already
# settled (cosine does not help and at w=1.0 hurts).
set -uo pipefail

START="${START:-s1}"; NEW="${NEW:-s2_norgb}"; SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-res/hallucination_correlation/dfc2020_recon}"
N_BATCHES="${N_BATCHES:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
mkdir -p "${OUT_ROOT}"

# SUBMIT=1 dispatches each arm as its own job (sh/halluc_analyze_job.sh) instead
# of running them inline. Inline only works from inside an allocation, and runs
# the arms one after another; submitted they run in parallel.
SUBMIT="${SUBMIT:-0}"

for ARM in ${ARMS:-control ccos ccos_w1}; do
    CKPT="checkpoints/halluc_dfc2020_${START}to${NEW}_${ARM}_s${SEED}.pt"
    OUT="${OUT_ROOT}/${ARM}"
    LOG="${OUT_ROOT}/${ARM}.log"
    if [ ! -f "${CKPT}" ]; then
        echo "[skip] ${ARM}: no checkpoint at ${CKPT}"; continue
    fi
    mkdir -p "${OUT}"
    if [ "${SUBMIT}" = "1" ]; then
        EX="ALL,CKPT=${CKPT},OUT_DIR=${OUT},DATASET=dfc2020"
        EX="${EX},N_BATCHES=${N_BATCHES},BATCH_SIZE=${BATCH_SIZE}"
        EX="${EX},N_VIS=${N_VIS:-4},SEED=${SEED}"
        jid=$(sbatch --parsable --export="${EX}" sh/halluc_analyze_job.sh)
        echo "[submit] ${jid}  ${ARM} -> ${OUT}"
        continue
    fi
    echo "=== ${ARM} -> ${LOG} ==="
    python -u analysis/analyze_hallucination_correlation.py \
        --checkpoint "${CKPT}" \
        --dataset dfc2020 \
        --n_batches "${N_BATCHES}" \
        --batch_size "${BATCH_SIZE}" \
        --n_vis "${N_VIS:-4}" \
        --seed "${SEED}" \
        --out_dir "${OUT}" 2>&1 | tee "${LOG}"
done

if [ "${SUBMIT}" = "1" ]; then
    echo
    echo "Submitted. The per-arm log is \${OUT_DIR}/analyze.log, not \${ARM}.log,"
    echo "so re-run this script without SUBMIT=1 semantics for the summary:"
    echo "  grep -E 'gap \\(|\\[KEY\\]|pearson - spearman' ${OUT_ROOT}/*/analyze.log"
    exit 0
fi

echo
echo "=== SUMMARY (aligned / shuffled / gap, then across-sample KEY) ==="
echo "    Pearson and Spearman rows are interleaved per arm; compare arms down a"
echo "    column, and Pearson-vs-Spearman across a row."
for ARM in ${ARMS:-control ccos ccos_w1}; do
    LOG="${OUT_ROOT}/${ARM}.log"
    [ -f "${LOG}" ] || continue
    echo "--- ${ARM} ---"
    grep -E "gap \(|\[KEY\]|pearson - spearman" "${LOG}" || echo "  (no metrics found)"
done
