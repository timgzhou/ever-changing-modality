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

for ARM in ${ARMS:-control ccos ccos_w1}; do
    CKPT="checkpoints/halluc_dfc2020_${START}to${NEW}_${ARM}_s${SEED}.pt"
    OUT="${OUT_ROOT}/${ARM}"
    LOG="${OUT_ROOT}/${ARM}.log"
    if [ ! -f "${CKPT}" ]; then
        echo "[skip] ${ARM}: no checkpoint at ${CKPT}"; continue
    fi
    mkdir -p "${OUT}"
    echo "=== ${ARM} -> ${LOG} ==="
    python -u analysis/analyze_hallucination_correlation.py \
        --checkpoint "${CKPT}" \
        --dataset dfc2020 \
        --n_batches "${N_BATCHES}" \
        --batch_size "${BATCH_SIZE}" \
        --n_vis "${N_VIS:-4}" \
        --out_dir "${OUT}" 2>&1 | tee "${LOG}"
done

echo
echo "=== SUMMARY (aligned / shuffled / gap, then across-sample KEY) ==="
for ARM in ${ARMS:-control ccos ccos_w1}; do
    LOG="${OUT_ROOT}/${ARM}.log"
    [ -f "${LOG}" ] || continue
    echo "--- ${ARM} ---"
    grep -E "gap \(|\[KEY\]" "${LOG}" || echo "  (no metrics found)"
done
