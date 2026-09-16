#!/bin/bash
# 3x3 projector probe: depth {2,3,4} x loss {mse, mse_ccos, ccos}.
#
# Question: does the reconstruction loss change what a cross-modal projector can
# learn, once nothing else is in the way?
#
# The full-stack recon A/B could not answer this. There, the projector sat under
# fusion blocks, a decoder, a task head and three other loss terms, any of which
# could absorb or mask a difference in projection quality. Here the encoder is
# FROZEN and only the projector trains, on the reconstruction loss alone. No
# fusion, no decoder, no task head. If a loss matters for hallucination, it has
# to show up here.
#
# Losses (all cosine terms are the MEAN-CENTERED variant):
#   mse       plain F.mse_loss                     -- the incumbent
#   mse_ccos  MSE + COS_WEIGHT * (1 - centered cos) -- the hybrid
#   ccos      (1 - centered cos) ONLY               -- pure direction, no magnitude
#
# The ccos arm is the real test. It has no magnitude anchor at all, so it CANNOT
# be satisfied by regressing toward the mean -- if directional collapse is what
# limits hallucination, this arm should win on batch_pearson. If it instead
# scores near zero, the mean-collapse story is wrong and MSE was never the
# binding constraint.
#
# Read: test_batch_r (per-image recovery, immune to mean-collapse) and
# test_gap (aligned minus derangement-shuffled). NOT mIoU -- no task head here.
set -uo pipefail

CKPT="${CKPT:-checkpoints/halluc_dfc2020_s1tos2_norgb_control_s0.pt}"
SRC="${SRC:-s1}"; TGT="${TGT:-s2_norgb}"
DEPTHS="${DEPTHS:-2 3 4}"
LOSSES="${LOSSES:-mse mse_ccos ccos}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-8}"
RESULTS_CSV="${RESULTS_CSV:-res/delulu/projector_probe_dfc2020.csv}"
LEDGER="logs/projector_probe/submitted.tsv"
mkdir -p logs/projector_probe res/delulu; touch "${LEDGER}"

[ -f "${CKPT}" ] || { echo "[error] no checkpoint at ${CKPT}"; exit 1; }

ACTIVE=" $(squeue -u "$USER" -h -o '%i' 2>/dev/null | tr '\n' ' ') "
INFLIGHT=""
while read -r jid rest; do
    case "${ACTIVE}" in *" ${jid} "*) INFLIGHT="${INFLIGHT}${rest}"$'\n' ;; esac
done < "${LEDGER}"

# Arms already in the results CSV: this probe is cheap enough to rerun, but
# silently doubling rows makes the table ambiguous.
DONE=""
if [ -f "${RESULTS_CSV}" ]; then
    DONE=$(awk -F, 'NR>1 {print $1}' "${RESULTS_CSV}")
fi

n=0; dup=0; skip=0
for D in ${DEPTHS}; do
  for L in ${LOSSES}; do
    # Direction is part of the identity: without it a reverse-direction run
    # (s2_norgb -> s1) collides with the forward one in the same results CSV.
    TAG="${SRC}to${TGT}_d${D}_${L}_s${SEED}"
    if printf '%s\n' "${DONE}" | grep -qxF "${TAG}"; then
        echo "  [done] ${TAG} (already in ${RESULTS_CSV})"; skip=$((skip+1)); continue
    fi
    if printf '%s' "${INFLIGHT}" | grep -qxF "${TAG}"; then
        echo "  [have] ${TAG}"; dup=$((dup+1)); continue
    fi
    EX="ALL,CKPT=${CKPT},SRC=${SRC},TGT=${TGT},DEPTH=${D},LOSS=${L}"
    EX="${EX},SEED=${SEED},EPOCHS=${EPOCHS},RESULTS_CSV=${RESULTS_CSV},TAG=${TAG}"
    [ -n "${COS_WEIGHT:-}" ] && EX="${EX},COS_WEIGHT=${COS_WEIGHT}"
    n=$((n+1))
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "[$n] depth=${D} loss=${L}"
    else
        jid=$(sbatch --parsable --export="${EX}" sh/projector_probe_job.sh)
        printf '%s\t%s\n' "${jid}" "${TAG}" >> "${LEDGER}"
        echo "[$n] ${jid}  depth=${D} loss=${L}"
    fi
  done
done
echo "total: ${n} submitted; ${dup} in flight; ${skip} already done  ->  ${RESULTS_CSV}"
