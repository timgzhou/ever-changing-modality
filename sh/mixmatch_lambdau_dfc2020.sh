#!/bin/bash
# MixMatch lambda_u sweep on DFC2020 / Copernicus-Bench.
#
# Why this sweep exists
# ---------------------
# baseline_mixmatch.py defaults to lambda_u=75, the value from the MixMatch
# paper. That value was calibrated for an unsupervised term that is an L2/Brier
# loss on probabilities (bounded in [0,2]). Our SEGMENTATION branch instead uses
# hard cross-entropy, which is ~2.4x larger on matched tensors and unbounded. At
# lambda_u=75 the pseudo-label term contributes ~164 against a supervised loss of
# ~2.1, so the labeled signal is ~1% of the gradient and the model just fits its
# own pseudo-labels.
#
# Measured (s2_norgb, upernet, 3 epochs):
#     lambda_u=75.0  ->  4.2 test mIoU
#     lambda_u=1.0   -> 46.0 test mIoU
#
# So 1.0 is known-better but was a hypothesis test, not a tuned value. This
# sweeps the decade around it. (MixUp on the labeled branch is separately
# disabled for dense tasks -- see train_utils.py; set MIXMATCH_SEG_MIXUP=1 to
# restore it.)
#
# Usage:
#   bash sh/mixmatch_lambdau_dfc2020.sh            # dry run
#   SUBMIT=1 bash sh/mixmatch_lambdau_dfc2020.sh

set -u
DECODER="${DECODER:-upernet}"
MODEL="${MODEL:-evan_base}"
EPOCHS="${EPOCHS:-64}"
LR="${LR:-0.0001}"
MODALITIES="${MODALITIES:-s1 s2_rgb s2_norgb}"
LAMBDA_US="${LAMBDA_US:-0.5 1.0 5.0 25.0 75.0}"
SUBMIT="${SUBMIT:-0}"

n=0
for M in ${MODALITIES}; do
  for LU in ${LAMBDA_US}; do
    TAG="mixmatch_lu${LU}_${M}_${DECODER}"
    ARGS="baseline/baseline_mixmatch.py --dataset dfc2020 --modality ${M}"
    ARGS="${ARGS} --decoder_type ${DECODER} --model ${MODEL} --use_dino_weights"
    ARGS="${ARGS} --epochs ${EPOCHS} --lr ${LR} --lambda_u ${LU}"
    ARGS="${ARGS} --results_csv res/baselines/dfc2020_cobench_mixmatch_lambdau_${DECODER}.csv"
    if [ "$SUBMIT" = "1" ]; then
        sbatch --export=ALL,BASELINE_ARGS="${ARGS}",RUN_TAG="${TAG}",DECODER="${DECODER}" \
            sh/baselines_dfc2020_job.sh >/dev/null
    fi
    n=$((n+1))
    echo "  [$n] ${TAG}"
  done
done
echo
echo "total: ${n} jobs (SUBMIT=${SUBMIT})"
