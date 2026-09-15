#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/train_delulu/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# One train_delulu direction, any dataset. Generic counterpart to
# sh/train_delulu_dfc2020_job.sh (which hardcodes the DFC2020 specifics).
#
# Expected env vars (set by sh/train_delulu_all.sh):
#   DATASET, NEW, TEACHER            required
#   EPOCHS, BATCH_SIZE, SEED, LAMBDA_LATENT, STUDENT_INIT, RESULTS_CSV   optional
#   CONFIG=<yaml> SELECT_BY=<transfer|peeking|addition>
#                             load tuned hyperparameters from a configs/*.yaml
#                             (see configs/delulu_best_{dfc2020,benv2}.yaml).
#                             Values loaded this way are overridden by any
#                             matching env var already set, so a launcher can
#                             still pin one parameter explicitly.
#   PROJ_LAYERS=<n>           depth of the cross-modal (masking) projector:
#                             (n-1) self-attention blocks + 1 cross-attention
#                             block. Default 2. Not a results-CSV column, so
#                             encode it in CONFIG_LABEL when sweeping depth.
#   RECON_LOSS=mse_cos        add a (1-cosine) term to the prefusion/latent
#                             feature-matching losses (default mse). mse_ccos
#                             takes that cosine on MEAN-CENTERED tokens, which
#                             is what actually penalises mean-collapse: a pure
#                             mean predictor already scores 0.87/0.71 raw cosine.
#   RECON_COS_W_PREFUSION=    weight on the cosine term in the prefusion loss
#                             (default 1.0). Prefusion MSE is ~0.009-0.040, so
#                             1.0 makes cosine 20-100x the MSE and rescales the
#                             whole term; ~0.01 keeps them comparable.
#   RECON_COS_W_LATENT=       same for the latent loss (default 1.0, already
#                             sane there: latent targets are post-LayerNorm)
#   RECON_DROP_CLS=1          drop the CLS token from those reconstruction
#                             targets (a segmenter decoder never reads it)
#   SAVE_CHECKPOINT=0         skip writing the final .pt (scratch is quota-bound;
#                             ablations that only need the metrics should set this)
#   SELF_DISTILL_ADDITION=1   opt-in: distil the new-modality heads against the
#                             student's own addition path instead of the frozen
#                             unimodal teacher (experimental, off by default)
#
# The STARTING modality is not passed: train_delulu.py reads it back out of the
# teacher checkpoint's evan_config, so teacher and student can never disagree.

set -euo pipefail
source sh/env.sh
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs/train_delulu checkpoints res/delulu

: "${DATASET:?}"; : "${NEW:?}"; : "${TEACHER:?}"
if [ ! -f "${TEACHER}" ]; then
    echo "[error] teacher checkpoint not found: ${TEACHER}"; exit 1
fi

# Optional: hydrate hyperparameters from a tuned config YAML. Emitted as shell
# assignments and eval'd, using ${VAR:=value} so anything already exported by the
# caller wins. `mae_mask_ratio` in the YAML maps to TOKEN_MASK_RATIO here.
if [ -n "${CONFIG:-}" ]; then
    : "${SELECT_BY:?CONFIG requires SELECT_BY=transfer|peeking|addition}"
    eval "$(python3 - "$CONFIG" "$SELECT_BY" <<'PYEOF'
import sys, yaml
cfg, sel = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(cfg))
if sel not in d['configs']:
    sys.exit(f"echo '[error] {sel} not in {cfg}'; exit 1")
ENV = {'lr':'LR','lambda_latent':'LAMBDA_LATENT','lambda_prefusion':'LAMBDA_PREFUSION',
       'lambda_distill':'LAMBDA_DISTILL','modality_dropout_startmod':'MODALITY_DROPOUT_STARTMOD',
       'modality_dropout_newmod':'MODALITY_DROPOUT_NEWMOD','labeled_frequency':'LABELED_FREQUENCY',
       'mae_mask_ratio':'TOKEN_MASK_RATIO'}
for k, v in d['configs'][sel]['hparams'].items():
    if k in ENV: print(f': "${{{ENV[k]}:={v}}}"')
for k, v in (d.get('fixed') or {}).items():
    if k == 'weight_decay':           print(f': "${{WEIGHT_DECAY:={v}}}"')
    elif k == 'protect_lrm':          print(f': "${{PROTECT_LRM:={v}}}"')
    elif k == 'labeled_start_fraction': print(f': "${{LABELED_START_FRACTION:={v}}}"')
if d.get('epochs'): print(f': "${{EPOCHS:={d["epochs"]}}}"')
PYEOF
)"
    echo "Loaded ${SELECT_BY} config from ${CONFIG}"
fi

# Delulu hyperparameters: the biomassters s1->s2 best config, carried over as a
# first pass for every dataset (NOT tuned per dataset -- sweep afterwards).
# Kept identical to sh/train_delulu_dfc2020_job.sh so the two are comparable.
LR="${LR:-0.0001569391767106977}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3.351617951860976e-05}"
MODALITY_DROPOUT="${MODALITY_DROPOUT:-0.3}"
MODALITY_DROPOUT_STARTMOD="${MODALITY_DROPOUT_STARTMOD:-0.33189226742900324}"
MODALITY_DROPOUT_NEWMOD="${MODALITY_DROPOUT_NEWMOD:-0.17068517311514753}"
LABELED_FREQUENCY="${LABELED_FREQUENCY:-0.23002477810989655}"
LABELED_START_FRACTION="${LABELED_START_FRACTION:-0}"
# The latent loss was inflated by embed_dim (768) until 2026-08-20, so this
# carried-over value was tuned at a scale where latent was ~98% of the signal;
# at the corrected scale it contributes ~5%. Sweep it from the launcher.
LAMBDA_LATENT="${LAMBDA_LATENT:-0.3613664751387723}"
LAMBDA_PREFUSION="${LAMBDA_PREFUSION:-0.6430194633931678}"
LAMBDA_DISTILL="${LAMBDA_DISTILL:-0.15374988356364516}"
TOKEN_MASK_RATIO="${TOKEN_MASK_RATIO:-0.40414477259411485}"
PROTECT_LRM="${PROTECT_LRM:-0.0}"
SEED="${SEED:-0}"
STUDENT_INIT="${STUDENT_INIT:-teacher}"
SELF_DISTILL_FLAG=""
if [ -n "${SELF_DISTILL_ADDITION:-}" ] && [ "${SELF_DISTILL_ADDITION}" != "0" ]; then
    SELF_DISTILL_FLAG="--self_distill_addition"
fi
# Default keeps the historical behaviour (save). SAVE_CHECKPOINT=0 opts out.
SAVE_CKPT_FLAG="--save_checkpoint"
if [ "${SAVE_CHECKPOINT:-1}" = "0" ]; then
    SAVE_CKPT_FLAG=""
fi
# Provenance columns: which tuned config produced this row. Without these every
# cross-config row looks identical apart from its hyperparameters.
RECON_ARGS=""
[ -n "${RECON_LOSS:-}" ] && RECON_ARGS="${RECON_ARGS} --recon_loss ${RECON_LOSS}"
[ -n "${RECON_DROP_CLS:-}" ] && [ "${RECON_DROP_CLS}" != "0" ] && RECON_ARGS="${RECON_ARGS} --recon_drop_cls"
[ -n "${RECON_COS_W_PREFUSION:-}" ] && RECON_ARGS="${RECON_ARGS} --recon_cos_weight_prefusion ${RECON_COS_W_PREFUSION}"
[ -n "${RECON_COS_W_LATENT:-}" ] && RECON_ARGS="${RECON_ARGS} --recon_cos_weight_latent ${RECON_COS_W_LATENT}"

PROV_ARGS=""
[ -n "${CONFIG_LABEL:-}" ] && PROV_ARGS="${PROV_ARGS} --config_label ${CONFIG_LABEL}"
[ -n "${PROJ_LAYERS:-}" ] && PROV_ARGS="${PROV_ARGS} --intermediate_projector_num_layers ${PROJ_LAYERS}"
[ -n "${SELECT_BY:-}" ]    && PROV_ARGS="${PROV_ARGS} --select_by ${SELECT_BY}"
EPOCHS="${EPOCHS:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RESULTS_CSV="${RESULTS_CSV:-res/delulu/${DATASET}_unimodal_pairs.csv}"

# BioMassters is temporal: pool features over this many timesteps (<=12).
EXTRA_ARGS=""
if [ "${DATASET}" = "biomassters" ]; then
    EXTRA_ARGS="--num_time_steps ${NUM_TIME_STEPS:-12}"
fi

# START is informational only (train_delulu.py reads the real starting modality
# out of the teacher checkpoint); it is echoed so the launcher's in-flight guard
# can identify which direction a queued job is running.
echo "=== ${DATASET} | ${START:-?} -> +${NEW} | teacher=${TEACHER} ==="
echo "    lr=${LR} epochs=${EPOCHS} bs=${BATCH_SIZE} lambda_latent=${LAMBDA_LATENT} seed=${SEED}"

python -u train_delulu.py \
    --dataset "${DATASET}" \
    --new_mod_group "${NEW}" \
    --stage0_checkpoint "${TEACHER}" \
    --active_losses latent prefusion distill ce \
    --wandb_project "delulu-${DATASET}-pairs" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
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
    ${SELF_DISTILL_FLAG} \
    ${RECON_ARGS} \
    ${PROV_ARGS} \
    --student_init "${STUDENT_INIT}" \
    --seed "${SEED}" \
    ${SAVE_CKPT_FLAG} \
    --results_csv "${RESULTS_CSV}" \
    ${EXTRA_ARGS}
