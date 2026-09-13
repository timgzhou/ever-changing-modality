#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/duo_temps/%j.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Asymmetric Duos (arXiv 2505.18636) temperature weighting for DeluluNet transfer.
# Fits two scalars on val2 -- once against the frozen teacher's predictions
# (label-free, the deployable setting) and once against val2 labels (reference).
# Test is held out from fitting.

source sh/env.sh
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DFC2020_SPLIT=cobench
mkdir -p logs/duo_temps res/delulu

OUT="res/delulu/duo_temps.csv"
rm -f "${OUT}"

teacher_for () {
    jq -r ".\"dfc2020_cobench/$1/evan_base/upernet/split1\".checkpoint // empty" artifacts/sft_teachers.json
}

for P in s1:s2_norgb s1:s2_rgb s2_norgb:s1 s2_norgb:s2_rgb s2_rgb:s1 s2_rgb:s2_norgb; do
    START="${P%%:*}"; NEW="${P##*:}"
    TEACHER=$(teacher_for "${START}")
    [ -f "${TEACHER}" ] || { echo "[skip] no teacher for ${START}"; continue; }
    for V in full distill_only; do
        if [ "$V" = "full" ]; then
            CKPT="checkpoints/delulunet_dfc2020_${START}_to_${NEW}_upernet_ll0.36_seed0.pt"
        else
            CKPT="checkpoints/delulunet_dfc2020_${START}_to_${NEW}_upernet_distillonly_initteacher_seed0.pt"
        fi
        [ -f "${CKPT}" ] || { echo "[skip] missing ${CKPT}"; continue; }
        python -u fit_duo_temps.py \
            --checkpoint "${CKPT}" --teacher_checkpoint "${TEACHER}" \
            --start_mod "${START}" --new_mod "${NEW}" --variant "${V}" \
            --dataset dfc2020 --batch_size 8 --out_csv "${OUT}"
    done
done

echo
echo "=== summary: transfer-path aggregation ==="
python -u - <<'PY'
import csv, statistics as st
rows = list(csv.DictReader(open('res/delulu/duo_temps.csv')))
f = lambda x: float(x)
print(f"{'direction':<22} {'variant':<13} {'hal':>6} {'real':>6} {'equal':>6} "
      f"{'duo/tch':>8} {'duo/lbl':>8} {'T_s':>7} {'T_w':>7}")
print('-' * 92)
for r in rows:
    print(f"{r['start_mod']+'->+'+r['new_mod']:<22} {r['variant']:<13} "
          f"{f(r['head_hal']):>6.2f} {f(r['head_real']):>6.2f} {f(r['equal_vote']):>6.2f} "
          f"{f(r['duo_teacher']):>8.2f} {f(r['duo_labels']):>8.2f} "
          f"{f(r['T_strong_teacher']):>7.3f} {f(r['T_weak_teacher']):>7.3f}")
print('-' * 92)
for v in ('full', 'distill_only'):
    sub = [r for r in rows if r['variant'] == v]
    if not sub: continue
    m = lambda k: st.mean(f(r[k]) for r in sub)
    print(f"{'MEAN ' + v:<36} {m('head_hal'):>6.2f} {m('head_real'):>6.2f} "
          f"{m('equal_vote'):>6.2f} {m('duo_teacher'):>8.2f} {m('duo_labels'):>8.2f}")
    print(f"{'':<36} duo/teacher - equal = {m('duo_teacher')-m('equal_vote'):+.2f} | "
          f"duo/teacher - real = {m('duo_teacher')-m('head_real'):+.2f}")
PY
