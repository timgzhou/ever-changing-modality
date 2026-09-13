#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/head_analysis/%j.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Per-head accuracy analysis for DeluluNet, all 6 directions x {full, distill_only}.
# Runs in ONE job: each direction is eval-only (test split, 986 tiles), so the
# whole sweep is far cheaper than the training runs.

source sh/env.sh
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# MUST match the split the teachers were trained on (8 classes, not the ROI 10).
export DFC2020_SPLIT=cobench
mkdir -p logs/head_analysis res/delulu

OUT="res/delulu/head_analysis.csv"
rm -f "${OUT}"

for P in s1:s2_norgb s1:s2_rgb s2_norgb:s1 s2_norgb:s2_rgb s2_rgb:s1 s2_rgb:s2_norgb; do
    START="${P%%:*}"; NEW="${P##*:}"
    for V in full distill_only; do
        if [ "$V" = "full" ]; then
            CKPT="checkpoints/delulunet_dfc2020_${START}_to_${NEW}_upernet_ll0.36_seed0.pt"
        else
            CKPT="checkpoints/delulunet_dfc2020_${START}_to_${NEW}_upernet_distillonly_initteacher_seed0.pt"
        fi
        if [ ! -f "${CKPT}" ]; then
            echo "[skip] missing ${CKPT}"; continue
        fi
        python -u analyze_delulu_heads.py \
            --checkpoint "${CKPT}" --start_mod "${START}" --new_mod "${NEW}" \
            --variant "${V}" --dataset dfc2020 --batch_size 8 --out_csv "${OUT}"
    done
done

echo
echo "=== summary ==="
python -u - <<'PY'
import csv
rows=list(csv.DictReader(open('res/delulu/head_analysis.csv')))
for path in ('transfer','peeking'):
    print(f"\n### {path} (real modality is the NON-hallucinated one)")
    print(f"{'direction':<22} {'variant':<13} {'head[hal]':>10} {'head[real]':>11} {'ensemble':>9} {'oracle':>8}")
    print('-'*78)
    for r in rows:
        if r['path']!=path: continue
        hal=r['hallucinated']
        h_hal = r['head_start'] if hal==r['start_mod'] else r['head_new']
        h_real= r['head_new']  if hal==r['start_mod'] else r['head_start']
        print(f"{r['start_mod']+'->+'+r['new_mod']:<22} {r['variant']:<13} "
              f"{float(h_hal):>10.2f} {float(h_real):>11.2f} {float(r['ensemble']):>9.2f} {float(r['oracle_pick']):>8.2f}")
PY
