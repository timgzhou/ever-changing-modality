#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/train_delulu/bm_chain_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Chain job: after the biomassters stage-0 reruns land, rebuild the teacher
# registry and relaunch the affected stage-1 cross-config runs.
#
# The registry is derived from the results CSVs by res/train_sft/sft_best.py, so
# stage 1 cannot be a plain afterok dependency -- that regeneration is the
# missing step.
#
# Submit with:
#   sbatch --dependency=afterany:<stage-0 ids> sh/biomassters_rerun_stage1.sh
set -uo pipefail
source sh/env.sh
mkdir -p logs/train_delulu

echo "=== regenerating teacher registry ==="
python res/train_sft/sft_best.py || { echo "[error] sft_best.py failed"; exit 1; }

echo
echo "=== biomassters teachers now registered ==="
python - <<'PYEOF'
import json, os
reg = json.load(open('artifacts/sft_teachers.json'))
for m in ['s1', 's2', 's2_rgb', 's2_norgb']:
    k = f'biomassters/{m}/evan_base/upernet+relu/split1'
    v = reg.get(k)
    if v:
        print(f"  {m:10s} val RMSE {v['val_metric']:6.2f}  test {v['test_metric']:6.2f}  "
              f"{os.path.basename(v['checkpoint'])}")
    else:
        print(f"  {m:10s} MISSING")
PYEOF

echo
echo "=== relaunching biomassters cross-config ==="
# The launcher's own ledger + squeue guard skips anything still in flight; rows
# already written are NOT re-detected, so the caller is expected to have moved
# the stale CSV aside first.
DATASETS=biomassters bash sh/train_delulu_crossconfig.sh
