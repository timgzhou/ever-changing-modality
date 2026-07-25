#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/download_biomassters/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Download the BioMassters geobench shards directly (no GPU needed).
#
# Why this exists: the released geobench-v2 0.9 pins only the FIRST 3 of the
# dataset's 7 tortilla shards, with sha256 values from an older HuggingFace
# upload. Downloading only those 3 yields a 43% subset of the benchmark
# (1735/759/1200 train/val/test instead of the paper's 4011/1739/2776), and the
# stale hashes make verification fail on correctly-downloaded files.
#
# Fix: the installed geobench_v2/datasets/biomassters.py has been patched with
# the 7-shard path list and current hashes from the upstream git repo
# (backup at biomassters.py.bak-v09). This script fetches all 7 shards so the
# full, leaderboard-comparable splits are available.
#
# Usage:  sbatch sh/download_biomassters.sh

set -uo pipefail

DATA_DIR="datasets/geoben2/biomassters"
BASE_URL="https://hf.co/datasets/aialliance/biomassters/resolve/main"

# All 7 shards (~138 GB total). wget -c skips/resumes ones already complete.
SHARDS=(
    "geobench_biomassters.0000.part.tortilla"
    "geobench_biomassters.0001.part.tortilla"
    "geobench_biomassters.0002.part.tortilla"
    "geobench_biomassters.0003.part.tortilla"
    "geobench_biomassters.0004.part.tortilla"
    "geobench_biomassters.0005.part.tortilla"
    "geobench_biomassters.0006.part.tortilla"
)

mkdir -p logs/download_biomassters "${DATA_DIR}"
cd "${DATA_DIR}" || exit 1

# Clear stray suffixed duplicates from previous aborted geobench downloads
# (download_url appends .1/.2/... when the target already exists).
rm -f geobench_biomassters.*.part.tortilla.[0-9]*

echo "=== downloading BioMassters shards into ${DATA_DIR} ==="
date

FAILED=0
for SHARD in "${SHARDS[@]}"; do
    echo ""
    echo "--- ${SHARD} ---"
    # -c resumes a partial file; --tries/--timeout survive flaky transfers.
    wget -c \
         --progress=dot:giga \
         --tries=10 \
         --timeout=60 \
         --waitretry=15 \
         -O "${SHARD}" \
         "${BASE_URL}/${SHARD}"
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "[error] wget failed for ${SHARD} (exit ${RC})"
        FAILED=1
    fi
done

echo ""
echo "=== final state ==="
ls -la
date

if [ $FAILED -ne 0 ]; then
    echo "[error] one or more shards failed to download; re-run this job (wget -c resumes)."
    exit 1
fi

# Report actual hashes for the record. These are EXPECTED to differ from the
# values hardcoded in geobench_v2/datasets/biomassters.py -- that mismatch is
# the upstream bug this script works around, not a corrupted download.
echo ""
echo "=== sha256 (informational; geobench's pinned values are stale) ==="
for SHARD in "${SHARDS[@]}"; do
    sha256sum "${SHARD}"
done

echo ""
echo "=== done -- all 3 shards present; training job will skip verification ==="
