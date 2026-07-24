"""
Regenerate the EuroSAT train1/train2 split files.

The original datasets/eurosat-train{1,2}.txt were lost and are not reproducible
from any record in this repo (no generator script, no git history -- datasets/
is gitignored). This script recreates them by splitting torchgeo's canonical
eurosat-train.txt in half, mirroring the val1/val2 procedure in
eurosat_data_utils.get_loaders_with_val().

The seed used originally is unknown. --seed 42 is the default because it is the
only seed default in the repo. Use --verify-stats to test a candidate seed
against .eurosat_zscore_stats_cache.json, which was computed from the original
train1 split and therefore acts as a fingerprint of it.

Usage:
    python generate_eurosat_splits.py                    # write splits, seed 42
    python generate_eurosat_splits.py --seed 0
    python generate_eurosat_splits.py --seed 42 --verify-stats
"""

from __future__ import annotations

import argparse
import json
import os
import random

SPLIT_SOURCE = 'ds_ers/eurosat-train.txt'
OUT_TRAIN1 = 'datasets/eurosat-train1.txt'
OUT_TRAIN2 = 'datasets/eurosat-train2.txt'
TIF_GLOB = 'ds_ers/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/*/*.tif'
STATS_CACHE = '.eurosat_zscore_stats_cache.json'


def generate(seed: int) -> tuple[list[str], list[str]]:
    """Split the canonical train list in half, mirroring the val1/val2 pattern."""
    with open(SPLIT_SOURCE) as f:
        samples = [line.strip() for line in f if line.strip()]

    # Same procedure as get_loaders_with_val(): random.Random(seed), shuffle, halve.
    rng = random.Random(seed)
    rng.shuffle(samples)
    half = len(samples) // 2
    return samples[:half], samples[half:]


def write_splits(train1: list[str], train2: list[str]) -> None:
    os.makedirs(os.path.dirname(OUT_TRAIN1), exist_ok=True)
    for path, names in ((OUT_TRAIN1, train1), (OUT_TRAIN2, train2)):
        with open(path, 'w') as f:
            f.write('\n'.join(names) + '\n')
        print(f"wrote {path}  ({len(names)} samples)")


def verify_stats(train1: list[str], tol: float = 0.5) -> bool:
    """
    Recompute z-score stats from a candidate train1 and compare to the cached
    stats, which were derived from the original split. A match identifies the
    original seed; a mismatch rules that seed out.
    """
    import glob

    import rasterio
    import torch

    if not os.path.exists(STATS_CACHE):
        print(f"no {STATS_CACHE} to verify against")
        return False

    with open(STATS_CACHE) as f:
        cached = json.load(f)

    wanted = {n.replace('.jpg', '.tif') for n in train1}
    tif_files = sorted(glob.glob(TIF_GLOB))
    if not tif_files:
        raise FileNotFoundError(
            f"No .tif files at {TIF_GLOB}. Run the EuroSAT loader once so "
            "torchgeo downloads the archive (download=True), then retry."
        )

    # Mirrors get_eurosat_zscore_stats() accumulation exactly.
    sums = torch.zeros(13)
    sq_sums = torch.zeros(13)
    count = 0
    processed = 0
    for tif_path in tif_files:
        if os.path.basename(tif_path) not in wanted:
            continue
        with rasterio.open(tif_path) as src:
            image = torch.tensor(src.read(), dtype=torch.float32)
            sums += image.sum(dim=(1, 2))
            sq_sums += (image ** 2).sum(dim=(1, 2))
            count += image.shape[1] * image.shape[2]
            processed += 1

    if processed == 0:
        raise RuntimeError("No candidate train1 samples matched any .tif on disk.")

    means = sums / count
    stds = torch.sqrt(torch.clamp((sq_sums / count) - (means ** 2), min=1e-8))

    print(f"\nprocessed {processed} images")
    print(f"{'band':>5}  {'cached mean':>13}  {'recomputed':>13}   match")
    ok = True
    for i, (band, (cm, cs)) in enumerate(cached.items()):
        dm, ds = abs(means[i].item() - cm), abs(stds[i].item() - cs)
        hit = dm < tol and ds < tol
        ok &= hit
        print(f"{band:>5}  {cm:>13.4f}  {means[i].item():>13.4f}   {'yes' if hit else 'NO'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--verify-stats', action='store_true',
                    help='check this seed against the cached stats fingerprint')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    train1, train2 = generate(args.seed)
    print(f"seed={args.seed}: train1={len(train1)} train2={len(train2)} "
          f"(total {len(train1) + len(train2)}, overlap {len(set(train1) & set(train2))})")

    if args.verify_stats:
        match = verify_stats(train1)
        print(f"\n=> seed {args.seed} {'MATCHES' if match else 'does NOT match'} the original split")
        if not match:
            print("   splits will be valid but not identical to the originals")

    if args.dry_run:
        print("\ndry run, nothing written")
        return
    write_splits(train1, train2)


if __name__ == '__main__':
    main()
