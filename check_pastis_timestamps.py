"""
Quick script to compute timestamp count statistics across PASTIS splits.
Usage: python check_pastis_timestamps.py [--data_root datasets/geoben2/pastis]
"""
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GEO-Bench-2'))

from geobench_v2.datasets.pastis import GeoBenchPASTIS

def count_timestamps(ds):
    counts = []
    for i in range(len(ds)):
        row = ds.data_df.read(i)
        dates = row["dates"].iloc[0]
        counts.append(len(dates))
    return np.array(counts)

def print_stats(name, counts):
    print(f"\n{name} ({len(counts)} samples)")
    print(f"  min:    {counts.min()}")
    print(f"  max:    {counts.max()}")
    print(f"  mean:   {counts.mean():.1f}")
    print(f"  median: {np.median(counts):.1f}")
    for q in [10, 25, 75, 90]:
        print(f"  p{q}:    {np.percentile(counts, q):.1f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="datasets/geoben2/pastis")
    args = parser.parse_args()

    shared = dict(
        root=args.data_root,
        num_time_steps=1000,       # large enough to never truncate
        temporal_aggregation=None, # don't collapse — we just want raw T counts
        band_order={"s2": ["B02"]},  # load only 1 band to keep it fast
    )

    for split in ["train", "val", "test"]:
        ds = GeoBenchPASTIS(split=split, **shared)
        counts = count_timestamps(ds)
        print_stats(split, counts)

if __name__ == "__main__":
    main()

# python check_pastis_timestamps.py