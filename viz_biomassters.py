"""
Visualize BioMassters samples: per-timestep, per-modality bands + the AGB target.

For one train, one val, and one test sample, produces a figure whose rows are
timesteps and whose columns are:
  - S2 true-color RGB (B04,B03,B02)
  - each S2 band (10)
  - each S1 band (4)
The AGB target map (constant over time) is shown once per sample.

Note: images are z-score normalized by the dataset's ZScoreNormalizer, and the
AGB mask is (agb / 289.89). We de-normalize the mask back to t/ha for display.

Usage:
    python viz_biomassters.py                       # T=6, one sample per split
    python viz_biomassters.py --num_time_steps 4 --out figs/bm
    python viz_biomassters.py --index 3             # pick sample index per split
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt
import numpy as np
import torch

from biomassters_data_utils import (
    AGB_STD,
    BIOMASSTERS_S1_BANDS,
    BIOMASSTERS_S2_BANDS,
    TemporalStackedDataset,
    _N_S1,
    _N_S2,
)
from geobench_v2.datasets.biomassters import GeoBenchBioMassters


S2_BANDS = list(BIOMASSTERS_S2_BANDS)
S1_BANDS = list(BIOMASSTERS_S1_BANDS)
# True-color RGB indices within the S2 stack: B04(R)=idx2, B03(G)=idx1, B02(B)=idx0.
RGB_IDX = [S2_BANDS.index(b) for b in ("B04", "B03", "B02")]


def _norm01(x: np.ndarray) -> np.ndarray:
    """Robust 2-98 percentile stretch to [0,1] for display."""
    lo, hi = np.nanpercentile(x, 2), np.nanpercentile(x, 98)
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _load_split(split: str, num_time_steps: int, data_root: str):
    ds = GeoBenchBioMassters(
        root=data_root,
        split=split,
        band_order={"s2": S2_BANDS, "s1": S1_BANDS},
        num_time_steps=num_time_steps,
        return_stacked_image=False,
        download=False,
    )
    return TemporalStackedDataset(ds, num_time_steps)


def plot_sample(sample: dict, split: str, index: int, out_path: str) -> None:
    image = sample["image"]              # [C, T, H, W]
    mask = sample["mask"].float()        # [H, W], normalized AGB
    C, T, H, W = image.shape

    s2 = image[:_N_S2]                   # [10, T, H, W]
    s1 = image[_N_S2:_N_S2 + _N_S1]      # [4,  T, H, W]

    # Columns: RGB + 10 S2 bands + 4 S1 bands. AGB target in its own trailing axis.
    col_titles = ["S2 RGB"] + S2_BANDS + S1_BANDS
    n_band_cols = len(col_titles)
    n_cols = n_band_cols + 1             # + AGB target column
    n_rows = T

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows), squeeze=False
    )

    agb_tha = (mask * AGB_STD).numpy()   # de-normalize to t/ha for display
    vmin, vmax = np.nanpercentile(agb_tha, 2), np.nanpercentile(agb_tha, 98)

    for t in range(T):
        # Col 0: S2 true-color RGB
        rgb = s2[RGB_IDX, t].permute(1, 2, 0).numpy()  # [H,W,3]
        rgb = np.stack([_norm01(rgb[..., c]) for c in range(3)], axis=-1)
        ax = axes[t][0]
        ax.imshow(rgb)
        ax.set_ylabel(f"t={t}", fontsize=9)

        # S2 bands
        for j, _band in enumerate(S2_BANDS):
            ax = axes[t][1 + j]
            ax.imshow(_norm01(s2[j, t].numpy()), cmap="viridis")

        # S1 bands
        for j, _band in enumerate(S1_BANDS):
            ax = axes[t][1 + len(S2_BANDS) + j]
            ax.imshow(_norm01(s1[j, t].numpy()), cmap="magma")

        # AGB target (same every timestep; label only on the first row)
        ax = axes[t][n_cols - 1]
        im = ax.imshow(agb_tha, cmap="YlGn", vmin=vmin, vmax=vmax)
        if t == 0:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AGB (t/ha)")

        # Clean ticks; titles only on the top row.
        for c in range(n_cols):
            axes[t][c].set_xticks([])
            axes[t][c].set_yticks([])
            if t == 0:
                title = col_titles[c] if c < n_band_cols else "AGB target"
                axes[t][c].set_title(title, fontsize=8)

    fig.suptitle(
        f"BioMassters — {split} sample #{index}  (T={T}, {H}x{W})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="datasets/geoben2/biomassters")
    p.add_argument("--num_time_steps", type=int, default=6)
    p.add_argument("--index", type=int, default=0,
                   help="Sample index to plot within each split.")
    p.add_argument("--out", default="figs/biomassters",
                   help="Output path prefix; files are <out>_<split>.png")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for split in ("train", "validation", "test"):
        ds = _load_split(split, args.num_time_steps, args.data_root)
        idx = min(args.index, len(ds) - 1)
        print(f"[{split}] {len(ds)} samples; plotting index {idx}")
        sample = ds[idx]
        tag = "val" if split == "validation" else split
        plot_sample(sample, split, idx, f"{args.out}_{tag}.png")


if __name__ == "__main__":
    main()
