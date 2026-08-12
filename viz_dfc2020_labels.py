"""
DFC2020 label provenance figure — MODIS `lc` vs official `dfc` ground truth.

Documents the label-source bug: the HuggingFace GFM-Bench/DFC2020 packaging
shipped the SEN12MS MODIS-derived `lc` product as the segmentation target
instead of the DFC2020 contest ground truth. MODIS land cover is ~500 m native,
so on a 96x96 tile (960 m) it resolves to 2-3 blobs, while the official labels
are semi-manually generated at 10 m.

Both label rasters ship side by side in the official release (lc_* and dfc_*
directories, same patch ids), so this figure reads them from the same patch and
needs no model or checkpoint.

Usage:
    python viz_dfc2020_labels.py --out figs/dfc2020_label_provenance.png
"""

from __future__ import annotations

import argparse
import collections

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy import ndimage

from dfc2020_official_data_utils import (
    IGBP2DFC,
    build_index,
)

# DFC scheme, classes 1-10 (0 = unlabeled).
CLASS_NAMES = [
    "Forest", "Shrubland", "Savanna", "Grassland", "Wetlands",
    "Croplands", "Urban/Built-up", "Snow/Ice", "Barren", "Water",
]
CLASS_COLORS = [
    "#1b7837",  # Forest
    "#a6dba0",  # Shrubland
    "#fbb4ae",  # Savanna
    "#d9f0a3",  # Grassland
    "#80cdc1",  # Wetlands
    "#f6e8c3",  # Croplands
    "#d73027",  # Urban/Built-up
    "#e0f3f8",  # Snow/Ice
    "#8c510a",  # Barren
    "#4575b4",  # Water
]

RGB_BANDS = [3, 2, 1]  # B4, B3, B2 within the 13-band S2 stack


def _stretch(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Percentile contrast stretch to [0, 1] for display."""
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[-1]):
        band = x[..., c].astype(np.float32)
        p_lo, p_hi = np.percentile(band, [lo, hi])
        out[..., c] = np.clip((band - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    return out


def _to_dfc_ids(raw: np.ndarray) -> np.ndarray:
    """Normalise a label raster to DFC ids 1-10 (0 = unlabeled)."""
    arr = raw.astype(np.int64)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
    if arr.max() > 10:
        arr = IGBP2DFC[np.clip(arr, 0, 17)]
    return arr


def _granularity(arr: np.ndarray) -> dict:
    """Structural granularity stats for a label raster."""
    present = [v for v in np.unique(arr) if v > 0]
    n_regions, sizes = 0, []
    for v in present:
        lab, n = ndimage.label(arr == v, structure=np.ones((3, 3)))
        n_regions += n
        if n:
            sizes.extend(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)))
    counts = np.bincount(arr.ravel())
    counts[0] = 0  # ignore unlabeled when measuring dominance
    return {
        "classes": len(present),
        "regions": n_regions,
        "median_region": float(np.median(sizes)) if sizes else 0.0,
        "dominant": counts.max() / max(arr.size, 1),
    }


def _draw_labels(ax, arr: np.ndarray, cmap, norm) -> None:
    """Draw a label raster; unlabeled (0) is hatched grey."""
    shown = np.ma.masked_where(arr == 0, arr)
    ax.imshow(shown, cmap=cmap, norm=norm, interpolation="nearest")
    if (arr == 0).any():
        ax.imshow(
            np.ma.masked_where(arr != 0, np.ones_like(arr)),
            cmap=ListedColormap(["#bdbdbd"]), interpolation="nearest",
        )
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",
                   default="datasets/DFC2020_official/DFC_Public_Dataset")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--min_classes", type=int, default=4,
                   help="Require this many DFC classes in the official label.")
    p.add_argument("--scan_limit", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="figs/dfc2020_label_provenance.png")
    args = p.parse_args()

    records = build_index(args.data_root)

    # Sample across ROIs so the figure is not one city; prefer class-diverse tiles.
    rng = np.random.default_rng(args.seed)
    by_roi = collections.defaultdict(list)
    for r in records:
        by_roi[r["roi"]].append(r)
    pool = []
    for roi in sorted(by_roi):
        rs = by_roi[roi]
        idx = rng.choice(len(rs), size=min(len(rs), args.scan_limit // len(by_roi)),
                         replace=False)
        pool.extend(rs[i] for i in idx)

    scored = []
    for rec in pool:
        if rec.get("lc") is None:
            continue  # need both rasters to compare
        dfc = _to_dfc_ids(tifffile.imread(rec["dfc"]))
        g = _granularity(dfc)
        if g["classes"] >= args.min_classes:
            scored.append((g["classes"], rec))
    if not scored:
        raise SystemExit(f"No tiles with >= {args.min_classes} classes found.")
    scored.sort(key=lambda t: -t[0])

    # Spread the picks across distinct ROIs where possible.
    chosen, seen_rois = [], set()
    for _, rec in scored:
        if rec["roi"] not in seen_rois:
            chosen.append(rec); seen_rois.add(rec["roi"])
        if len(chosen) == args.num_samples:
            break
    for _, rec in scored:
        if len(chosen) == args.num_samples:
            break
        if rec not in chosen:
            chosen.append(rec)

    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(0.5, 11.5, 1.0), cmap.N)

    n = len(chosen)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.5 * n))
    if n == 1:
        axes = axes[None, :]

    for row, rec in enumerate(chosen):
        s2 = tifffile.imread(rec["s2"])
        if s2.shape[0] == 13 and s2.ndim == 3 and s2.shape[-1] != 13:
            s2 = np.transpose(s2, (1, 2, 0))
        rgb = _stretch(s2[..., RGB_BANDS])

        lc = _to_dfc_ids(tifffile.imread(rec["lc"]))
        dfc = _to_dfc_ids(tifffile.imread(rec["dfc"]))
        g_lc, g_dfc = _granularity(lc), _granularity(dfc)

        ax = axes[row, 0]
        ax.imshow(rgb); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"{rec['roi']}\np{rec['patch']}", fontsize=9)
        if row == 0:
            ax.set_title("Sentinel-2 (true colour)", fontsize=10)

        ax = axes[row, 1]
        _draw_labels(ax, lc, cmap, norm)
        if row == 0:
            ax.set_title("MODIS `lc`  (what we trained on)", fontsize=10)
        ax.set_xlabel(f"{g_lc['regions']} regions / median {g_lc['median_region']:.0f} px"
                      f" / dominant {g_lc['dominant']:.2f}", fontsize=8)

        ax = axes[row, 2]
        _draw_labels(ax, dfc, cmap, norm)
        if row == 0:
            ax.set_title("Official `dfc`  (true ground truth)", fontsize=10)
        ax.set_xlabel(f"{g_dfc['regions']} regions / median {g_dfc['median_region']:.0f} px"
                      f" / dominant {g_dfc['dominant']:.2f}", fontsize=8)

    present = sorted(set(np.unique(_to_dfc_ids(tifffile.imread(r["dfc"]))).tolist()
                         + np.unique(_to_dfc_ids(tifffile.imread(r["lc"]))).tolist())
                     for r in chosen)
    present = sorted({v for sub in present for v in sub if v > 0})
    handles = [Patch(facecolor=CLASS_COLORS[v - 1], label=CLASS_NAMES[v - 1])
               for v in present]
    handles.append(Patch(facecolor="#bdbdbd", label="unlabeled"))
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "DFC2020 label provenance — the benchmark target was MODIS, not the contest ground truth\n"
        "Both rasters ship in the official release for the SAME patch. MODIS is ~500 m native; "
        "the official labels are semi-manual at 10 m.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
