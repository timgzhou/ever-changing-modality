"""
Qualitative DFC2020 predictions on the Copernicus-Bench split.

For each modality, loads the best-by-val checkpoint from
res/train_sft/dfc2020_cobench.csv and renders, for a fixed random sample of
test tiles: the modality's own raw input, the model's prediction, and the
ground truth.

Val is a reliable selector on this split (test-vs-val Spearman rho = 0.99), so
best-by-val is also effectively best-by-test.

Usage:
    DFC2020_SPLIT=cobench python viz_dfc2020_cobench_preds.py [--n 4] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

os.environ.setdefault("DFC2020_SPLIT", "cobench")

from data_utils import get_loaders, create_multimodal_batch          # noqa: E402
from delulunet_main import EvanSegmenter                              # noqa: E402
from dfc2020_cobench_data_utils import (                              # noqa: E402
    COBENCH_CLASS_NAMES, COBENCH_NUM_CLASSES,
)
from dfc2020_official_data_utils import DFC2020_IGNORE_INDEX          # noqa: E402

RESULTS = Path("res/train_sft/dfc2020_cobench.csv")
OUT = Path("figs/dfc2020_cobench_predictions.png")

MODALITIES = ["s1", "s2_rgb", "s2_norgb",
              "s2_rgb+s1", "s2_norgb+s1", "s2_rgb+s2_norgb"]

# Class colours: 8 land-cover classes. Chosen to read as land cover rather than
# as a categorical series (greens for vegetation, blue water, grey urban), with
# distinct lightness steps so the map stays legible in greyscale/CVD.
CLASS_COLORS = [
    "#1b6b3a",  # 0 Forest
    "#8fbf5f",  # 1 Shrubland
    "#d9e06b",  # 2 Grassland
    "#4aa3c7",  # 3 Wetlands
    "#e0a33a",  # 4 Croplands
    "#9b9b9b",  # 5 Urban/Built-up
    "#c8ab86",  # 6 Barren
    "#2a5ea8",  # 7 Water
]
IGNORE_COLOR = "#f2f2ef"

SURFACE, INK, INK2, INK3 = "#fcfcfb", "#1a1a19", "#4a4a48", "#8a8a86"


def _stretch(x: np.ndarray, lo=2.0, hi=98.0) -> np.ndarray:
    """Percentile contrast stretch to [0,1] for display."""
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    if b <= a:
        b = a + 1e-6
    return np.clip((x - a) / (b - a), 0, 1)


def render_input(img: torch.Tensor, modality: str) -> tuple[np.ndarray, str]:
    """
    Build a displayable image for one modality from the 15-channel stack
    (S2 ch0-12 then S1 ch13-14). Returns (HxWx3 or HxW array, caption).
    """
    a = img.numpy()
    if modality == "s1":
        vv, vh = _stretch(a[13]), _stretch(a[14])
        return np.stack([vv, vh, vv], -1), "S1 (VV,VH,VV)"
    if modality == "s2_rgb":
        return np.stack([_stretch(a[3]), _stretch(a[2]), _stretch(a[1])], -1), "S2 RGB (B4,B3,B2)"
    if modality == "s2_norgb":
        # false colour from the non-RGB bands: NIR / SWIR1 / red-edge
        return np.stack([_stretch(a[7]), _stretch(a[11]), _stretch(a[5])], -1), \
               "S2 non-RGB (B8,B11,B6)"
    raise ValueError(modality)


def panels_for(modality: str) -> list[str]:
    """Which input panels a (possibly combined) modality entry shows."""
    return modality.split("+")


def infer_modalities(modality: str) -> tuple[str, ...]:
    """
    Which modalities to FEED the model — all of them, matching training.

    History: until 2026-08-20 train_sft.py passed only `args.modalities[0]` into
    the training loop, so a combined `s2_rgb+s1` checkpoint was a single-modality
    s2_rgb model carrying untrained s1 parameters. Feeding both at inference was
    off-distribution and collapsed the prediction (59.18 -> 6.02 mIoU on the
    s2_rgb+s1 upernet checkpoint). Those checkpoints are archived alongside
    res/train_sft/dfc2020_cobench_BROKEN_bimodal_pre20260820.csv; every combined
    entry in the live CSV was trained on BOTH inputs, so both are fed here.

    If you point this script at a pre-fix checkpoint, feed only the first
    modality instead.
    """
    return tuple(modality.split("+"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="test samples to show")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", dest="out_dir", type=str, default="figs")
    ap.add_argument("--decoder", type=str, default="linear",
                    choices=["linear", "upernet"],
                    help="which decoder's checkpoints to visualize")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = list(csv.DictReader(RESULTS.open()))
    for r in rows:
        r["val_metric"] = float(r["val_metric"])
        r["test_metric"] = float(r["test_metric"])

    best = {}
    for m in MODALITIES:
        cand = [r for r in rows if r["modality"] == m
                and r.get("decoder", "linear") == args.decoder]
        if not cand:
            print(f"[warn] no rows for {m}, skipping")
            continue
        b = max(cand, key=lambda r: r["val_metric"])
        if not Path(b["saved_checkpoint"]).exists():
            print(f"[warn] checkpoint missing for {m}: {b['saved_checkpoint']}")
            continue
        best[m] = b

    if not best:
        raise SystemExit("no usable checkpoints found")

    # Fixed sample of test tiles. The loader is deterministic given the split
    # files, so a fixed seed picks the same tiles on every run.
    _, _, _, _, test_loader, task_config = get_loaders(
        "dfc2020", "s2_rgb", batch_size=1, num_workers=2, new_modality="s1")
    test_ds = test_loader.dataset
    rng = random.Random(args.seed)
    idxs = sorted(rng.sample(range(len(test_ds)), args.n))
    print(f"test tiles (seed {args.seed}): {idxs}")
    samples = [test_ds[i] for i in idxs]

    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(list(range(COBENCH_NUM_CLASSES + 1)), cmap.N)

    def show_mask(ax, mask: np.ndarray):
        disp = np.ma.masked_where(mask == DFC2020_IGNORE_INDEX, mask)
        cmap_ = cmap.copy()
        cmap_.set_bad(IGNORE_COLOR)
        ax.imshow(disp, cmap=cmap_, norm=norm, interpolation="nearest")

    # ---- one figure per test tile ----------------------------------------
    # Rows = modality, columns = input | prediction | ground truth. A combined
    # entry (e.g. s2_rgb+s1) has two inputs, so the input cell splits into two
    # side-by-side thumbnails via a nested gridspec -- that keeps the three
    # columns aligned across every row.
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    blocks = {m: panels_for(m) for m in best}
    mods = list(blocks.keys())
    nrows = len(mods)

    plt.rcParams.update({"font.size": 8.5, "figure.facecolor": SURFACE,
                         "text.color": INK, "axes.titlecolor": INK})

    out_paths = []
    with torch.no_grad():
        for si, sample in enumerate(samples):
            tile = idxs[si]
            mask = sample["mask"].numpy()

            fig, axes = plt.subplots(nrows, 3, figsize=(8.4, 2.28 * nrows + 1.5),
                                     squeeze=False)
            fig.subplots_adjust(left=0.175, right=0.985, top=0.885, bottom=0.085,
                                wspace=0.05, hspace=0.09)
            for ax in axes.ravel():
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)

            for r, m in enumerate(mods):
                panels = blocks[m]

                # --- column 1: input(s) ---
                host = axes[r][0]
                host.axis("off")
                sub = GridSpecFromSubplotSpec(1, len(panels),
                                              subplot_spec=host.get_subplotspec(),
                                              wspace=0.04)
                for k, pmod in enumerate(panels):
                    cax = fig.add_subplot(sub[0, k])
                    im, cap = render_input(sample["image"], pmod)
                    cax.imshow(im, interpolation="nearest")
                    cax.set_xticks([]); cax.set_yticks([])
                    for sp in cax.spines.values():
                        sp.set_visible(False)
                    cax.set_xlabel(cap, fontsize=7, color=INK3, labelpad=2)
                    if r == 0 and k == 0:
                        cax.set_title("input", fontsize=9.5, pad=6, color=INK,
                                      loc="left")

                # --- column 2: prediction ---
                batch = {"image": sample["image"].unsqueeze(0)}
                modal_input = create_multimodal_batch(
                    batch, task_config.modality_bands_dict,
                    infer_modalities(m))
                modal_input = {k: v.to(device) for k, v in modal_input.items()}
                pred = MODELS[m](modal_input).argmax(1)[0].cpu().numpy()
                show_mask(axes[r][1], pred)

                # --- column 3: ground truth (repeated per row for direct compare) ---
                show_mask(axes[r][2], mask)

                if r == 0:
                    axes[r][1].set_title("prediction", fontsize=9.5, pad=6,
                                         color=INK, loc="left")
                    axes[r][2].set_title("ground truth", fontsize=9.5, pad=6,
                                         color=INK, loc="left")

                axes[r][0].text(-0.045, 0.5,
                                f"{m}\n{best[m]['test_metric']:.1f} mIoU",
                                transform=axes[r][0].transAxes, va="center",
                                ha="right", fontsize=9, color=INK,
                                linespacing=1.4)

            handles = [Patch(facecolor=CLASS_COLORS[i], label=COBENCH_CLASS_NAMES[i])
                       for i in range(COBENCH_NUM_CLASSES)]
            handles.append(Patch(facecolor=IGNORE_COLOR, label="ignored"))
            fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                       fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.56, 0.004))

            fig.suptitle(f"DFC2020 test tile {tile} — prediction by modality",
                         x=0.006, ha="left", y=0.985, fontsize=13, color=INK)
            fig.text(0.006, 0.947,
                     f"Copernicus-Bench split · best-by-val checkpoint per modality · "
                     f"{args.decoder} decoder · combined entries train on and are fed BOTH modalities",
                     ha="left", fontsize=8.5, color=INK3)

            outp = Path(args.out_dir) / f"dfc2020_cobench_pred_tile{tile}_{args.decoder}.png"
            outp.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(outp, dpi=165)
            plt.close(fig)
            out_paths.append(outp)
            print("wrote", outp)

    print(f"\nwrote {len(out_paths)} figures")
    return

    handles = [Patch(facecolor=CLASS_COLORS[i], label=COBENCH_CLASS_NAMES[i])
               for i in range(COBENCH_NUM_CLASSES)]
    handles.append(Patch(facecolor=IGNORE_COLOR, label="ignored"))
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False,
               fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.57, 0.004))

    fig.suptitle("DFC2020 · Copernicus-Bench split — best-by-val checkpoint per modality",
                 x=0.006, ha="left", y=0.985, fontsize=13, color=INK)
    fig.text(0.006, 0.952,
             f"{args.n} random test tiles (seed {args.seed}) · linear decoder · "
             "rows: the three raw inputs, then one prediction row per modality, then ground truth",
             ha="left", fontsize=8.5, color=INK3)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=165)
    print("wrote", args.out)


if __name__ == "__main__":
    # models are loaded once, before main()'s render loop
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    import sys as _sys
    _dec = "linear"
    if "--decoder" in _sys.argv:
        _dec = _sys.argv[_sys.argv.index("--decoder") + 1]
    _rows = list(csv.DictReader(RESULTS.open()))
    for _r in _rows:
        _r["val_metric"] = float(_r["val_metric"])
    MODELS = {}
    for _m in MODALITIES:
        _c = [r for r in _rows if r["modality"] == _m
              and r.get("decoder", "linear") == _dec]
        if not _c:
            continue
        _b = max(_c, key=lambda r: r["val_metric"])
        _p = _b["saved_checkpoint"]
        if not Path(_p).exists():
            continue
        _mod = EvanSegmenter.from_checkpoint(_p, device=_device)
        _mod.eval()
        MODELS[_m] = _mod
        print(f"loaded {_m}: {Path(_p).name}")
    main()
