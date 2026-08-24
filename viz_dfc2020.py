"""
Visualize DFC2020 segmentation predictions for the s2_rgb -> s2_norgb transition.

For each of the first N test samples, produces one row with 8 panels:

  inputs (2)       S2 true-colour RGB (the starting modality, s2_rgb)
                   S2 no-RGB composite (the new modality, s2_norgb)

  predictions (5)  SFT s2_rgb      — stage-0 unimodal teacher on the start modality
                   SFT s2_norgb    — stage-0 unimodal model on the new modality
                   Delulu transfer — real s2_norgb, hallucinated s2_rgb
                   Delulu peeking  — real s2_rgb,  hallucinated s2_norgb
                   Delulu addition — both modalities real

  target (1)       ground-truth land-cover mask

The three Delulu columns come from a single SHOT checkpoint and reproduce
evaluate_multimodal()'s three paths exactly: one shared
forward_modality_specific_features() pass, then predict_from_real_modalities()
with a different `real_modalities` tuple per path (see shot.py:274-284).
Each panel is annotated with its own per-sample mIoU against the target.

Ignore-index handling: DFC2020_CLASSES maps raw labels 0/8/9/15 to 255, and
value 8 (savanna) is the most common label in this split, so ~50% of the average
tile is unlabeled (median 54% over the first 40 test tiles; the leading four are
49/87/80/65%). Those pixels are excluded from the training loss and from mIoU, so
they are drawn hatched-grey in BOTH the target and the prediction panels.

Sample selection balances two failure modes. Taking the front of the split gives
near-worst-case masking; but selecting purely for low masking is worse, since a
uniform single-class tile is 0% masked and every model scores mIoU 100 on it. So
tiles are required to have >=--min_classes classes and <=--max_ignore masked, then
ranked by class entropy minus a masking penalty. Pass --indices to override.

Usage:
    python viz_dfc2020.py \
        --sft_start   checkpoints/delulu-checkpoints/sft_evan_base_dfc2020_s2_rgb_fft_lr0.0005_20260420_184627.pt \
        --sft_new     checkpoints/delulu-checkpoints/sft_evan_base_dfc2020_s2_norgb_fft_lr0.0005_20260424_002258.pt \
        --delulu      checkpoints/delulu-checkpoints/delulunet_dfc2020_0420_0752.pt \
        --num_samples 4 \
        --out figs/dfc2020_preds.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from data_utils import get_loaders, create_multimodal_batch
from delulunet_main import EvanSegmenter
from dfc2020_official_data_utils import (
    DFC2020_IGNORE_INDEX,
    DFC2020_NUM_CLASSES,
    DFC2020_S2_MEAN,
    DFC2020_S2_STD,
)

# The 10 scored DFC2020 classes (official IGBP2DFC mapping, ids 1-10 shifted to
# 0-9). The old MODIS-backed loader collapsed these to 8 by dropping Savanna and
# Snow/Ice; the official benchmark scores both.
# NOTE: dfc2020_cobench_data_utils uses a DIFFERENT 8-class mapping (Copernicus-
# Bench ignores Background/Savanna/Snow-Ice) -- see COBENCH_CLASS_NAMES there.
CLASS_NAMES = [
    "Forest",
    "Shrubland",
    "Savanna",
    "Grassland",
    "Wetlands",
    "Croplands",
    "Urban/Built-up",
    "Snow/Ice",
    "Barren",
    "Water",
]

# Perceptually distinct, colour-blind-safe-ish palette, one entry per class.
CLASS_COLORS = [
    "#1b7837",  # Forest         — dark green
    "#a6dba0",  # Shrubland      — light green
    "#c2a5cf",  # Savanna        — mauve
    "#d9f0a3",  # Grassland      — yellow-green
    "#80cdc1",  # Wetlands       — teal
    "#f6e8c3",  # Croplands      — wheat
    "#d73027",  # Urban/Built-up — red
    "#f7f7f7",  # Snow/Ice       — near-white
    "#8c510a",  # Barren         — brown
    "#4575b4",  # Water          — blue
]

# S2 band order in the stacked image: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12.
RGB_IDX = [3, 2, 1]  # B4 (R), B3 (G), B2 (B) — matches modality_bands_dict['s2_rgb']

STARTING_MODALITY = "s2_rgb"
NEW_MODALITY = "s2_norgb"


def _denormalize_s2(x: torch.Tensor, band_indices: list[int]) -> np.ndarray:
    """Undo the dataset's z-score to recover raw reflectance DN for display.

    x: [len(band_indices), H, W] z-scored. Returns [H, W, len(band_indices)].
    """
    mean = torch.tensor([DFC2020_S2_MEAN[i] for i in band_indices]).view(-1, 1, 1)
    std = torch.tensor([DFC2020_S2_STD[i] for i in band_indices]).view(-1, 1, 1)
    return (x.cpu() * std + mean).permute(1, 2, 0).numpy()


def _stretch(x: np.ndarray) -> np.ndarray:
    """Per-channel robust 2-98 percentile stretch to [0,1] for display."""
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[-1]):
        band = x[..., c]
        lo, hi = np.nanpercentile(band, 2), np.nanpercentile(band, 98)
        out[..., c] = 0.0 if hi - lo < 1e-8 else np.clip((band - lo) / (hi - lo), 0, 1)
    return out


def _per_sample_scores(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return (mIoU, pixel accuracy) as percentages for a single tile.

    Ignore-index pixels are excluded from both. mIoU averages over classes present
    in the union of pred and target, so a class the model hallucinates counts as a
    miss (union > 0, intersection = 0) rather than being skipped.

    Both numbers are reported because neither alone is trustworthy on one 96x96
    tile: mIoU over a single present class collapses to 0 or 100 with nothing in
    between, while pixel accuracy flatters a tile dominated by one class.
    """
    valid = target != DFC2020_IGNORE_INDEX
    if not valid.any():
        return float("nan"), float("nan")
    p, t = pred[valid], target[valid]

    ious = []
    for c in range(DFC2020_NUM_CLASSES):
        pred_c, tgt_c = (p == c), (t == c)
        union = np.logical_or(pred_c, tgt_c).sum()
        if union == 0:
            continue  # class in neither pred nor target — not a miss, just absent
        ious.append(np.logical_and(pred_c, tgt_c).sum() / union)

    miou = float(np.mean(ious) * 100) if ious else float("nan")
    acc = float((p == t).mean() * 100)
    return miou, acc


def _select_samples(test_ds, num_samples: int, max_ignore: float, scan_limit: int,
                    min_classes: int = 3) -> list[int]:
    """Pick informative test tiles: multi-class targets that are mostly labeled.

    Two competing failure modes have to be balanced here.

    Masking: DFC2020_CLASSES maps raw labels 0/8/9/15 to 255, and value 8
    (savanna) is the most common label in this split, so ~50% of the average tile
    is unlabeled (median 54% over the first 40; the leading four are 49/87/80/65%).
    A heavily masked target is mostly holes and its mIoU rests on a few pixels.

    Triviality: minimizing the ignore fraction alone is actively harmful, because
    a tile that is 100% one class has 0% masked. Selecting on that criterion
    returns uniform Water/Forest tiles where every model scores exactly 100.0 and
    the comparison shows nothing.

    So: require at least `min_classes` distinct classes and no more than
    `max_ignore` masked, then rank by a balanced score that rewards class variety
    (Shannon entropy over the class histogram) and penalizes masking. Constraints
    relax step-wise if too few tiles qualify, rather than silently returning a
    short or degenerate selection.
    """
    n = min(scan_limit, len(test_ds))
    print(f"Scanning {n} test tiles (need >={min_classes} classes, "
          f"<={max_ignore:.0%} unlabeled)...")

    stats = []
    for i in range(n):
        mask = test_ds[i]["mask"].numpy()
        ign = float((mask == DFC2020_IGNORE_INDEX).mean())
        valid = mask[mask != DFC2020_IGNORE_INDEX]
        if valid.size == 0:
            continue
        counts = np.bincount(valid, minlength=DFC2020_NUM_CLASSES)
        counts = counts[counts > 0]
        p = counts / counts.sum()
        entropy = float(-(p * np.log(p)).sum())  # 0 for a single-class tile
        stats.append({"idx": i, "ignore": ign, "n_classes": len(counts),
                      "entropy": entropy,
                      # Entropy dominates; masking is a mild penalty.
                      "score": entropy - 0.5 * ign})

    # Relax the constraints in stages rather than returning something degenerate.
    for need_cls, allow_ign in ((min_classes, max_ignore),
                                (min_classes, 0.60),
                                (2, 0.60),
                                (2, 1.0)):
        pool = [s for s in stats if s["n_classes"] >= need_cls and s["ignore"] <= allow_ign]
        if len(pool) >= num_samples:
            if (need_cls, allow_ign) != (min_classes, max_ignore):
                print(f"  relaxed to >={need_cls} classes, <={allow_ign:.0%} unlabeled "
                      f"({len(pool)} candidates)")
            break
    else:
        print("  [warn] no tile meets even the relaxed constraints; "
              "falling back to the most diverse available")
        pool = stats

    pool.sort(key=lambda s: -s["score"])
    keep = pool[:num_samples]
    for s in keep:
        print(f"  test #{s['idx']}: {s['n_classes']} classes, "
              f"{s['ignore']:.0%} unlabeled, entropy {s['entropy']:.2f}")
    return [s["idx"] for s in keep]


def _load_segmenter(path: str, device: str, expect_modalities: tuple[str, ...]) -> EvanSegmenter:
    """Load a segmenter and assert it really carries the modalities we expect.

    Several rows in res/delulu/hptuned_apr21.csv are mislabeled — timestamped
    checkpoint names collided across runs, so a row claiming s2_rgb->s2_norgb can
    point at, say, an s1->s2 model. Without this check the mismatch only surfaces
    as a KeyError inside prepare_tokens_with_masks, long after loading.
    """
    model = EvanSegmenter.from_checkpoint(path, device)
    have = set(model.evan.patch_embedders.keys())
    missing = set(expect_modalities) - have
    if missing:
        raise SystemExit(
            f"[error] {path}\n"
            f"        expected modalities {sorted(expect_modalities)}, "
            f"but checkpoint has {sorted(have)} (missing {sorted(missing)}).\n"
            f"        This checkpoint is for a different modality pair."
        )
    model.eval()
    return model


@torch.no_grad()
def _sft_predict(model: EvanSegmenter, batch: dict, modality_bands_dict: dict,
                 modality: str, device: str) -> torch.Tensor:
    """Unimodal stage-0 prediction: forward the single modality it was trained on."""
    mm = create_multimodal_batch(batch, modality_bands_dict=modality_bands_dict,
                                 modalities=(modality,))
    mm = {k: v.to(device) for k, v in mm.items()}
    return model(mm).argmax(1).cpu()  # [B, H, W]


@torch.no_grad()
def _delulu_predict(model: EvanSegmenter, batch: dict, modality_bands_dict: dict,
                    device: str) -> dict[str, torch.Tensor]:
    """Transfer / peeking / addition predictions from one SHOT checkpoint.

    Mirrors evaluate_multimodal() in shot.py: a single modality-specific pass is
    shared across the three paths, which differ only in which modalities are
    treated as real (the rest are hallucinated by the intermediate projectors).
    """
    all_mods = (STARTING_MODALITY, NEW_MODALITY)
    mm = create_multimodal_batch(batch, modality_bands_dict=modality_bands_dict,
                                 modalities=all_mods)
    mm = {k: v.to(device) for k, v in mm.items()}

    intermediate = model.evan.forward_modality_specific_features(mm)
    paths = {
        # real s2_norgb, hallucinated s2_rgb
        "transfer": (NEW_MODALITY,),
        # real s2_rgb, hallucinated s2_norgb
        "peeking": (STARTING_MODALITY,),
        # both real
        "addition": all_mods,
    }
    return {
        name: model.predict_from_real_modalities(intermediate, real, all_mods).argmax(1).cpu()
        for name, real in paths.items()
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sft_start", required=True,
                   help=f"Stage-0 SFT checkpoint trained on {STARTING_MODALITY}.")
    p.add_argument("--sft_new", required=True,
                   help=f"Stage-0 SFT checkpoint trained on {NEW_MODALITY}.")
    p.add_argument("--delulu", required=True,
                   help=f"SHOT checkpoint for {STARTING_MODALITY} -> {NEW_MODALITY}.")
    p.add_argument("--num_samples", type=int, default=4,
                   help="Number of test samples to plot (default: 4).")
    p.add_argument("--max_ignore", type=float, default=0.35,
                   help="Skip test tiles whose target is more than this fraction "
                        "ignore-index. DFC2020 masks savanna, which costs ~50%% of "
                        "the average tile (default: 0.35).")
    p.add_argument("--min_classes", type=int, default=3,
                   help="Require at least this many distinct classes in the target. "
                        "Guards against uniform tiles, where every model trivially "
                        "scores mIoU 100 (default: 3).")
    p.add_argument("--scan_limit", type=int, default=300,
                   help="How many leading test tiles to scan when selecting "
                        "(default: 300).")
    p.add_argument("--indices", type=int, nargs="+", default=None,
                   help="Explicit test indices to plot, bypassing ignore-fraction "
                        "selection (e.g. --indices 0 1 2 3 for the literal first four).")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out", default="figs/dfc2020_preds.png")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    _, _, _, _, test_loader, task_config = get_loaders(
        "dfc2020", STARTING_MODALITY, batch_size=args.num_samples,
        num_workers=args.num_workers, new_modality=NEW_MODALITY,
    )
    modality_bands_dict = task_config.modality_bands_dict
    test_ds = test_loader.dataset

    if args.indices:
        indices = args.indices[:args.num_samples]
        print(f"\nUsing explicit test indices: {indices}")
    else:
        print()
        indices = _select_samples(test_ds, args.num_samples, args.max_ignore,
                                  args.scan_limit, args.min_classes)

    # Collate the chosen samples by hand — the loader is unshuffled but we need
    # arbitrary indices, not a contiguous leading slice.
    samples = [test_ds[i] for i in indices]
    batch = {
        "image": torch.stack([s["image"] for s in samples]),
        "mask": torch.stack([s["mask"] for s in samples]),
    }

    print("\n=== Loading models ===")
    sft_start = _load_segmenter(args.sft_start, device, (STARTING_MODALITY,))
    sft_new = _load_segmenter(args.sft_new, device, (NEW_MODALITY,))
    delulu = _load_segmenter(args.delulu, device, (STARTING_MODALITY, NEW_MODALITY))

    print("\n=== Running inference ===")
    preds = {
        f"SFT {STARTING_MODALITY}": _sft_predict(sft_start, batch, modality_bands_dict,
                                                 STARTING_MODALITY, device),
        f"SFT {NEW_MODALITY}": _sft_predict(sft_new, batch, modality_bands_dict,
                                            NEW_MODALITY, device),
    }
    delulu_preds = _delulu_predict(delulu, batch, modality_bands_dict, device)
    for name in ("transfer", "peeking", "addition"):
        preds[f"Delulu {name}"] = delulu_preds[name]

    # ---------------------------------------------------------------- plotting
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, DFC2020_NUM_CLASSES), cmap.N)

    pred_names = list(preds)                       # 5 prediction columns
    col_titles = [
        f"Input: {STARTING_MODALITY}\n(true colour)",
        f"Input: {NEW_MODALITY}\n(B8,B11,B5 composite)",
        *pred_names,
        "Ground truth",
    ]
    n_cols = len(col_titles)                       # 2 + 5 + 1 = 8
    n_rows = args.num_samples

    # h_pad keeps the per-panel "mIoU / acc" xlabels from colliding with the row
    # below (they overlapped at the default spacing).
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.1 * n_cols, 2.5 * n_rows),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.22)

    images = batch["image"]      # [B, 15, 96, 96], z-scored
    targets = batch["mask"].numpy()  # [B, 96, 96]

    def _draw_labels(ax, arr: np.ndarray, ignore: np.ndarray) -> None:
        """Draw a class map, rendering ignore-index pixels as hatched grey.

        The hatch sits on an axes-spanning background patch that the masked class
        image is drawn over, so unlabeled area is unmistakably 'not evaluated'
        rather than reading as a pale class or as white background.
        """
        ax.add_patch(plt.Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes, zorder=0,
            facecolor="#d9d9d9", edgecolor="#9e9e9e", hatch="///", linewidth=0,
        ))
        ax.imshow(np.ma.masked_where(ignore, arr), cmap=cmap, norm=norm,
                  interpolation="nearest", zorder=1)

    for i in range(n_rows):
        target = targets[i]
        ignore = target == DFC2020_IGNORE_INDEX
        ign_frac = float(ignore.mean())

        # --- Col 0: S2 true colour ---
        rgb = _stretch(_denormalize_s2(images[i][RGB_IDX], RGB_IDX))
        axes[i][0].imshow(rgb)

        # --- Col 1: false-colour composite of three s2_norgb bands ---
        # B8 (NIR, idx 7), B11 (SWIR, idx 11), B5 (red-edge, idx 4) — a standard
        # vegetation/moisture composite showing what the new modality carries.
        norgb_idx = [7, 11, 4]
        false_colour = _stretch(_denormalize_s2(images[i][norgb_idx], norgb_idx))
        axes[i][1].imshow(false_colour)

        # --- Cols 2-6: predictions, each with its own per-sample mIoU ---
        # Predictions are masked by the SAME ignore map as the target: the models
        # emit a class everywhere, but those pixels are excluded from the loss and
        # from mIoU, so showing them would invite comparing unscored regions.
        for j, name in enumerate(pred_names):
            pred = preds[name][i].numpy()
            ax = axes[i][2 + j]
            _draw_labels(ax, pred, ignore)
            miou, acc = _per_sample_scores(pred, target)
            ax.set_xlabel(f"mIoU {miou:.1f}  /  acc {acc:.1f}", fontsize=8, labelpad=2)

        # --- Last col: ground truth ---
        _draw_labels(axes[i][n_cols - 1], target, ignore)

        for c in range(n_cols):
            axes[i][c].set_xticks([])
            axes[i][c].set_yticks([])
            if c == 0:
                n_cls = len(np.unique(target[~ignore]))
                axes[i][c].set_ylabel(
                    f"test #{indices[i]}\n{n_cls} classes, {ign_frac:.0%} unlabeled",
                    fontsize=9)
            if i == 0:
                axes[i][c].set_title(col_titles[c], fontsize=9)

    # Shared class legend below the grid, plus the ignore swatch.
    handles = [Patch(facecolor=CLASS_COLORS[k], label=CLASS_NAMES[k])
               for k in range(DFC2020_NUM_CLASSES)]
    handles.append(Patch(facecolor="#d9d9d9", edgecolor="#9e9e9e", hatch="///",
                         label="unlabeled (not scored)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(
        f"DFC2020 — {STARTING_MODALITY} $\\rightarrow$ {NEW_MODALITY}   "
        f"(mIoU / pixel-accuracy are per-tile, not split-level)\n"
        f"Hatched pixels are ignore-index: DFC2020 masks savanna (raw label 8), "
        f"~50% of the average tile. Samples chosen for class diversity and low masking.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94), h_pad=1.8)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    pdf_out = os.path.splitext(args.out)[0] + ".pdf"
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {args.out}")
    print(f"[saved] {pdf_out}")


if __name__ == "__main__":
    main()
