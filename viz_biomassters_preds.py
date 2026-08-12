"""
Visualize BioMassters AGB predictions for either modality transition
(s2 -> s1 or s1 -> s2; set --starting_modality).

For each of the first N test samples, produces one row with 8 panels:

  inputs (2)       the starting modality, then the new one
                   (S2 = true colour, S1 = VV/VH backscatter composite)

  predictions (5)  SFT <start>     — stage-0 unimodal teacher on the start modality
                   SFT <new>       — stage-0 unimodal model on the new modality
                   Delulu transfer — real new mod, hallucinated starting mod
                   Delulu peeking  — real starting mod, hallucinated new mod
                   Delulu addition — both modalities real

  target (1)       ground-truth above-ground biomass

The three Delulu columns come from a single SHOT checkpoint and reproduce
evaluate_multimodal()'s three paths exactly: one shared
forward_modality_specific_features() pass, then predict_from_real_modalities()
with a different `real_modalities` tuple per path (see shot.py:274-284).

Unlike the DFC2020 segmentation figure, this is a REGRESSION task: panels are
continuous AGB maps in t/ha sharing one colour scale per row, and each is
annotated with its own RMSE. Following evaluate_multimodal(), pixels with
AGB >= BIOMASSTERS_AGB_MASK_THRESHOLD (400 t/ha) are excluded from the RMSE, so
they are hatched out in the same way the DFC2020 figure hatches ignore-index.

Inputs are temporal ([C, T, H, W], T=12 by default); the input panels show the
median over time, which is far more legible than any single timestep.

Usage:
    python viz_biomassters_preds.py \
        --starting_modality s2 \
        --sft_start checkpoints/sft_evan_base_biomassters_s2_fft_lr0.0005_20260725_075836.pt \
        --sft_new   checkpoints/sft_evan_base_biomassters_s1_fft_lr0.0005_20260725_075947.pt \
        --delulu    checkpoints/delulunet_biomassters_s2s1_peeking_rank1_seed1.pt \
        --num_samples 4 \
        --out figs/biomassters_s2_to_s1.png
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
    BIOMASSTERS_AGB_MASK_THRESHOLD,
    BIOMASSTERS_S1_BANDS,
    BIOMASSTERS_S2_BANDS,
    _N_S2,
)
from data_utils import create_multimodal_batch, get_loaders
from delulunet_main import EvanSegmenter

# True-colour indices within the S2 stack (B02,B03,B04,...): B04=2, B03=1, B02=0.
S2_BANDS = list(BIOMASSTERS_S2_BANDS)
RGB_IDX = [S2_BANDS.index(b) for b in ("B04", "B03", "B02")]
# S1 false colour: VV_asc, VH_asc, and their difference — a conventional
# dual-pol composite that makes structure visible.
S1_BANDS = list(BIOMASSTERS_S1_BANDS)

PANEL_TITLES = {
    "s2": "Input: s2\n(true colour, median over T)",
    "s1": "Input: s1\n(VV/VH asc, median over T)",
}


def _stretch(x: np.ndarray) -> np.ndarray:
    """Per-channel robust 2-98 percentile stretch to [0,1] for display."""
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[-1]):
        band = x[..., c]
        lo, hi = np.nanpercentile(band, 2), np.nanpercentile(band, 98)
        out[..., c] = 0.0 if hi - lo < 1e-8 else np.clip((band - lo) / (hi - lo), 0, 1)
    return out


def _temporal_median(x: torch.Tensor) -> torch.Tensor:
    """Collapse [C, T, H, W] -> [C, H, W] by the per-pixel median over time.

    The median rejects cloudy timesteps far better than the mean, which matters
    for the S2 panel; for S1 it simply reduces speckle.
    """
    return x.median(dim=1).values if x.dim() == 4 else x


def _sample_rmse(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    """RMSE in t/ha over the scored pixels of one tile."""
    if not valid.any():
        return float("nan")
    return float(np.sqrt(((pred[valid] - target[valid]) ** 2).mean()))


def _load_segmenter(path: str, device: str, expect_modalities: tuple[str, ...]) -> EvanSegmenter:
    """Load a segmenter and assert it really carries the modalities we expect.

    Guards against checkpoints saved for a different modality pair; without it a
    mismatch only surfaces as a KeyError deep inside prepare_tokens_with_masks.
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
    out = model(mm)
    return out.squeeze(1).cpu() if out.dim() == 4 else out.cpu()  # [B, H, W]


@torch.no_grad()
def _delulu_predict(model: EvanSegmenter, batch: dict, modality_bands_dict: dict,
                    device: str, start_mod: str, new_mod: str) -> dict[str, torch.Tensor]:
    """Transfer / peeking / addition predictions from one SHOT checkpoint.

    Mirrors evaluate_multimodal() in shot.py: a single modality-specific pass is
    shared across the three paths, which differ only in which modalities are
    treated as real (the rest are hallucinated by the intermediate projectors).
    """
    all_mods = (start_mod, new_mod)
    mm = create_multimodal_batch(batch, modality_bands_dict=modality_bands_dict,
                                 modalities=all_mods)
    mm = {k: v.to(device) for k, v in mm.items()}

    intermediate = model.evan.forward_modality_specific_features(mm)
    paths = {
        "transfer": (new_mod,),      # real new mod, hallucinated starting mod
        "peeking": (start_mod,),     # real starting mod, hallucinated new mod
        "addition": all_mods,        # both real
    }
    out = {}
    for name, real in paths.items():
        sv = model.predict_from_real_modalities(intermediate, real, all_mods)
        out[name] = sv.squeeze(1).cpu() if sv.dim() == 4 else sv.cpu()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--starting_modality", default="s2", choices=["s2", "s1"],
                   help="Modality the stage-0 model was trained on (default: s2).")
    p.add_argument("--new_modality", default=None, choices=["s2", "s1"],
                   help="Modality SHOT adds. Defaults to the other one.")
    p.add_argument("--sft_start", required=True,
                   help="Stage-0 SFT checkpoint for the starting modality.")
    p.add_argument("--sft_new", required=True,
                   help="Stage-0 SFT checkpoint for the new modality.")
    p.add_argument("--delulu", required=True,
                   help="SHOT checkpoint for starting -> new.")
    p.add_argument("--num_samples", type=int, default=4,
                   help="Number of leading test samples to plot (default: 4).")
    p.add_argument("--num_time_steps", type=int, default=12,
                   help="Timesteps to load; MUST match what the checkpoints were "
                        "trained with (default: 12).")
    p.add_argument("--indices", type=int, nargs="+", default=None,
                   help="Explicit test indices to plot instead of the first N.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out", default="figs/biomassters_preds.png")
    args = p.parse_args()

    start_mod = args.starting_modality
    new_mod = args.new_modality or ("s1" if start_mod == "s2" else "s2")
    if new_mod == start_mod:
        p.error("--new_modality must differ from --starting_modality")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Direction: {start_mod} -> {new_mod}")

    _, _, _, _, test_loader, task_config = get_loaders(
        "biomassters", start_mod, batch_size=args.num_samples,
        num_workers=args.num_workers, new_modality=new_mod,
        num_time_steps=args.num_time_steps,
    )
    modality_bands_dict = task_config.modality_bands_dict
    test_ds = test_loader.dataset

    indices = (args.indices[:args.num_samples] if args.indices
               else list(range(args.num_samples)))
    print(f"Test indices: {indices}")

    samples = [test_ds[i] for i in indices]
    batch = {
        "image": torch.stack([s["image"] for s in samples]),
        "mask": torch.stack([s["mask"] for s in samples]),
    }

    print("\n=== Loading models ===")
    sft_start = _load_segmenter(args.sft_start, device, (start_mod,))
    sft_new = _load_segmenter(args.sft_new, device, (new_mod,))
    delulu = _load_segmenter(args.delulu, device, (start_mod, new_mod))

    print("\n=== Running inference ===")
    preds = {
        f"SFT {start_mod}": _sft_predict(sft_start, batch, modality_bands_dict,
                                         start_mod, device),
        f"SFT {new_mod}": _sft_predict(sft_new, batch, modality_bands_dict,
                                       new_mod, device),
    }
    delulu_preds = _delulu_predict(delulu, batch, modality_bands_dict, device,
                                   start_mod, new_mod)
    for name in ("transfer", "peeking", "addition"):
        preds[f"Delulu {name}"] = delulu_preds[name]

    # ---------------------------------------------------------------- plotting
    pred_names = list(preds)                       # 5 prediction columns
    # Input panels are ordered starting-modality first, so the leftmost column is
    # always what the stage-0 model saw regardless of direction.
    input_mods = (start_mod, new_mod)
    col_titles = [
        *(PANEL_TITLES[m] for m in input_mods),
        *pred_names,
        "Ground truth AGB",
    ]
    n_cols = len(col_titles)                       # 2 + 5 + 1 = 8
    n_rows = len(indices)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.6 * n_rows),
                             squeeze=False)

    images = batch["image"]              # [B, C, T, H, W]
    targets = batch["mask"].numpy()      # [B, H, W], raw AGB in t/ha

    for i in range(n_rows):
        target = targets[i]
        # evaluate_multimodal() excludes AGB >= 400 t/ha from the metric, so those
        # pixels are hatched here and left out of the per-tile RMSE.
        scored = target < BIOMASSTERS_AGB_MASK_THRESHOLD
        masked_frac = float((~scored).mean())

        # One colour scale per row, from the target's scored range, so the five
        # prediction panels and the target are directly comparable.
        vmax = float(np.nanpercentile(target[scored], 98)) if scored.any() else 1.0
        vmax = max(vmax, 1.0)

        img = images[i]                  # [C, T, H, W]
        s2 = _temporal_median(img[:_N_S2])
        s1 = _temporal_median(img[_N_S2:])

        def _panel(mod: str) -> np.ndarray:
            if mod == "s2":
                # True colour from B04/B03/B02.
                return _stretch(s2[RGB_IDX].permute(1, 2, 0).numpy())
            # S1 dual-pol composite (VV_asc, VH_asc, VV-VH).
            vv, vh = s1[0].numpy(), s1[1].numpy()
            return _stretch(np.stack([vv, vh, vv - vh], axis=-1))

        # --- Cols 0-1: inputs, starting modality first ---
        for j, mod in enumerate(input_mods):
            axes[i][j].imshow(_panel(mod))

        def _draw_agb(ax, arr: np.ndarray):
            """Draw an AGB map on the row's shared scale, hatching unscored pixels."""
            ax.add_patch(plt.Rectangle(
                (0, 0), 1, 1, transform=ax.transAxes, zorder=0,
                facecolor="#d9d9d9", edgecolor="#9e9e9e", hatch="///", linewidth=0,
            ))
            return ax.imshow(np.ma.masked_where(~scored, arr), cmap="YlGn",
                             vmin=0, vmax=vmax, interpolation="nearest", zorder=1)

        # --- Cols 2-6: predictions, each with its own per-sample RMSE ---
        for j, name in enumerate(pred_names):
            pred = preds[name][i].numpy()
            ax = axes[i][2 + j]
            _draw_agb(ax, pred)
            rmse = _sample_rmse(pred, target, scored)
            ax.set_xlabel(f"RMSE {rmse:.1f} t/ha", fontsize=8, labelpad=2)

        # --- Last col: ground truth, with the row's colourbar ---
        im = _draw_agb(axes[i][n_cols - 1], target)
        cb = fig.colorbar(im, ax=axes[i][n_cols - 1], fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("AGB (t/ha)", fontsize=7)

        for c in range(n_cols):
            axes[i][c].set_xticks([])
            axes[i][c].set_yticks([])
            if c == 0:
                label = f"test #{indices[i]}"
                if masked_frac > 0:
                    label += f"\n{masked_frac:.1%} >400 t/ha"
                axes[i][c].set_ylabel(label, fontsize=9)
            if i == 0:
                axes[i][c].set_title(col_titles[c], fontsize=9)

    fig.suptitle(
        f"BioMassters AGB — {start_mod} $\\rightarrow$ {new_mod}   "
        f"(RMSE is per-tile, not split-level; colour scale is per-row)\n"
        f"Hatched pixels are AGB $\\geq$ {BIOMASSTERS_AGB_MASK_THRESHOLD:.0f} t/ha, "
        f"excluded from the metric by evaluate_multimodal()",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.8)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    pdf_out = os.path.splitext(args.out)[0] + ".pdf"
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {args.out}")
    print(f"[saved] {pdf_out}")


if __name__ == "__main__":
    main()
