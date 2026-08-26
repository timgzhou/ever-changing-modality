"""
Qualitative DeluluNet predictions vs teacher and oracle, per SHOT scenario.

One figure per scenario (transfer / peeking / addition). Rows are test samples;
columns are:

    raw input        the modality (or modalities) the scenario feeds at test time
    teacher SFT      the frozen stage-0 unimodal teacher, on ITS OWN modality
    DeluluNet        the scenario's prediction path
    oracle SFT       supervised full-split model on the SCENARIO'S test input
    ground truth

The three scenarios differ only in `real_modalities` passed to
predict_from_real_modalities (mirroring evaluate_multimodal in shot.py):

    transfer  real = (new,)            start modality hallucinated
    peeking   real = (start,)          new modality hallucinated
    addition  real = (start, new)      nothing hallucinated

so the oracle differs per scenario: SFT(new) for transfer, SFT(start) for
peeking, SFT(start+new) for addition -- always the supervised model that sees
the same test-time input.

Usage:
    DFC2020_SPLIT=cobench python viz_dfc2020_delulu_preds.py \
        --start s2_norgb --new s2_rgb --n 4 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import os
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
    COBENCH_CLASS_NAMES, COBENCH_NUM_CLASSES)
from dfc2020_official_data_utils import DFC2020_IGNORE_INDEX          # noqa: E402

SFT_CSV = Path("res/train_sft/dfc2020_cobench.csv")
TEACHERS = Path("artifacts/sft_teachers.json")

CLASS_COLORS = ["#1b6b3a", "#8fbf5f", "#d9e06b", "#4aa3c7",
                "#e0a33a", "#9b9b9b", "#c8ab86", "#2a5ea8"]
IGNORE_COLOR = "#f2f2ef"
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#1a1a19", "#4a4a48", "#8a8a86"

SCENARIOS = ("transfer", "peeking", "addition")


def _stretch(x, lo=2.0, hi=98.0):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / (b - a if b > a else 1e-6), 0, 1)


def render_input(img: torch.Tensor, modality: str):
    a = img.numpy()
    if modality == "s1":
        vv, vh = _stretch(a[13]), _stretch(a[14])
        return np.stack([vv, vh, vv], -1), "S1 (VV,VH,VV)"
    if modality == "s2_rgb":
        return np.stack([_stretch(a[3]), _stretch(a[2]), _stretch(a[1])], -1), "S2 RGB"
    if modality == "s2_norgb":
        return np.stack([_stretch(a[7]), _stretch(a[11]), _stretch(a[5])], -1), "S2 non-RGB"
    raise ValueError(modality)


def sft_checkpoint(modality: str, split: str, decoder="upernet"):
    """Best-by-val SFT checkpoint for a modality/split."""
    best = None
    for r in csv.DictReader(SFT_CSV.open()):
        if (r["modality"] == modality and r["train_split"] == split
                and r["decoder"] == decoder):
            v = float(r["val_metric"])
            if best is None or v > best[0]:
                best = (v, r["saved_checkpoint"], float(r["test_metric"]))
    return best


def real_mods_for(scenario, start, new):
    return {"transfer": (new,), "peeking": (start,),
            "addition": (start, new)}[scenario]


def oracle_modality(scenario, start, new):
    """The SFT modality whose test-time input matches this scenario."""
    if scenario == "transfer":
        return new
    if scenario == "peeking":
        return start
    for cand in (f"{start}+{new}", f"{new}+{start}"):
        for r in csv.DictReader(SFT_CSV.open()):
            if r["modality"] == cand:
                return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--delulu", default=None, help="delulu checkpoint (auto if unset)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", dest="out_dir", default="figs")
    ap.add_argument("--decoder", default="upernet")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start, new = args.start, args.new

    # ---- models ----------------------------------------------------------
    dl_path = args.delulu
    if dl_path is None:
        cands = sorted(Path("checkpoints").glob(
            f"delulunet_dfc2020_{start}_to_{new}_{args.decoder}_ll*_seed0.pt"))
        if not cands:
            raise SystemExit(f"no delulu checkpoint for {start}->{new}")
        dl_path = str(cands[0])
    print("delulu :", dl_path)
    delulu = EvanSegmenter.from_checkpoint(dl_path, device=device); delulu.eval()

    t = sft_checkpoint(start, "split1", args.decoder)
    if t is None:
        raise SystemExit(f"no split1 SFT teacher for {start}")
    print(f"teacher: {t[1]}  (test {t[2]:.2f})")
    teacher = EvanSegmenter.from_checkpoint(t[1], device=device); teacher.eval()

    # one oracle per scenario
    oracles = {}
    for sc in SCENARIOS:
        om = oracle_modality(sc, start, new)
        o = sft_checkpoint(om, "full", args.decoder) if om else None
        if o is None:
            print(f"[warn] no full-split oracle for {sc} ({om})")
            oracles[sc] = None
            continue
        print(f"oracle {sc:9s}: {om:16s} {o[1].split('/')[-1]}  (test {o[2]:.2f})")
        m = EvanSegmenter.from_checkpoint(o[1], device=device); m.eval()
        oracles[sc] = (m, om, o[2])

    # ---- data ------------------------------------------------------------
    _, _, _, _, test_loader, tc = get_loaders(
        "dfc2020", start, batch_size=1, num_workers=2, new_modality=new)
    ds = test_loader.dataset
    import random
    idxs = sorted(random.Random(args.seed).sample(range(len(ds)), args.n))
    print("test tiles:", idxs)
    samples = [ds[i] for i in idxs]

    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(list(range(COBENCH_NUM_CLASSES + 1)), cmap.N)

    def show_mask(ax, m):
        d = np.ma.masked_where(m == DFC2020_IGNORE_INDEX, m)
        c = cmap.copy(); c.set_bad(IGNORE_COLOR)
        ax.imshow(d, cmap=c, norm=norm, interpolation="nearest")

    plt.rcParams.update({"font.size": 8.5, "figure.facecolor": SURFACE,
                         "text.color": INK, "axes.titlecolor": INK})
    all_mods = (start, new)
    out_paths = []

    with torch.no_grad():
        for sc in SCENARIOS:
            real = real_mods_for(sc, start, new)
            ncol = len(real) + 4          # inputs + teacher + delulu + oracle + GT
            fig, axes = plt.subplots(args.n, ncol,
                                     figsize=(2.0 * ncol + 0.6, 2.05 * args.n + 1.5),
                                     squeeze=False)
            fig.subplots_adjust(left=0.055, right=0.99, top=0.855, bottom=0.10,
                                wspace=0.05, hspace=0.06)
            for ax in axes.ravel():
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)

            for si, smp in enumerate(samples):
                col = 0
                for rm in real:                       # raw input(s)
                    im, cap = render_input(smp["image"], rm)
                    axes[si][col].imshow(im, interpolation="nearest")
                    if si == 0:
                        axes[si][col].set_title(f"input\n{cap}", fontsize=8.5,
                                                pad=6, color=INK)
                    col += 1

                batch = {"image": smp["image"].unsqueeze(0)}

                # teacher: its own modality only
                ti = create_multimodal_batch(batch, tc.modality_bands_dict, (start,))
                ti = {k: v.to(device) for k, v in ti.items()}
                tp = teacher(ti).argmax(1)[0].cpu().numpy()
                show_mask(axes[si][col], tp)
                if si == 0:
                    axes[si][col].set_title(f"teacher SFT\n{start} · {t[2]:.1f}",
                                            fontsize=8.5, pad=6, color=INK)
                col += 1

                # delulu: the scenario's path
                mi = create_multimodal_batch(batch, tc.modality_bands_dict, all_mods)
                mi = {k: v.to(device) for k, v in mi.items()}
                inter = delulu.evan.forward_modality_specific_features(mi)
                dp = delulu.predict_from_real_modalities(
                    inter, tuple(real), tuple(all_mods)).argmax(1)[0].cpu().numpy()
                show_mask(axes[si][col], dp)
                if si == 0:
                    axes[si][col].set_title(f"DeluluNet\n{sc}", fontsize=8.5,
                                            pad=6, color=INK)
                col += 1

                # oracle: supervised model on the same test-time input
                ax = axes[si][col]
                if oracles[sc] is not None:
                    om_model, om, om_test = oracles[sc]
                    oi = create_multimodal_batch(batch, tc.modality_bands_dict,
                                                 tuple(om.split("+")))
                    oi = {k: v.to(device) for k, v in oi.items()}
                    op = om_model(oi).argmax(1)[0].cpu().numpy()
                    show_mask(ax, op)
                    if si == 0:
                        ax.set_title(f"oracle SFT\n{om} · {om_test:.1f}",
                                     fontsize=8.5, pad=6, color=INK)
                else:
                    ax.axis("off")
                col += 1

                show_mask(axes[si][col], smp["mask"].numpy())
                if si == 0:
                    axes[si][col].set_title("ground truth", fontsize=8.5,
                                            pad=6, color=INK)
                axes[si][0].text(-0.06, 0.5, f"tile {idxs[si]}",
                                 transform=axes[si][0].transAxes, rotation=90,
                                 va="center", ha="right", fontsize=8, color=INK3)

            handles = [Patch(facecolor=CLASS_COLORS[i], label=COBENCH_CLASS_NAMES[i])
                       for i in range(COBENCH_NUM_CLASSES)]
            handles.append(Patch(facecolor=IGNORE_COLOR, label="ignored"))
            fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                       fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.53, 0.004))

            realtxt = " + ".join(real)
            fig.suptitle(f"DFC2020 · {start} → +{new} · {sc.upper()}",
                         x=0.055, ha="left", y=0.975, fontsize=12.5, color=INK)
            fig.text(0.055, 0.925,
                     f"test-time input = {realtxt}"
                     + ("  (start modality hallucinated)" if sc == "transfer"
                        else "  (new modality hallucinated)" if sc == "peeking"
                        else "  (both real)")
                     + " · Copernicus-Bench split · upernet decoder",
                     ha="left", fontsize=8.5, color=INK3)

            p = Path(args.out_dir) / f"dfc2020_delulu_{start}_to_{new}_{sc}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=165); plt.close(fig)
            out_paths.append(p); print("wrote", p)

    print(f"\nwrote {len(out_paths)} figures")


if __name__ == "__main__":
    main()
