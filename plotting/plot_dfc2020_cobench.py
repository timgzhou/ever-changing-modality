"""
DFC2020 / Copernicus-Bench SFT sweep — test mIoU by modality, decoder, DINO init,
and against val.

Run:  python plotting/plot_dfc2020_cobench.py [out.png]
"""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV = Path("res/train_sft/dfc2020_cobench.csv")
OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else "res/train_sft/dfc2020_cobench_test_metric.png")

# validated categorical slots (dataviz validator, light surface, --pairs all: PASS)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "violet": "#4a3aa7"}
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#1a1a19", "#4a4a48", "#8a8a86", "#e3e3df"

rows = list(csv.DictReader(CSV.open()))
for r in rows:
    r["test_metric"] = float(r["test_metric"]); r["val_metric"] = float(r["val_metric"])
METRIC = rows[0]["metric_name"]

SINGLES = ["s1", "s2_rgb", "s2_norgb"]
PAIRS = ["s2_rgb+s1", "s2_norgb+s1", "s2_rgb+s2_norgb"]
MODS = SINGLES + PAIRS
DEC = [("linear", C["blue"]), ("upernet", C["orange"])]
# published Copernicus-Bench supervised ViT-B/16 baselines (test mIoU)
REF = {"S1": 50.8, "S2": 66.2}

plt.rcParams.update({"font.size": 9, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlecolor": INK, "axes.spines.top": False,
    "axes.spines.right": False})

fig, axes = plt.subplots(1, 4, figsize=(16.5, 5.0),
                         gridspec_kw={"width_ratios": [1.75, 1, 1, 1.12]})
fig.subplots_adjust(left=0.045, right=0.99, top=0.76, bottom=0.24, wspace=0.28)

tlo = min(r["test_metric"] for r in rows); thi = max(r["test_metric"] for r in rows)
pad = (thi - tlo) * 0.10
ylim = (tlo - pad, max(thi, max(REF.values())) + pad)


def refs(ax, label=False):
    for name, v in REF.items():
        ax.axhline(v, color=INK3, lw=1.0, ls=(0, (4, 3)), zorder=1)
        if label:
            ax.text(0.985, v, f"CoBench {name} {v} ", va="top", ha="right", fontsize=7.5,
                    color=INK3, transform=ax.get_yaxis_transform(),
                    bbox=dict(fc=SURFACE, ec="none", pad=1.0))


def strip(ax, key, levels, colors, title, xlabel):
    for i, lev in enumerate(levels):
        vals = sorted(r["test_metric"] for r in rows if r[key] == lev)
        n = len(vals)
        xs = [i + (j - (n - 1) / 2) * (0.44 / max(n - 1, 1)) for j in range(n)]
        ax.scatter(xs, vals, s=26, color=colors[i], alpha=.85, linewidths=1.0,
                   edgecolors=SURFACE, zorder=3)
        m = sum(vals) / n
        ax.hlines(m, i - .33, i + .33, color=colors[i], lw=2.5, zorder=4)
        ax.text(i, ylim[1] - pad * .05, f"{m:.1f}", ha="center", va="top",
                fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels)
    ax.set_xlim(-.6, len(levels) - .4); ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10.5, pad=10, loc="left")
    ax.set_xlabel(xlabel, labelpad=6)
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)


# ── panel 1: modality x decoder, singles then pairs ─────────────────────────
ax = axes[0]
for i, m in enumerate(MODS):
    for k, (dec, col) in enumerate(DEC):
        vals = sorted(r["test_metric"] for r in rows
                      if r["modality"] == m and r["decoder"] == dec)
        if not vals:
            continue
        off = (k - 0.5) * 0.34
        n = len(vals)
        xs = [i + off + (j - (n - 1) / 2) * (0.22 / max(n - 1, 1)) for j in range(n)]
        ax.scatter(xs, vals, s=22, color=col, alpha=.85, linewidths=1.0,
                   edgecolors=SURFACE, zorder=3)
        ax.hlines(sum(vals) / n, i + off - .15, i + off + .15, color=col, lw=2.4, zorder=4)
ax.axvline(2.5, color=INK3, lw=1.0, ls=(0, (3, 3)), zorder=1)
ax.set_xticks(range(len(MODS))); ax.set_xticklabels(MODS, rotation=30, ha="right")
ax.set_xlim(-.6, len(MODS) - .4); ax.set_ylim(*ylim)
ax.set_ylabel(f"test {METRIC}")
ax.set_title(f"test {METRIC} by modality and decoder", fontsize=10.5, pad=10, loc="left")
ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
refs(ax, label=True)
ax.text(1.0, ylim[0] + pad * .15, "single modality", ha="center", fontsize=8.5, color=INK2)
ax.text(4.0, ylim[0] + pad * .15, "two modalities", ha="center", fontsize=8.5, color=INK2)
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=6, mfc=c, mec=SURFACE, label=d)
                   for d, c in DEC],
          loc="lower right", frameon=False, fontsize=8, labelcolor=INK2)

strip(axes[1], "decoder", ["linear", "upernet"], [C["blue"], C["orange"]],
      f"test {METRIC} by decoder", "decoder"); refs(axes[1])
strip(axes[2], "dino_init", ["False", "True"], [C["aqua"], C["violet"]],
      f"test {METRIC} by DINO init", "dino_init"); refs(axes[2])

# ── panel 4: test vs val ────────────────────────────────────────────────────
ax = axes[3]
slo = min(tlo, min(r["val_metric"] for r in rows))
shi = max(thi, max(r["val_metric"] for r in rows))
spad = (shi - slo) * .07
slim = (slo - spad, shi + spad)
ax.plot(slim, slim, color=INK3, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.annotate("y = x", xy=(slim[1] - spad, slim[1] - spad), xytext=(-4, 6),
            textcoords="offset points", color=INK3, fontsize=8, ha="right", va="bottom")
for r in rows:
    ax.scatter(r["val_metric"], r["test_metric"], s=30,
               color=C["orange"] if r["decoder"] == "upernet" else C["blue"],
               marker="^" if "+" in r["modality"] else "o",
               alpha=.85, linewidths=1.0, edgecolors=SURFACE, zorder=3)
ax.set_xlim(*slim); ax.set_ylim(*slim); ax.set_aspect("equal")
ax.set_xlabel(f"val {METRIC}", labelpad=6); ax.set_ylabel(f"test {METRIC}")
ax.set_title(f"test vs val {METRIC}", fontsize=10.5, pad=10, loc="left")
ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True)


def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); rk = [0.] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                rk[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return rk
    rx, ry = rank(xs), rank(ys); n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** .5
    return num / den if den else float("nan")


rho = spearman([r["val_metric"] for r in rows], [r["test_metric"] for r in rows])
ax.text(.03, .97, f"Spearman ρ = {rho:.2f}   n = {len(rows)}", transform=ax.transAxes,
        va="top", ha="left", fontsize=8.5, color=INK2)
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", ms=6.5, mfc=C["orange"], mec=SURFACE, label="upernet"),
    Line2D([], [], marker="o", ls="", ms=6.5, mfc=C["blue"], mec=SURFACE, label="linear"),
    Line2D([], [], marker="o", ls="", ms=6.5, mfc=INK3, mec=SURFACE, label="single modality"),
    Line2D([], [], marker="^", ls="", ms=6.5, mfc=INK3, mec=SURFACE, label="two modalities")],
    loc="upper left", bbox_to_anchor=(0, -.20), ncol=2, frameon=False, fontsize=8,
    labelcolor=INK2, handletextpad=.4, columnspacing=1.2, borderpad=0)

fig.suptitle(f"DFC2020 · evan_base · SFT sweep — test {METRIC} "
             f"({len(rows)} runs, each dot one run; rule = mean)",
             x=.045, ha="left", y=.955, fontsize=12.5, color=INK)
fig.text(.045, .90,
         "Copernicus-Bench official split (3156/986/986, 8 classes) · linear (96) + upernet (24) · "
         "combined modalities now train on BOTH inputs (bimodal fix, 2026-08-20) · "
         "dashed = published supervised ViT-B/16 baselines",
         ha="left", fontsize=8.5, color=INK3)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170)
print("wrote", OUT)
