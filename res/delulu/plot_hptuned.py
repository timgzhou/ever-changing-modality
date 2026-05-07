import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

df = pd.read_csv("res/delulu/hptuned_masking_may6.csv", on_bad_lines="skip")
df = df[df["dataset"].isin(["dfc2020", "benv2", "eurosat"])]

METRIC_NAME = {"benv2": "mAP", "dfc2020": "mIoU", "eurosat": "Acc"}

TASKS        = ["transfer",      "peeking",      "addition",      "ens_addition"]
TEST_COLS    = ["test_transfer", "test_peeking", "test_addition", "test_ens_addition"]
VAL_COLS     = ["val_transfer",  "val_peeking",  "val_addition",  "val_ens_addition"]
TASK_LABELS  = ["Transfer",      "Peek",         "Addition",      "Ens. Addition"]
SELECT_BY    = ["transfer",      "peeking",      "addition",      "addition"]

for col in TEST_COLS + VAL_COLS + ["teacher_test_metric"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

datasets = sorted(df["dataset"].unique())

for dataset in datasets:
    sub = df[df["dataset"] == dataset]
    if sub.empty:
        continue

    metric_name = METRIC_NAME.get(dataset, "")
    starting_mods = sorted(sub["starting_modality"].unique())
    n_rows = len(starting_mods)
    n_cols = len(TASKS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle(f"{dataset}", fontsize=14, fontweight="bold")

    for r, start_mod in enumerate(starting_mods):
        row_df = sub[sub["starting_modality"] == start_mod]
        teacher_val = row_df["teacher_test_metric"].dropna().iloc[0] if not row_df.empty else None

        for c, (task, test_col, val_col, task_label, sel) in enumerate(zip(TASKS, TEST_COLS, VAL_COLS, TASK_LABELS, SELECT_BY)):
            ax = axes[r][c]
            task_df = row_df[row_df["select_by"] == sel]
            new_mods = sorted(task_df["new_modality"].unique())
            colors = plt.cm.tab10.colors
            top3_per_mod = []
            run_counts = []
            for new_mod in new_mods:
                mod_df = task_df[task_df["new_modality"] == new_mod].dropna(subset=[val_col, test_col])
                run_counts.append(len(task_df[task_df["new_modality"] == new_mod]))
                top3_per_mod.append(mod_df.nlargest(3, val_col)[test_col].values)

            valid = [(m, v, n) for m, v, n in zip(new_mods, top3_per_mod, run_counts) if len(v) > 0]
            if not valid:
                ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray")
                ax.tick_params(axis="x", labelsize=6, labelrotation=30)
            else:
                valid_labels = [f"{m}\n(n={n})" for m, v, n in valid]
                bp = ax.boxplot(
                    [v for _, v, _ in valid],
                    labels=valid_labels,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black", markersize=4),
                    medianprops=dict(visible=False),
                )
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                ax.tick_params(axis="x", labelsize=6, labelrotation=30)

            if teacher_val is not None:
                ax.axhline(teacher_val, color="red", linewidth=1.2, linestyle="--", label=f"teacher ({teacher_val:.1f})")

            ax.set_title(task_label, fontsize=9)
            ax.set_xlabel("")
            if c == 0:
                ax.set_ylabel(f"{start_mod}\n{metric_name}", fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
            ax.legend(fontsize=6, loc="lower right")

    plt.tight_layout()
    out = f"res/delulu/hptuned_{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


# --- val vs test correlation plots ---
VAL_TEST_PAIRS = [
    ("val_peeking",  "test_peeking",  "Peek",     True),
    ("val_transfer", "test_transfer", "Transfer", False),
    ("val_addition", "test_addition", "Addition", False),
]

for dataset in datasets:
    sub = df[df["dataset"] == dataset]
    if sub.empty:
        continue

    metric_name = METRIC_NAME.get(dataset, "")
    mod_pairs = sorted(sub[["starting_modality", "new_modality"]].drop_duplicates().itertuples(index=False, name=None))
    n_rows = len(mod_pairs)
    n_cols = len(VAL_TEST_PAIRS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    fig.suptitle(f"{dataset} — val vs test correlation", fontsize=14, fontweight="bold")

    for r, (start_mod, new_mod) in enumerate(mod_pairs):
        pair_df = sub[(sub["starting_modality"] == start_mod) & (sub["new_modality"] == new_mod)]

        for c, (val_col, test_col, task_label, plot_identity) in enumerate(VAL_TEST_PAIRS):
            ax = axes[r][c]
            xy = pair_df[[val_col, test_col]].dropna()
            x, y = xy[val_col].values, xy[test_col].values

            if len(x) < 2:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray")
            else:
                r_val, p_val = pearsonr(x, y)
                ax.scatter(x, y, s=18, alpha=0.6, color="steelblue", edgecolors="none")

                if plot_identity:
                    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
                    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)

                p_str = f"p={p_val:.2e}" if p_val < 0.001 else f"p={p_val:.3f}"
                ax.set_title(f"{task_label}  r={r_val:.3f} ({p_str})", fontsize=8)

            if c == 0:
                ax.set_ylabel(f"{start_mod}→{new_mod}\ntest {metric_name}", fontsize=7)
            else:
                ax.set_ylabel(f"test {metric_name}", fontsize=7)
            ax.set_xlabel(f"val {metric_name}", fontsize=7)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = f"res/delulu/hptuned_{dataset}_val_vs_test.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


# --- focused val-vs-test: EuroSAT rgb→vre  &  DFC2020 S2→S1 ---
FOCUS_CASES = [
    dict(dataset="eurosat", start="rgb",    new="vre",  label="EuroSAT  (RGB → VRE)", metric="Accuracy (%)"),
    dict(dataset="dfc2020", start="s2",     new="s1",   label="DFC2020  (S2 → S1)",  metric="mIoU (%)"),
]

FOCUS_TASKS = [
    ("val_peeking",  "test_peeking",  "Peeking",  True,  None),
    ("val_transfer", "test_transfer", "Transfer", False, "Transfer-Teacher Agreement on Unlabeled Val"),
    ("val_addition", "test_addition", "Addition", False, "Transfer-Teacher Agreement on Unlabeled Val"),
]

palette = sns.color_palette("Set2", n_colors=len(FOCUS_TASKS))

n_rows = len(FOCUS_CASES)
n_cols = len(FOCUS_TASKS)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(3 * n_cols, 2.8 * n_rows),
    squeeze=False,
)

for r, case in enumerate(FOCUS_CASES):
    sub = df[(df["dataset"] == case["dataset"]) &
             (df["starting_modality"] == case["start"]) &
             (df["new_modality"] == case["new"])]
    metric = case["metric"]

    for c, (val_col, test_col, task_label, plot_identity, xlab_override) in enumerate(FOCUS_TASKS):
        ax = axes[r][c]
        color = palette[c]

        xy = sub[[val_col, test_col]].dropna()
        x, y = xy[val_col].values, xy[test_col].values

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        if len(x) < 2:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
        else:
            r_val, _ = pearsonr(x, y)
            ax.scatter(x, y, s=28, alpha=0.75, color=color,
                       edgecolors="white", linewidths=0.4, zorder=3)

            if plot_identity:
                lims = [min(x.min(), y.min()) - 0.5, max(x.max(), y.max()) + 0.5]
                ax.plot(lims, lims, color="black", linewidth=1.0,
                        linestyle="--", alpha=0.45, zorder=2, label="identity")
                ax.set_xlim(lims); ax.set_ylim(lims)

            ax.annotate(f"r = {r_val:.3f}",
                        xy=(0.05, 0.93), xycoords="axes fraction",
                        fontsize=9, fontstyle="italic",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7, ec="none"))

        # column header (only top row)
        if r == 0:
            ax.set_title(task_label, fontsize=11, fontweight="bold", pad=6)

        # row label on left column
        if c == 0:
            ax.set_ylabel(f"{case['label']}\nTest {metric}", fontsize=9)
        else:
            ax.set_ylabel(f"Test {metric}", fontsize=9)

        ax.set_xlabel(xlab_override if xlab_override else f"Val {metric}", fontsize=9)
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)

plt.tight_layout(h_pad=2.5, w_pad=2.0)
out = "res/delulu/hptuned_val_vs_test_focused.pdf"
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"saved {out}")


# python res/delulu/plot_hptuned.py
