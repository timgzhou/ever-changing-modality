import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("res/delulu/hptuned_may5.csv", on_bad_lines="skip")
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

            valid_labels = [f"{m}\n(n={n})" for m, v, n in zip(new_mods, top3_per_mod, run_counts) if len(v) > 0]
            bp = ax.boxplot(
                [v for v in top3_per_mod if len(v) > 0],
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


# python res/delulu/plot_hptuned.py
