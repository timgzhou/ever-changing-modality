import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

df = pd.read_csv("res/delulu/hptuned_apr21.csv", on_bad_lines="skip")
df = df[df["dataset"].isin(["dfc2020", "benv2", "eurosat"])]
df = df[df["model_arch"].isin(["evan_base", "evan_large"])]
df["teacher_test_metric"] = pd.to_numeric(df["teacher_test_metric"], errors="coerce")
for col in ["valchecked_transfer", "valchecked_peek", "valchecked_add", "valchecked_add_ens",
            "valchecked_val_transfer", "valchecked_val_peek", "valchecked_val_add", "valchecked_val_add_ens"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

TASKS = ["transfer", "peek", "addition", "ens_addition"]
VALCHECKED_COLS = ["valchecked_transfer", "valchecked_peek", "valchecked_add", "valchecked_add_ens"]
VAL_SEL_COLS = ["valchecked_val_transfer", "valchecked_val_peek", "valchecked_val_add", "valchecked_val_add_ens"]
TASK_LABELS = ["Transfer", "Peek", "Addition", "Ens. Addition"]

datasets = sorted(df["dataset"].unique())
model_archs = sorted(df["model_arch"].unique())

for dataset in datasets:
    for arch in model_archs:
        sub = df[(df["dataset"] == dataset) & (df["model_arch"] == arch)]
        if sub.empty:
            continue

        starting_mods = sorted(sub["starting_modality"].unique())
        n_rows = len(starting_mods)
        n_cols = len(TASKS)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
        fig.suptitle(f"{dataset} / {arch}", fontsize=14, fontweight="bold")

        for r, start_mod in enumerate(starting_mods):
            row_df = sub[sub["starting_modality"] == start_mod]
            teacher_val = row_df["teacher_test_metric"].iloc[0] if not row_df.empty else None
            metric_name = row_df["metric_name"].iloc[0] if not row_df.empty else ""

            for c, (task, vcol, vsel, task_label) in enumerate(zip(TASKS, VALCHECKED_COLS, VAL_SEL_COLS, TASK_LABELS)):
                ax = axes[r][c]
                new_mods = sorted(row_df["new_modality"].unique())
                colors = plt.cm.tab10.colors
                top3_per_mod = []
                run_counts = []
                for new_mod in new_mods:
                    mod_df = row_df[row_df["new_modality"] == new_mod].dropna(subset=[vsel, vcol])
                    run_counts.append(len(row_df[row_df["new_modality"] == new_mod]))
                    top3_per_mod.append(mod_df.nlargest(3, vsel)[vcol].values)

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

                # teacher horizontal line
                if teacher_val is not None:
                    ax.axhline(teacher_val, color="red", linewidth=1.2, linestyle="--", label=f"teacher ({teacher_val:.1f})")

                ax.set_title(f"{task_label}", fontsize=9)
                ax.set_xlabel("")
                if c == 0:
                    ax.set_ylabel(f"{start_mod}\n{metric_name}", fontsize=8)
                ax.tick_params(axis="y", labelsize=7)
                ax.legend(fontsize=6, loc="lower right")

        plt.tight_layout()
        out = f"res/delulu/hptuned_{dataset}_{arch}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"saved {out}")


# python res/delulu/plot_hptuned.py