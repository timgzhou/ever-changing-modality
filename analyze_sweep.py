"""
Analyze sweep results: find best hyperparameters per metric (val_transfer,
val_peeking, val_addition, val_ens_addition) and save to JSON.

Usage:
    python analyze_sweep.py [--csv res/sweep_results_128.csv] [--out best_hparams.json]
"""

import argparse
import json
import pandas as pd

METRICS = ["val_transfer", "val_peeking", "val_addition", "val_ens_addition"]

HPARAM_COLS = [
    "ssl_lr",
    "weight_decay",
    "epochs",
    "mask_ratio",
    "modality_dropout",
    "labeled_frequency",
    "labeled_start_fraction",
    "use_mae",
    "use_latent",
]


def find_best(df: pd.DataFrame, metric: str) -> dict:
    idx = df[metric].idxmax()
    row = df.loc[idx]
    result = {col: row[col].item() if hasattr(row[col], "item") else row[col]
              for col in HPARAM_COLS if col in df.columns}
    result[metric] = float(row[metric])
    result["wandb_run_id"] = row["wandb_run_id"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="res/sweep_results_128.csv")
    parser.add_argument("--out", default="res/best_hparams.json")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    best = {}
    for metric in METRICS:
        if metric not in df.columns:
            print(f"WARNING: {metric} not found in CSV, skipping")
            continue
        best[metric] = find_best(df, metric)
        print(f"{metric}: {best[metric][metric]:.2f}  (run {best[metric]['wandb_run_id']})")

    with open(args.out, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
