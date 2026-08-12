import pandas as pd
import os

csvs = {
    "eurosat": "res/baselines/freematch/baseline_freematch_eurosat.csv",
    "benv2":   "res/baselines/freematch/baseline_freematch_benv2.csv",
}

frames = []
for dataset, path in csvs.items():
    if not os.path.isfile(path):
        print(f"[warn] missing: {path}")
        continue
    df = pd.read_csv(path)
    df["dataset"] = dataset
    frames.append(df)

df = pd.concat(frames, ignore_index=True)

# Group by no_strong_aug too, so the augmentation-parity ablation is reported
# as its own arm rather than being pooled with the main sweep.
if "no_strong_aug" not in df.columns:
    df["no_strong_aug"] = False   # pre-ablation CSVs

KEYS = ["dataset", "modality", "use_dino_weights", "no_strong_aug"]

# Select best run per key by best_val_metric, report best_val_test_metric.
counts = df.groupby(KEYS).size().reset_index(name="n_runs")

best = (
    df.sort_values("best_val_metric", ascending=False)
    .groupby(KEYS, as_index=False)
    .first()[KEYS + ["model_type", "train_mode", "learning_rate", "lambda_u",
                     "lambda_e", "ema_momentum", "metric_name",
                     "best_val_metric", "best_val_test_metric"]]
    .merge(counts, on=KEYS)
    .sort_values(KEYS)
)

for dataset, group in best.groupby("dataset"):
    print(f"\n=== {dataset} ===")
    print(group.drop(columns="dataset").to_string(index=False))

# python res/baselines/freematch/freematch_best.py
