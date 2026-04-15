import pandas as pd
import os

csvs = {
    "eurosat": "res/baselines/mixmatch/baseline_mixmatch_eurosat.csv",
    "benv2":   "res/baselines/mixmatch/baseline_mixmatch_benv2.csv",
    "dfc2020": "res/baselines/mixmatch/baseline_mixmatch_dfc2020.csv",
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

# Select best run per (dataset, modality, use_dino_weights) by best_val_metric,
# report best_val_test_metric as the final score.
counts = df.groupby(["dataset", "modality", "use_dino_weights"]).size().reset_index(name="n_runs")

best = (
    df.sort_values("best_val_metric", ascending=False)
    .groupby(["dataset", "modality", "use_dino_weights"], as_index=False)
    .first()[["dataset", "modality", "use_dino_weights", "model_type",
              "train_mode", "learning_rate", "lambda_u", "alpha",
              "metric_name", "best_val_metric", "best_val_test_metric"]]
    .merge(counts, on=["dataset", "modality", "use_dino_weights"])
    .sort_values(["dataset", "modality", "use_dino_weights"])
)

for dataset, group in best.groupby("dataset"):
    print(f"\n=== {dataset} ===")
    print(group.drop(columns="dataset").to_string(index=False))

# python res/baselines/mixmatch/mixmatch_best.py
