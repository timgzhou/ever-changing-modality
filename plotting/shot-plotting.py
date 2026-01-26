import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and combine data from all dates
dates = ["jan22", "jan24", "jan25", "jan25b"]
dfs = []
for date in dates:
    df = pd.read_csv(f"res/modality-transfer_{date}.csv")
    df["date"] = date
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df.sort_values(by=['starting_modality', 'eval_type', 'real_modality'], inplace=True)
df = df[df["num_supervised_epochs"] == 32]

g = sns.FacetGrid(df, row="starting_modality", col="eval_type", hue="date", sharex=False, sharey=False)
g.map(sns.scatterplot, "real_modality", "test_acc", alpha=.7)
g.set_titles(row_template="Start: {row_name}\n", col_template="{col_name}")
g.add_legend(bbox_to_anchor=(1.05, 0.5), loc='center left')

plt.suptitle("Modality Transfer: Test Accuracy by Evaluation Type", y=1.02)
plt.tight_layout()
plt.savefig("artifacts/shot-plotting_combined.png", bbox_inches='tight', dpi=150)
print("Saved to artifacts/shot-plotting_combined.png")

# python plotting/shot-plotting.py
