import pandas as pd
import glob
import os

os.makedirs("results", exist_ok=True)

files = [
    "results/random_forest_metrics.csv",
    "results/decision_tree_metrics.csv",
    "results/xgboost_metrics.csv",
    "results/isolation_forest_tuning.csv"
]

results = []

# Supervised model results
for file in files[:3]:
    df = pd.read_csv(file)
    results.append(df)

# Best Isolation Forest result
iso_df = pd.read_csv(files[3])
best_iso = iso_df.loc[iso_df["F1_Score"].idxmax()]

results.append(
    pd.DataFrame([{
        "Model": "Isolation Forest",
        "Accuracy": best_iso["Accuracy"],
        "Precision": best_iso["Precision"],
        "Recall": best_iso["Recall"],
        "F1_Score": best_iso["F1_Score"]
    }])
)

final_results = pd.concat(
    results,
    ignore_index=True
)

final_results.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\n======================================")
print("MODEL COMPARISON")
print("======================================")

print(
    final_results.to_string(index=False)
)

print("\nSaved:")
print("results/model_comparison.csv")