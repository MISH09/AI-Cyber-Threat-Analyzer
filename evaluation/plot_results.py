import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("evaluation/graphs", exist_ok=True)

# Load model comparison results
df = pd.read_csv("results/model_comparison.csv")

# Convert metrics to percentage
metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]

# =========================================================
# 1. Accuracy Comparison
# =========================================================

plt.figure(figsize=(9, 6))

plt.bar(df["Model"], df["Accuracy"] * 100)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 105)

for i, value in enumerate(df["Accuracy"] * 100):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "evaluation/graphs/model_accuracy.png",
    dpi=300
)

plt.close()


# =========================================================
# 2. Precision, Recall and F1 Comparison
# =========================================================

plot_df = df.melt(
    id_vars="Model",
    value_vars=["Precision", "Recall", "F1_Score"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=plot_df,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.ylabel("Score")
plt.xlabel("Model")
plt.title("Precision, Recall and F1-Score Comparison")
plt.ylim(0, 1.05)

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "evaluation/graphs/model_metrics_comparison.png",
    dpi=300
)

plt.close()


# =========================================================
# 3. Model Comparison Heatmap
# =========================================================

heatmap_df = df.set_index("Model")[metrics]

plt.figure(figsize=(10, 5))

sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".4f",
    cmap="Blues"
)

plt.title("Model Performance Heatmap")
plt.tight_layout()

plt.savefig(
    "evaluation/graphs/model_performance_heatmap.png",
    dpi=300
)

plt.close()


print("\n======================================")
print("GRAPHS CREATED SUCCESSFULLY")
print("======================================")

print("\nGenerated files:")

print("1. evaluation/graphs/model_accuracy.png")
print("2. evaluation/graphs/model_metrics_comparison.png")
print("3. evaluation/graphs/model_performance_heatmap.png")