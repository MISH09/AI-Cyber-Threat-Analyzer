import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("evaluation/graphs", exist_ok=True)

# =========================================================
# 1. Confusion Matrix - Random Forest
# =========================================================

cm_rf = pd.read_csv(
    "results/random_forest_confusion_matrix.csv",
    index_col=0
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    "evaluation/graphs/random_forest_confusion_matrix.png",
    dpi=300
)

plt.close()


# =========================================================
# 2. Confusion Matrix - Decision Tree
# =========================================================

cm_dt = pd.read_csv(
    "results/decision_tree_confusion_matrix.csv",
    index_col=0
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm_dt,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    "evaluation/graphs/decision_tree_confusion_matrix.png",
    dpi=300
)

plt.close()


# =========================================================
# 3. Confusion Matrix - XGBoost
# =========================================================

cm_xgb = pd.read_csv(
    "results/xgboost_confusion_matrix.csv",
    index_col=0
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm_xgb,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    "evaluation/graphs/xgboost_confusion_matrix.png",
    dpi=300
)

plt.close()


# =========================================================
# 4. Random Forest Feature Importance
# =========================================================

rf_features = pd.read_csv(
    "results/random_forest_feature_importance.csv"
)

rf_features = rf_features.sort_values(
    by="Importance",
    ascending=False
).head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    rf_features["Feature"][::-1],
    rf_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features - Random Forest")

plt.tight_layout()

plt.savefig(
    "evaluation/graphs/random_forest_feature_importance.png",
    dpi=300
)

plt.close()


# =========================================================
# 5. XGBoost Feature Importance
# =========================================================

xgb_features = pd.read_csv(
    "results/xgboost_feature_importance.csv"
)

xgb_features = xgb_features.sort_values(
    by="Importance",
    ascending=False
).head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    xgb_features["Feature"][::-1],
    xgb_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features - XGBoost")

plt.tight_layout()

plt.savefig(
    "evaluation/graphs/xgboost_feature_importance.png",
    dpi=300
)

plt.close()


print("\n======================================")
print("MODEL DETAILS GENERATED SUCCESSFULLY")
print("======================================")

print("\nGenerated files:")
print("1. random_forest_confusion_matrix.png")
print("2. decision_tree_confusion_matrix.png")
print("3. xgboost_confusion_matrix.png")
print("4. random_forest_feature_importance.png")
print("5. xgboost_feature_importance.png")