import pandas as pd
import numpy as np
import os

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

X_TRAIN = "preprocessing/X_train.csv"
X_TEST = "preprocessing/X_test.csv"
Y_TRAIN = "preprocessing/y_train.csv"
Y_TEST = "preprocessing/y_test.csv"

os.makedirs("results", exist_ok=True)

print("Loading data...")

X_train = pd.read_csv(X_TRAIN)
X_test = pd.read_csv(X_TEST)

y_train = pd.read_csv(Y_TRAIN)["Attack_Type"]
y_test = pd.read_csv(Y_TEST)["Attack_Type"]

# Train only on BENIGN traffic
X_train_benign = X_train[y_train == 0]

# Binary ground truth:
# 0 = BENIGN
# 1 = ATTACK
y_true = np.where(y_test == 0, 0, 1)

contamination_values = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30
]

results = []

print("\nStarting Isolation Forest tuning...\n")

for contamination in contamination_values:

    print(f"Testing contamination = {contamination}")

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_benign)

    predictions = model.predict(X_test)

    # Isolation Forest:
    # 1  = normal
    # -1 = anomaly

    y_pred = np.where(predictions == 1, 0, 1)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    results.append({
        "Contamination": contamination,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)

print("\n======================================")
print("ISOLATION FOREST TUNING RESULTS")
print("======================================")

print(
    results_df.to_string(index=False)
)

results_df.to_csv(
    "results/isolation_forest_tuning.csv",
    index=False
)

best = results_df.iloc[0]

print("\n======================================")
print("BEST CONFIGURATION")
print("======================================")

print(
    f"Contamination: {best['Contamination']}"
)

print(
    f"Accuracy: {best['Accuracy']:.4f}"
)

print(
    f"Precision: {best['Precision']:.4f}"
)

print(
    f"Recall: {best['Recall']:.4f}"
)

print(
    f"F1 Score: {best['F1_Score']:.4f}"
)

print(
    "\nSaved to: results/isolation_forest_tuning.csv"
)