import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

X_TRAIN = "preprocessing/X_train.csv"
X_TEST = "preprocessing/X_test.csv"
Y_TRAIN = "preprocessing/y_train.csv"
Y_TEST = "preprocessing/y_test.csv"

os.makedirs("results", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading training and testing data...")

X_train = pd.read_csv(X_TRAIN)
X_test = pd.read_csv(X_TEST)

y_train = pd.read_csv(Y_TRAIN)["Attack_Type"]
y_test = pd.read_csv(Y_TEST)["Attack_Type"]

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")
print(f"Features        : {X_train.shape[1]}")

# --------------------------------------------------
# Select only BENIGN traffic for training
# --------------------------------------------------

benign_mask = y_train == 0

X_train_benign = X_train[benign_mask]

print(f"\nBENIGN training samples: {len(X_train_benign):,}")

# --------------------------------------------------
# Train Isolation Forest
# --------------------------------------------------

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=100,
    contamination=0.20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_benign)

print("Training completed.")

# --------------------------------------------------
# Predict anomalies
# --------------------------------------------------

print("\nDetecting anomalies...")

predictions = model.predict(X_test)

# Isolation Forest:
#  1  = normal
# -1  = anomaly

# Convert to:
# 0 = BENIGN
# 1 = ATTACK

y_true_binary = np.where(y_test == 0, 0, 1)

y_pred_binary = np.where(predictions == 1, 0, 1)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

accuracy = accuracy_score(
    y_true_binary,
    y_pred_binary
)

precision = precision_score(
    y_true_binary,
    y_pred_binary,
    zero_division=0
)

recall = recall_score(
    y_true_binary,
    y_pred_binary,
    zero_division=0
)

f1 = f1_score(
    y_true_binary,
    y_pred_binary,
    zero_division=0
)

print("\n======================================")
print("ISOLATION FOREST RESULTS")
print("======================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# --------------------------------------------------
# Classification Report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_true_binary,
        y_pred_binary,
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )
)

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_true_binary,
    y_pred_binary
)

cm_df = pd.DataFrame(
    cm,
    index=["BENIGN", "ATTACK"],
    columns=["BENIGN", "ATTACK"]
)

cm_df.to_csv(
    "results/isolation_forest_confusion_matrix.csv"
)

# --------------------------------------------------
# Save metrics
# --------------------------------------------------

metrics = pd.DataFrame([{
    "Model": "Isolation Forest",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1
}])

metrics.to_csv(
    "results/isolation_forest_metrics.csv",
    index=False
)

# --------------------------------------------------
# Save anomaly scores
# --------------------------------------------------

anomaly_scores = model.decision_function(X_test)

score_df = pd.DataFrame({
    "True_Label": y_test,
    "Prediction": y_pred_binary,
    "Anomaly_Score": anomaly_scores
})

score_df.to_csv(
    "results/isolation_forest_predictions.csv",
    index=False
)

# --------------------------------------------------
# Save model
# --------------------------------------------------

joblib.dump(
    model,
    "saved_models/isolation_forest.pkl"
)

print("\nFiles saved:")
print("results/isolation_forest_metrics.csv")
print("results/isolation_forest_confusion_matrix.csv")
print("results/isolation_forest_predictions.csv")
print("saved_models/isolation_forest.pkl")

print("\nIsolation Forest completed successfully.")