import pandas as pd
import os
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

X_TRAIN = "preprocessing/X_train.csv"
X_TEST = "preprocessing/X_test.csv"
Y_TRAIN = "preprocessing/y_train.csv"
Y_TEST = "preprocessing/y_test.csv"
CLASS_MAPPING = "preprocessing/class_mapping.csv"

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

class_mapping = pd.read_csv(CLASS_MAPPING)
class_names = class_mapping["Attack_Type"].tolist()

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")
print(f"Features        : {X_train.shape[1]}")

# --------------------------------------------------
# Train Decision Tree
# --------------------------------------------------

print("\nTraining Decision Tree...")

model = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced",
    max_depth=None
)

model.fit(X_train, y_train)

print("Training completed.")

# --------------------------------------------------
# Predictions
# --------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n======================================")
print("DECISION TREE RESULTS")
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
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv(
    "results/decision_tree_confusion_matrix.csv"
)

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "results/decision_tree_feature_importance.csv",
    index=False
)

# --------------------------------------------------
# Save metrics
# --------------------------------------------------

metrics = pd.DataFrame([{
    "Model": "Decision Tree",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1
}])

metrics.to_csv(
    "results/decision_tree_metrics.csv",
    index=False
)

# --------------------------------------------------
# Save model
# --------------------------------------------------

joblib.dump(
    model,
    "saved_models/decision_tree.pkl"
)

print("\nFiles saved:")
print("results/decision_tree_metrics.csv")
print("results/decision_tree_confusion_matrix.csv")
print("results/decision_tree_feature_importance.csv")
print("saved_models/decision_tree.pkl")

print("\nDecision Tree completed successfully.")