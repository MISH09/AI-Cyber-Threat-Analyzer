import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

INPUT_FILE = "preprocessing/clean_dataset.csv"

print("Loading cleaned dataset...")
df = pd.read_csv(INPUT_FILE)

# Features and target
X = df.drop(columns=["Attack_Type"])
y = df["Attack_Type"]

# Encode attack labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nAttack classes:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{i}: {label}")

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\n======================================")
print("DATA SPLIT")
print("======================================")
print(f"Training samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")
print(f"Features        : {X_train.shape[1]}")

# Save datasets
X_train.to_csv("preprocessing/X_train.csv", index=False)
X_test.to_csv("preprocessing/X_test.csv", index=False)

pd.DataFrame({
    "Attack_Type": y_train
}).to_csv("preprocessing/y_train.csv", index=False)

pd.DataFrame({
    "Attack_Type": y_test
}).to_csv("preprocessing/y_test.csv", index=False)

# Save class mapping
pd.DataFrame({
    "Encoded": range(len(label_encoder.classes_)),
    "Attack_Type": label_encoder.classes_
}).to_csv(
    "preprocessing/class_mapping.csv",
    index=False
)

print("\nFiles saved successfully.")