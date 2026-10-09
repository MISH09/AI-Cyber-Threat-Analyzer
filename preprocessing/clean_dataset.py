import pandas as pd
import numpy as np

INPUT_FILE = "preprocessing/ml_dataset.csv"
OUTPUT_FILE = "preprocessing/clean_dataset.csv"

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original shape: {df.shape}")

# --------------------------------------------------
# 1. Remove unnecessary original label
# --------------------------------------------------

if "Label" in df.columns:
    df = df.drop(columns=["Label"])

# --------------------------------------------------
# 2. Replace infinite values with NaN
# --------------------------------------------------

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# --------------------------------------------------
# 3. Remove rows containing missing values
# --------------------------------------------------

before = len(df)

df.dropna(inplace=True)

after = len(df)

print(f"Removed {before - after} rows containing NaN/Infinity")

# --------------------------------------------------
# 4. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["Attack_Type"])
y = df["Attack_Type"]

# --------------------------------------------------
# 5. Make sure all features are numeric
# --------------------------------------------------

X = X.apply(pd.to_numeric, errors="coerce")

# Remove anything that became NaN
valid_rows = X.notna().all(axis=1)

X = X[valid_rows]
y = y[valid_rows]

# --------------------------------------------------
# 6. Combine cleaned data
# --------------------------------------------------

clean_df = X.copy()
clean_df["Attack_Type"] = y.values

# --------------------------------------------------
# 7. Shuffle dataset
# --------------------------------------------------

clean_df = clean_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# --------------------------------------------------
# 8. Save
# --------------------------------------------------

clean_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n======================================")
print("CLEANING COMPLETED")
print("======================================")

print(f"Final shape: {clean_df.shape}")

print("\nClass distribution:")
print(clean_df["Attack_Type"].value_counts())

print(f"\nSaved to: {OUTPUT_FILE}")
