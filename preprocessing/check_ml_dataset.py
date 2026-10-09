import pandas as pd
import numpy as np

FILE = "preprocessing/ml_dataset.csv"

df = pd.read_csv(FILE)

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

print("\nColumns:")
for col in df.columns:
    print(col)

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print(df["Attack_Type"].value_counts())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()
print(missing[missing > 0])

print("\n" + "=" * 60)
print("INFINITE VALUES")
print("=" * 60)

numeric_df = df.select_dtypes(include=np.number)

print(np.isinf(numeric_df).sum().sum())

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes.value_counts())