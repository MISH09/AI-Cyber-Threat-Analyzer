import pandas as pd
import glob
import os

DATASET_PATH = "MachineLearningCVE"

files = glob.glob(os.path.join(DATASET_PATH, "*.csv"))

print(f"\nFound {len(files)} CSV files.\n")

for file in files:
    print("=" * 70)
    print(os.path.basename(file))

    # Read only the Label column in chunks
    label_counts = {}

    for chunk in pd.read_csv(
        file,
        usecols=[" Label"],
        chunksize=100_000
    ):
        counts = chunk[" Label"].value_counts()

        for label, count in counts.items():
            label_counts[label] = label_counts.get(label, 0) + count

    print("\nLabels:")
    for label, count in label_counts.items():
        print(f"  {label}: {count:,}")