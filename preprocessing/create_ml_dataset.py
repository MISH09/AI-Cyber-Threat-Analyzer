import pandas as pd
import glob
import os

DATASET_PATH = "MachineLearningCVE"
OUTPUT_FILE = "preprocessing/ml_dataset.csv"

MAX_PER_CLASS = 10000

# Map original labels to our 7 main classes
LABEL_MAP = {
    "BENIGN": "BENIGN",

    "DDoS": "DDoS",

    "DoS Hulk": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS GoldenEye": "DoS",

    "PortScan": "PortScan",

    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",

    "Bot": "Bot",

    "Web Attack � Brute Force": "Web Attack",
    "Web Attack � XSS": "Web Attack",
    "Web Attack � Sql Injection": "Web Attack"
}

files = glob.glob(os.path.join(DATASET_PATH, "*.csv"))

class_data = {}

for file in files:

    print(f"\nProcessing: {os.path.basename(file)}")

    for chunk in pd.read_csv(file, chunksize=100000):

        # Remove leading/trailing spaces from column names
        chunk.columns = chunk.columns.str.strip()

        if "Label" not in chunk.columns:
            print("Label column not found!")
            continue

        # Clean labels
        chunk["Label"] = chunk["Label"].astype(str).str.strip()

        # Map original labels to our classes
        chunk["Attack_Type"] = chunk["Label"].map(LABEL_MAP)

        # Remove labels we don't want
        chunk = chunk.dropna(subset=["Attack_Type"])

        for attack_type in chunk["Attack_Type"].unique():

            if attack_type not in class_data:
                class_data[attack_type] = []

            remaining = MAX_PER_CLASS - sum(
                len(x) for x in class_data[attack_type]
            )

            if remaining <= 0:
                continue

            samples = chunk[
                chunk["Attack_Type"] == attack_type
            ].head(remaining)

            if len(samples) > 0:
                class_data[attack_type].append(samples)

    print("Current class counts:")

    for label, parts in class_data.items():
        print(label, sum(len(x) for x in parts))


# Combine everything
final_parts = []

for label, parts in class_data.items():
    if parts:
        final_parts.append(pd.concat(parts, ignore_index=True))

final_df = pd.concat(final_parts, ignore_index=True)

# Shuffle
final_df = final_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n======================================")
print("Dataset creation completed!")
print("======================================")
print(f"Total rows: {len(final_df):,}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nFinal class distribution:")
print(final_df["Attack_Type"].value_counts())