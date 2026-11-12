
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "lstm"

files = {
    "Train": "train_sequence_metadata.csv",
    "Validation": "validation_sequence_metadata.csv",
    "Test": "test_sequence_metadata.csv",
}

frames = []

for split, filename in files.items():
    df = pd.read_csv(DATA_DIR / filename)
    df["split"] = split
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

GROUP = ["vehicle_id", "pedestrian_id"]

# ------------------------------------------------
# 1. Number of unique pairs per split and zone
# ------------------------------------------------

pair_zone = (
    data.groupby(["split", "zone_type"] + GROUP)
    .agg(
        sequences=("risk_label", "size"),
        risk_sequences=("risk_label", "sum"),
    )
    .reset_index()
)

pair_zone["has_risk"] = (
    pair_zone["risk_sequences"] > 0
)

summary = (
    pair_zone.groupby(["split", "zone_type"])
    .agg(
        unique_pairs=("vehicle_id", "size"),
        pairs_with_risk=("has_risk", "sum"),
        sequences=("sequences", "sum"),
        risk_sequences=("risk_sequences", "sum"),
    )
    .reset_index()
)

print("\n========== UNIQUE PAIRS BY ZONE AND SPLIT ==========")
print(summary.to_string(index=False))

# ------------------------------------------------
# 2. Check if a pair appears in multiple zones
# ------------------------------------------------

pair_zones = (
    data.groupby(GROUP)["zone_type"]
    .nunique()
)

print("\n========== PAIR / ZONE ANALYSIS ==========")
print(f"Total unique pairs: {len(pair_zones)}")
print(f"Pairs appearing in multiple zones: {(pair_zones > 1).sum()}")
print(f"Pairs appearing in one zone only: {(pair_zones == 1).sum()}")

# ------------------------------------------------
# 3. Check pair overlap between splits
# ------------------------------------------------

split_pairs = {}

for split in files:
    subset = data[data["split"] == split]

    split_pairs[split] = set(
        zip(subset["vehicle_id"], subset["pedestrian_id"])
    )

print("\n========== PAIR OVERLAP CHECK ==========")

for first, second in [
    ("Train", "Validation"),
    ("Train", "Test"),
    ("Validation", "Test"),
]:
    overlap = split_pairs[first] & split_pairs[second]
    print(f"{first} / {second}: {len(overlap)} overlapping pairs")

# ------------------------------------------------
# 4. Save diagnostic report
# ------------------------------------------------

output = ROOT / "results" / "zone_group_diagnostic.csv"
summary.to_csv(output, index=False)

print(f"\nSaved: {output}")
print("\nDiagnostic completed.")
