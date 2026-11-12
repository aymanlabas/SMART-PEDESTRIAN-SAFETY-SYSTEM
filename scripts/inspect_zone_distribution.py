
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "lstm"

splits = {
    "Train": "train_sequence_metadata.csv",
    "Validation": "validation_sequence_metadata.csv",
    "Test": "test_sequence_metadata.csv",
}

frames = []

for split_name, filename in splits.items():
    df = pd.read_csv(DATA_DIR / filename)

    df["split"] = split_name
    frames.append(df)

all_data = pd.concat(frames, ignore_index=True)

# Count Risk and No Risk per zone and split
summary = (
    all_data.groupby(["split", "zone_type", "risk_label"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

summary = summary.rename(columns={
    0: "no_risk",
    1: "risk"
})

summary["total"] = summary["no_risk"] + summary["risk"]

summary["risk_percentage"] = (
    summary["risk"] / summary["total"] * 100
).round(3)

print("\n========== ZONE DISTRIBUTION BY SPLIT ==========")
print(summary.to_string(index=False))

# Overall risk counts by zone
overall = (
    all_data.groupby("zone_type")
    .agg(
        total_sequences=("risk_label", "size"),
        risk_sequences=("risk_label", "sum"),
    )
    .reset_index()
)

overall["no_risk_sequences"] = (
    overall["total_sequences"] - overall["risk_sequences"]
)

overall["risk_percentage"] = (
    overall["risk_sequences"]
    / overall["total_sequences"] * 100
).round(3)

print("\n========== OVERALL DISTRIBUTION ==========")
print(overall.to_string(index=False))

output = ROOT / "results" / "zone_distribution_by_split.csv"
summary.to_csv(output, index=False)

print(f"\nSaved: {output}")
