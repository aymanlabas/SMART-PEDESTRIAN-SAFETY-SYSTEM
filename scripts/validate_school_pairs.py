
import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

INPUT = PROJECT / "datasets/scenarios/school/school_risk_dataset.csv"

df = pd.read_csv(INPUT)

# Keep known labels
known = df[df["risk_label"].isin([0, 1])].copy()

# Pair-level analysis
pairs = known.groupby(
    ["vehicle_id", "pedestrian_id"]
).agg(
    total_records=("risk_label", "size"),
    risk_records=("risk_label", "sum"),
    junctions=("nearest_school_junction",
               lambda x: ", ".join(sorted(x.unique())))
).reset_index()

pairs["has_risk"] = pairs["risk_records"] > 0

print("\n========== PAIR ANALYSIS ==========")
print("Known pairs:", len(pairs))
print("Pairs with risk:", pairs["has_risk"].sum())
print("Pairs without risk:", (~pairs["has_risk"]).sum())

print("\n========== RISK PAIRS DETAILS ==========")
print(
    pairs[pairs["has_risk"]]
    .sort_values("risk_records", ascending=False)
    .to_string(index=False)
)

print("\n========== UNIQUE PAIRS BY JUNCTION ==========")

risk_rows = known[known["risk_label"] == 1]

junction_pairs = (
    risk_rows.groupby("nearest_school_junction")
    .apply(
        lambda x: x[["vehicle_id", "pedestrian_id"]]
        .drop_duplicates().shape[0],
        include_groups=False
    )
)

print(junction_pairs.to_string())

print("\n========== RISK RECORDS PER PAIR ==========")
print(pairs[pairs["has_risk"]]["risk_records"].describe())

OUTPUT = PROJECT / "results/scenarios/school/pair_validation.csv"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
pairs.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
