
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT / "datasets" /
    "pedestrian_vehicle_interactions_zones.csv"
)

OUTPUT_FILE = (
    PROJECT / "datasets" /
    "pedestrian_vehicle_risk_dataset_v2.csv"
)

RISK_DISTANCE = 2.5
PREDICTION_HORIZON = 3.0

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

df = df.dropna(
    subset=["time", "distance", "vehicle_id", "pedestrian_id"]
).copy()

df = df.sort_values(
    ["vehicle_id", "pedestrian_id", "time"],
    kind="mergesort"
).reset_index(drop=True)

print(f"Total records: {len(df):,}")

# NaN = insufficient future observations
risk_labels = np.full(len(df), np.nan)

groups = df.groupby(
    ["vehicle_id", "pedestrian_id"],
    sort=False
)

print(f"Unique pairs: {groups.ngroups:,}")
print("Generating labels...")

for number, (_, group) in enumerate(groups, start=1):

    indices = group.index.to_numpy()

    times = group["time"].to_numpy(dtype=np.float64)
    distances = group["distance"].to_numpy(dtype=np.float64)

    # We only label observations with a complete
    # 3-second observation horizon for this pair.
    last_pair_time = times[-1]

    valid = times <= last_pair_time - PREDICTION_HORIZON

    if not np.any(valid):
        continue

    # Find all observed near-conflict moments
    near_times = times[distances <= RISK_DISTANCE]

    # Find the next near-conflict strictly AFTER current time
    next_positions = np.searchsorted(
        near_times,
        times,
        side="right"
    )

    future_risk = np.zeros(len(times), dtype=bool)

    has_event = next_positions < len(near_times)

    future_risk[has_event] = (
        near_times[next_positions[has_event]]
        <= times[has_event] + PREDICTION_HORIZON
    )

    # Assign both labels:
    # 1 = near-conflict within next 3 seconds
    # 0 = no observed near-conflict within next 3 seconds
    risk_labels[indices[valid]] = (
        future_risk[valid].astype(np.int8)
    )

    if number % 500 == 0:
        print(f"Processed {number:,}/{groups.ngroups:,} pairs")

df["risk_label"] = risk_labels

print("\nSaving dataset...")

df.to_csv(OUTPUT_FILE, index=False)

# REPORT
known = df["risk_label"].dropna()

risk_count = int((known == 1).sum())
no_risk_count = int((known == 0).sum())
unknown_count = int(df["risk_label"].isna().sum())

print("\n========== RISK LABEL REPORT V2 ==========")
print(f"Total records: {len(df):,}")
print(f"Known labels: {len(known):,}")
print(f"Risk (1): {risk_count:,}")
print(f"No Risk (0): {no_risk_count:,}")
print(f"Unknown: {unknown_count:,}")

if len(known):
    print(
        f"Risk percentage: "
        f"{risk_count / len(known) * 100:.2f}%"
    )

print("\nRisk distribution by zone:")

print(
    df.dropna(subset=["risk_label"])
      .groupby("zone_type")["risk_label"]
      .agg(
          records="count",
          risk_percentage=lambda x: x.mean() * 100
      )
      .round(2)
      .to_string()
)

print("\nOutput:", OUTPUT_FILE)
print("Done!")
