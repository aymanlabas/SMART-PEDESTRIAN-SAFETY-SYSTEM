
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent

INPUT = PROJECT / "datasets" / "pedestrian_vehicle_interactions_zones.csv"
OUTPUT = PROJECT / "datasets" / "pedestrian_vehicle_risk_dataset_v3.csv"

HORIZON = 3.0
RISK_DISTANCE = 2.5
MAX_GAP = 0.2

print("Loading dataset...")

df = pd.read_csv(INPUT)

df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

df = df.dropna(
    subset=["time", "distance", "vehicle_id", "pedestrian_id"]
).copy()

df = df.sort_values(
    ["vehicle_id", "pedestrian_id", "time"],
    kind="mergesort"
).reset_index(drop=True)

labels = np.full(len(df), np.nan)

groups = df.groupby(
    ["vehicle_id", "pedestrian_id"],
    sort=False
)

print("Total records:", f"{len(df):,}")
print("Unique pairs:", f"{groups.ngroups:,}")

print("\nGenerating continuous-observation labels...")

for number, (_, group) in enumerate(groups, start=1):

    idx = group.index.to_numpy()

    times = group["time"].to_numpy(dtype=np.float64)
    distances = group["distance"].to_numpy(dtype=np.float64)

    n = len(times)

    if n < 2:
        continue

    # First observation at or after t + 3 seconds
    end_idx = np.searchsorted(
        times,
        times + HORIZON,
        side="left"
    )

    # Complete horizon must be available
    complete = end_idx < n

    # Identify gaps greater than 0.2 seconds
    gaps = np.diff(times)
    bad_gap = gaps > MAX_GAP + 1e-9

    # Prefix sum allows fast gap validation
    prefix = np.concatenate([
        [0],
        np.cumsum(bad_gap)
    ])

    positions = np.arange(n)

    # Check all gaps between current time and horizon end
    safe_end = np.minimum(end_idx, n - 1)

    continuous = (
        prefix[safe_end] - prefix[positions] == 0
    )

    valid = complete & continuous

    if not np.any(valid):
        continue

    # Find future near-conflict events
    near_times = times[distances <= RISK_DISTANCE]

    next_pos = np.searchsorted(
        near_times,
        times,
        side="right"
    )

    has_event = next_pos < len(near_times)

    future_event_time = np.full(n, np.inf)

    future_event_time[has_event] = near_times[
        next_pos[has_event]
    ]

    future_risk = (
        future_event_time <= times + HORIZON
    )

    labels[idx[valid]] = future_risk[valid].astype(np.int8)

    if number % 500 == 0:
        print(f"Processed {number:,}/{groups.ngroups:,} pairs")

df["risk_label"] = labels

print("\nSaving V3 dataset...")

df.to_csv(OUTPUT, index=False)

known = df["risk_label"].dropna()

print("\n========== V3 VALIDATION REPORT ==========")
print("Total:", f"{len(df):,}")
print("Known:", f"{len(known):,}")
print("Risk:", int((known == 1).sum()))
print("No Risk:", int((known == 0).sum()))
print("Unknown:", int(df["risk_label"].isna().sum()))

if len(known):
    print(
        "Risk percentage:",
        round((known == 1).mean() * 100, 3),
        "%"
    )

    print("\nRisk by zone:")

    print(
        df.dropna(subset=["risk_label"])
          .groupby("zone_type")["risk_label"]
          .agg(
              records="count",
              risk_count="sum",
              risk_percentage=lambda x: x.mean() * 100
          )
          .round(3)
          .to_string()
    )

print("\nOutput:", OUTPUT)
print("Done!")
