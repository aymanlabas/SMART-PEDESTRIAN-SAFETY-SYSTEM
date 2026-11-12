
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

FILE = (
    PROJECT / "datasets" /
    "pedestrian_vehicle_risk_dataset_v2.csv"
)

print("Loading dataset...")

df = pd.read_csv(FILE)

df = df.dropna(subset=["risk_label"]).copy()

df["risk_label"] = df["risk_label"].astype(int)

# 1. Class distribution
print("\n===== CLASS DISTRIBUTION =====")

print(df["risk_label"].value_counts())

print("\nPercentages:")
print(
    (df["risk_label"].value_counts(normalize=True) * 100)
    .round(3)
)

# 2. Pair observation gaps
print("\n===== OBSERVATION GAPS =====")

df = df.sort_values(
    ["vehicle_id", "pedestrian_id", "time"]
)

gaps = (
    df.groupby(["vehicle_id", "pedestrian_id"])["time"]
      .diff()
      .dropna()
)

print("Total time gaps:", len(gaps))
print("Average gap:", round(gaps.mean(), 3), "seconds")
print("Median gap:", round(gaps.median(), 3), "seconds")
print("Maximum gap:", round(gaps.max(), 3), "seconds")

print("\nGap distribution:")
print("<= 0.2s:", (gaps <= 0.2).sum())
print("0.2-0.5s:", ((gaps > 0.2) & (gaps <= 0.5)).sum())
print("0.5-1s:", ((gaps > 0.5) & (gaps <= 1)).sum())
print("> 1s:", (gaps > 1).sum())

# 3. Risk examples
print("\n===== RISK EXAMPLES =====")

risk = df[df["risk_label"] == 1]

columns = [
    "time",
    "vehicle_id",
    "pedestrian_id",
    "distance",
    "closing_speed",
    "ttc",
    "zone_type"
]

print(risk[columns].head(15).to_string(index=False))

# 4. Risk by zone
print("\n===== RISK BY ZONE =====")

print(
    df.groupby("zone_type")["risk_label"]
      .agg(
          total="count",
          risk_count="sum",
          risk_percentage=lambda x: x.mean() * 100
      )
      .round(3)
      .to_string()
)

print("\nValidation report completed.")
