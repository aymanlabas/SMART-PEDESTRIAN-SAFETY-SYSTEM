
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

PATH = ROOT / "results" / "test_predictions.csv"

df = pd.read_csv(PATH)

print("\n========== PROBABILITY DISTRIBUTION BY ZONE ==========")

summary = (
    df.groupby(["zone_type", "actual_risk"])["risk_probability"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        minimum="min",
        maximum="max",
    )
    .round(4)
)

print(summary.to_string())

print("\n========== RISK PROBABILITY ABOVE THRESHOLD ==========")

threshold = 0.07262967526912689

df["above_threshold"] = (
    df["risk_probability"] >= threshold
)

result = (
    df.groupby("zone_type")
    .agg(
        total=("actual_risk", "size"),
        actual_risk=("actual_risk", "sum"),
        predicted_risk=("above_threshold", "sum"),
        average_probability=("risk_probability", "mean"),
        max_probability=("risk_probability", "max"),
    )
    .round(4)
)

print(result.to_string())

print("\nAnalysis completed.")
