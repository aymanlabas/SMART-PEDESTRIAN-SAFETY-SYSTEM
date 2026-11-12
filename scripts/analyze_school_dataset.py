
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

INPUT = PROJECT / "datasets" / "scenarios" / "school" / "interactions.csv"

SCHOOL_JUNCTIONS = {
    "A2": (0, 400),
    "B2": (200, 400),
    "A3": (0, 600),
    "B3": (200, 600),
}

def main():

    if not INPUT.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT}")

    df = pd.read_csv(INPUT)

    print("\n========== SCHOOL DATASET ==========")
    print("Total records:", len(df))
    print("Vehicles:", df["vehicle_id"].nunique())
    print("Pedestrians:", df["pedestrian_id"].nunique())
    print(
        "Unique pairs:",
        df.groupby(["vehicle_id", "pedestrian_id"]).ngroups
    )

    # Calculate pair midpoint
    mid_x = (df["vehicle_x"] + df["pedestrian_x"]) / 2
    mid_y = (df["vehicle_y"] + df["pedestrian_y"]) / 2

    # Distance to nearest school junction
    distances = {}

    for junction, (jx, jy) in SCHOOL_JUNCTIONS.items():

        distances[junction] = np.sqrt(
            (mid_x - jx) ** 2 +
            (mid_y - jy) ** 2
        )

    distance_df = pd.DataFrame(distances)

    df["nearest_school_junction"] = distance_df.idxmin(axis=1)
    df["school_junction_distance"] = distance_df.min(axis=1)

    print("\n========== SCHOOL PROXIMITY ==========")

    for radius in [30, 50, 75, 100]:

        selected = df[df["school_junction_distance"] <= radius]

        pairs = selected.groupby(
            ["vehicle_id", "pedestrian_id"]
        ).ngroups

        percentage = (
            len(selected) / len(df) * 100
            if len(df) else 0
        )

        print(f"\nRadius: {radius} meters")
        print("Records:", len(selected))
        print("Unique pairs:", pairs)
        print(f"Percentage: {percentage:.2f}%")

    print("\n========== RECORDS BY JUNCTION ==========")

    print(
        df["nearest_school_junction"]
        .value_counts()
        .to_string()
    )

    # Export analyzed dataset
    output = PROJECT / "datasets" / "scenarios" / "school" / "school_analysis.csv"

    df.to_csv(output, index=False)

    print("\nAnalysis saved to:", output)


if __name__ == "__main__":
    main()
