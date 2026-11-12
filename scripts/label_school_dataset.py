
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

PROJECT = Path(__file__).resolve().parent.parent

INPUT = PROJECT / "datasets" / "scenarios" / "school" / "interactions.csv"

OUTPUT = PROJECT / "datasets" / "scenarios" / "school" / "school_risk_dataset.csv"

SCHOOL_JUNCTIONS = {
    "A2": (0, 400),
    "B2": (200, 400),
    "A3": (0, 600),
    "B3": (200, 600),
}

SCHOOL_RADIUS = 50.0
RISK_DISTANCE = 2.5
HORIZON = 3.0
MAX_GAP = 0.2

# ==========================================
# SCHOOL DISTANCE
# ==========================================

def add_school_distances(df):
    midpoint_x = (df["vehicle_x"] + df["pedestrian_x"]) / 2
    midpoint_y = (df["vehicle_y"] + df["pedestrian_y"]) / 2

    distances = {}

    for junction, (jx, jy) in SCHOOL_JUNCTIONS.items():
        distances[junction] = np.sqrt(
            (midpoint_x - jx) ** 2 +
            (midpoint_y - jy) ** 2
        )

    distance_df = pd.DataFrame(distances, index=df.index)

    df["nearest_school_junction"] = distance_df.idxmin(axis=1)
    df["school_junction_distance"] = distance_df.min(axis=1)

    return df


# ==========================================
# SELECT SCHOOL PAIRS
# ==========================================

def select_school_pairs(df):
    school_records = df[
        df["school_junction_distance"] <= SCHOOL_RADIUS
    ]

    selected_pairs = school_records[
        ["vehicle_id", "pedestrian_id"]
    ].drop_duplicates()

    print("\n========== SCHOOL SELECTION ==========")
    print("Records near School (50m):", len(school_records))
    print("Selected unique pairs:", len(selected_pairs))

    selected = df.merge(
        selected_pairs,
        on=["vehicle_id", "pedestrian_id"],
        how="inner"
    )

    return selected


# ==========================================
# RISK LABELING
# ==========================================

def label_pair(group):
    group = group.sort_values("time").copy()

    times = group["time"].to_numpy(dtype=float)
    distances = group["distance"].to_numpy(dtype=float)

    labels = np.full(len(group), -1, dtype=np.int8)

    for i in range(len(group)):
        target_time = times[i] + HORIZON

        # All observed records from current time to horizon
        end = np.searchsorted(times, target_time, side="right")

        future_distances = distances[i:end]

        # A near-conflict observed within the horizon
        if np.any(future_distances <= RISK_DISTANCE):
            labels[i] = 1
            continue

        # Check whether the full horizon is observable
        endpoint = np.searchsorted(times, target_time, side="left")

        if endpoint >= len(times):
            continue

        # The last observation must reach the horizon
        if times[endpoint] - target_time > MAX_GAP:
            continue

        observed_times = times[i:endpoint + 1]
        gaps = np.diff(observed_times)

        # Missing temporal observations => Unknown
        if len(gaps) > 0 and np.any(gaps > MAX_GAP + 1e-9):
            continue

        labels[i] = 0

    group["risk_label"] = labels

    return group


# ==========================================
# MAIN
# ==========================================

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    print("Loading School interactions...")

    df = pd.read_csv(INPUT)

    required = [
        "time", "vehicle_id", "pedestrian_id",
        "vehicle_x", "vehicle_y",
        "pedestrian_x", "pedestrian_y",
        "distance"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required).copy()
    df = df.sort_values(
        ["vehicle_id", "pedestrian_id", "time"]
    ).reset_index(drop=True)

    print("Original records:", len(df))

    # Calculate proximity to School junctions
    df = add_school_distances(df)

    # Keep complete timelines for pairs that entered School vicinity
    df = select_school_pairs(df)

    if df.empty:
        print("No School pairs found.")
        return

    # Label each pair independently
    print("\nGenerating risk labels...")

    groups = []

    for _, group in df.groupby(
        ["vehicle_id", "pedestrian_id"],
        sort=False
    ):
        groups.append(label_pair(group))

    result = pd.concat(groups, ignore_index=True)

    # ======================================
    # SAVE
    # ======================================

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    # ======================================
    # REPORT
    # ======================================

    counts = result["risk_label"].value_counts().sort_index()

    total = len(result)
    risk = int(counts.get(1, 0))
    no_risk = int(counts.get(0, 0))
    unknown = int(counts.get(-1, 0))
    known = risk + no_risk

    print("\n========== SCHOOL RISK DATASET ==========")
    print("Total records:", total)
    print("Unique vehicles:", result["vehicle_id"].nunique())
    print("Unique pedestrians:", result["pedestrian_id"].nunique())
    print(
        "Unique pairs:",
        result[["vehicle_id", "pedestrian_id"]].drop_duplicates().shape[0]
    )

    print("\n========== LABEL DISTRIBUTION ==========")
    print(f"Risk (1):     {risk:,}")
    print(f"No Risk (0):  {no_risk:,}")
    print(f"Unknown (-1): {unknown:,}")

    print(
        f"\nRisk percentage among known labels: "
        f"{risk / known * 100:.3f}%"
        if known else "\nNo known labels."
    )

    print("\n========== RISK BY SCHOOL JUNCTION ==========")

    known_df = result[result["risk_label"].isin([0, 1])]

    print(
        known_df.groupby("nearest_school_junction")["risk_label"]
        .agg(records="count", risk_records="sum")
        .to_string()
    )

    print("\nDataset saved:", OUTPUT)


if __name__ == "__main__":
    main()
