
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold

# ============================================================
# CONFIGURATION - MUST MATCH prepare_lstm_dataset.py
# ============================================================

DATA_PATH = "datasets/pedestrian_vehicle_risk_dataset_v3.csv"
OUTPUT_DIR = "datasets/lstm"

SEQ_LEN = 20
STRIDE = 5
MAX_GAP = 0.2

RANDOM_STATE_OUTER = 42
RANDOM_STATE_INNER = 123

GROUP_COLUMNS = ["vehicle_id", "pedestrian_id"]
TARGET = "risk_label"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/4] Loading original dataset...")

df = pd.read_csv(DATA_PATH)

df = df[df[TARGET].isin([0, 1])].copy()
df.reset_index(drop=True, inplace=True)
df[TARGET] = df[TARGET].astype(np.int8)

y = df[TARGET].to_numpy()

groups = pd.MultiIndex.from_frame(
    df[GROUP_COLUMNS]
).factorize()[0]

print("Known records:", len(df))


# ============================================================
# 2. REPRODUCE EXACT SAME SPLIT
# ============================================================

print("\n[2/4] Reproducing original data split...")

outer_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE_OUTER
)

train_val_idx, test_idx = next(
    outer_splitter.split(df, y, groups)
)

inner_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE_INNER
)

inner_train_idx, val_relative_idx = next(
    inner_splitter.split(
        df.iloc[train_val_idx],
        y[train_val_idx],
        groups[train_val_idx]
    )
)

train_idx = train_val_idx[inner_train_idx]
val_idx = train_val_idx[val_relative_idx]

print("Train records:", len(train_idx))
print("Validation records:", len(val_idx))
print("Test records:", len(test_idx))


# ============================================================
# 3. RECREATE SEQUENCE METADATA
# ============================================================

def create_metadata(indices, split_name):

    subset = df.iloc[indices].copy()
    subset["_original_index"] = indices

    records = []

    for (vehicle_id, pedestrian_id), group in subset.groupby(
        GROUP_COLUMNS, sort=False
    ):

        group = group.sort_values("time")

        original_indices = group["_original_index"].to_numpy()
        times = group["time"].to_numpy(dtype=np.float64)

        gaps = np.diff(times)

        break_positions = np.where(
            gaps > MAX_GAP + 1e-9
        )[0] + 1

        segments = np.split(
            np.arange(len(group)),
            break_positions
        )

        for segment in segments:

            if len(segment) < SEQ_LEN:
                continue

            for start in range(
                0,
                len(segment) - SEQ_LEN + 1,
                STRIDE
            ):

                positions = segment[start:start + SEQ_LEN]
                row_indices = original_indices[positions]

                window = df.iloc[row_indices]
                last = window.iloc[-1]

                record = {
                    "sequence_index": len(records),
                    "vehicle_id": vehicle_id,
                    "pedestrian_id": pedestrian_id,

                    "start_time": float(window.iloc[0]["time"]),
                    "end_time": float(last["time"]),

                    "distance": float(last["distance"]),
                    "vehicle_speed": float(last["vehicle_speed"]),
                    "pedestrian_speed": float(last["pedestrian_speed"]),
                    "relative_speed": float(last["relative_speed"]),
                    "closing_speed": float(last["closing_speed"]),

                    "intersection_distance": float(last["intersection_distance"]),
                    "zone_type": last["zone_type"],
                    "nearest_intersection_id": last["nearest_intersection_id"],

                    "min_distance_in_sequence": float(window["distance"].min()),
                    "max_closing_speed": float(window["closing_speed"].max()),

                    "is_school_zone": int(last["is_school_zone"]),
                    "is_hospital_zone": int(last["is_hospital_zone"]),
                    "is_commercial_zone": int(last["is_commercial_zone"]),
                    "is_residential_zone": int(last["is_residential_zone"]),

                    "risk_label": int(last[TARGET])
                }

                records.append(record)

    result = pd.DataFrame(records)

    # Verify exact alignment with the existing NPZ file
    npz_path = os.path.join(OUTPUT_DIR, f"{split_name}.npz")

    existing = np.load(npz_path)

    existing_y = existing["y"]

    if len(result) != len(existing_y):
        raise ValueError(
            f"{split_name}: Sequence count mismatch! "
            f"Metadata={len(result)}, NPZ={len(existing_y)}"
        )

    metadata_y = result["risk_label"].to_numpy(dtype=np.int64)

    if not np.array_equal(metadata_y, existing_y):
        raise ValueError(
            f"{split_name}: Label/order mismatch! "
            "Metadata does not match existing NPZ sequences."
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_sequence_metadata.csv"
    )

    result.to_csv(output_path, index=False)

    print(f"\n{split_name.upper()}:")
    print("Sequences:", len(result))
    print("Risk:", int((metadata_y == 1).sum()))
    print("No Risk:", int((metadata_y == 0).sum()))
    print("Metadata saved:", output_path)

    return result


# ============================================================
# 4. EXPORT ALL SPLITS
# ============================================================

print("\n[3/4] Exporting metadata...")

train_meta = create_metadata(train_idx, "train")
val_meta = create_metadata(val_idx, "validation")
test_meta = create_metadata(test_idx, "test")

print("\n[4/4] Verification completed.")

print("\nAll metadata files created successfully.")
