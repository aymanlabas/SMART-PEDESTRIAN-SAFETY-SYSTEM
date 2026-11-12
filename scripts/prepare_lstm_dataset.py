
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "datasets/pedestrian_vehicle_risk_dataset_v3.csv"
OUTPUT_DIR = "datasets/lstm"

SEQ_LEN = 20
STRIDE = 5
MAX_GAP = 0.2

RANDOM_STATE_OUTER = 42
RANDOM_STATE_INNER = 123

NUMERIC_FEATURES = [
    "distance",
    "vehicle_speed",
    "pedestrian_speed",
    "relative_speed",
    "closing_speed",
    "intersection_distance",
    "vehicle_density_30m",
    "pedestrian_density_30m",
    "avg_vehicle_speed_30m",
    "is_school_zone",
    "is_hospital_zone",
    "is_commercial_zone",
    "is_residential_zone",
    "is_intersection",
]

CATEGORICAL_FEATURES = [
    "zone_type",
    "nearest_intersection_id",
]

TARGET = "risk_label"
GROUP_COLUMNS = ["vehicle_id", "pedestrian_id"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/6] Loading dataset...")

df = pd.read_csv(DATA_PATH)

required_columns = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + GROUP_COLUMNS
    + ["time", TARGET]
)

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Keep only known labels
df = df[df[TARGET].isin([0, 1])].copy()
df.reset_index(drop=True, inplace=True)

df[TARGET] = df[TARGET].astype(np.int8)

print("Known records:", len(df))
print("Risk cases:", int((df[TARGET] == 1).sum()))
print("No Risk cases:", int((df[TARGET] == 0).sum()))


# ============================================================
# 2. GROUP-BASED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n[2/6] Splitting data by vehicle-pedestrian pairs...")

y = df[TARGET].to_numpy()

groups = pd.MultiIndex.from_frame(
    df[GROUP_COLUMNS]
).factorize()[0]

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

train_pairs = set(groups[train_idx])
val_pairs = set(groups[val_idx])
test_pairs = set(groups[test_idx])

assert train_pairs.isdisjoint(val_pairs)
assert train_pairs.isdisjoint(test_pairs)
assert val_pairs.isdisjoint(test_pairs)

print("Training records:", len(train_idx))
print("Validation records:", len(val_idx))
print("Testing records:", len(test_idx))

print("Training risk cases:", int(y[train_idx].sum()))
print("Validation risk cases:", int(y[val_idx].sum()))
print("Testing risk cases:", int(y[test_idx].sum()))

print("Pair overlap: 0")


# ============================================================
# 3. PREPROCESSING
# Fit ONLY on training data
# ============================================================

print("\n[3/6] Preprocessing features...")

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ],
    remainder="drop",
    sparse_threshold=0
)

X_train_raw = df.iloc[train_idx][
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
]

preprocessor.fit(X_train_raw)

# Transform all rows using training-fitted preprocessing
X_all = preprocessor.transform(
    df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
).astype(np.float32)

print("Feature matrix shape:", X_all.shape)

joblib.dump(
    preprocessor,
    os.path.join(OUTPUT_DIR, "lstm_preprocessor.joblib")
)


# ============================================================
# 4. CREATE SEQUENCES
# ============================================================

print("\n[4/6] Creating temporal sequences...")


def create_sequences(indices, split_name):

    subset = df.iloc[indices].copy()
    subset["_original_index"] = indices

    sequences = []
    labels = []

    for _, group in subset.groupby(GROUP_COLUMNS, sort=False):

        group = group.sort_values("time")

        original_indices = group["_original_index"].to_numpy()
        times = group["time"].to_numpy(dtype=np.float64)

        # Divide into continuous segments
        gaps = np.diff(times)
        break_positions = np.where(gaps > MAX_GAP + 1e-9)[0] + 1

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

                window_positions = segment[
                    start:start + SEQ_LEN
                ]

                row_indices = original_indices[window_positions]

                sequence = X_all[row_indices]

                target_index = row_indices[-1]
                label = int(df.iloc[target_index][TARGET])

                sequences.append(sequence)
                labels.append(label)

    if not sequences:
        raise ValueError(
            f"No sequences created for {split_name}. "
            "Check sequence length and observation gaps."
        )

    X_seq = np.stack(sequences).astype(np.float32)
    y_seq = np.asarray(labels, dtype=np.int64)

    print(f"\n{split_name}:")
    print("Sequences:", len(X_seq))
    print("Shape:", X_seq.shape)
    print("Risk:", int((y_seq == 1).sum()))
    print("No Risk:", int((y_seq == 0).sum()))

    return X_seq, y_seq


X_train, y_train = create_sequences(train_idx, "TRAIN")
X_val, y_val = create_sequences(val_idx, "VALIDATION")
X_test, y_test = create_sequences(test_idx, "TEST")


# ============================================================
# 5. SAVE ARRAYS
# ============================================================

print("\n[5/6] Saving prepared sequences...")

np.savez(
    os.path.join(OUTPUT_DIR, "train.npz"),
    X=X_train,
    y=y_train
)

np.savez(
    os.path.join(OUTPUT_DIR, "validation.npz"),
    X=X_val,
    y=y_val
)

np.savez(
    os.path.join(OUTPUT_DIR, "test.npz"),
    X=X_test,
    y=y_test
)


# ============================================================
# 6. SAVE METADATA
# ============================================================

metadata = {
    "sequence_length": SEQ_LEN,
    "stride": STRIDE,
    "max_gap_seconds": MAX_GAP,
    "number_of_features": int(X_train.shape[2]),
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "target": TARGET,
    "train_sequences": len(X_train),
    "validation_sequences": len(X_val),
    "test_sequences": len(X_test),
    "train_positive": int(y_train.sum()),
    "validation_positive": int(y_val.sum()),
    "test_positive": int(y_test.sum())
}

with open(
    os.path.join(OUTPUT_DIR, "metadata.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)

print("\n[6/6] Preparation completed successfully!")
print("Output folder:", OUTPUT_DIR)
