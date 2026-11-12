
import os
import copy
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATA_PATH = "datasets/pedestrian_vehicle_risk_dataset_v3.csv"
SEQUENCE_DIR = "datasets/lstm"

LSTM_MODEL_PATH = "models/lstm_model.pt"

RESULTS_DIR = "results"
MODEL_DIR = "models"

SEQ_LEN = 20
STRIDE = 5
MAX_GAP = 0.2

MIN_RECALL = 0.65
BATCH_SIZE = 256

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

GROUP_COLUMNS = ["vehicle_id", "pedestrian_id"]
TARGET = "risk_label"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
torch.set_num_threads(min(4, os.cpu_count() or 1))


# ============================================================
# LSTM ARCHITECTURE
# Must match the trained model
# ============================================================

class PedestrianRiskLSTM(nn.Module):

    def __init__(self, input_size, hidden_size=64):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)

        return self.fc(last_output).squeeze(1)


# ============================================================
# METRICS
# ============================================================

def evaluate(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities
            )
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "confusion_matrix": cm.tolist()
    }


def select_threshold(y_true, probabilities):

    results = []

    for threshold in np.arange(0.05, 0.951, 0.01):

        results.append(
            evaluate(y_true, probabilities, threshold)
        )

    eligible = [
        item for item in results
        if item["recall"] >= MIN_RECALL
    ]

    if eligible:

        selected = max(
            eligible,
            key=lambda item: (
                item["precision"],
                item["f1"]
            )
        )

        rule = "Highest precision with validation recall >= 0.65"

    else:

        selected = max(
            results,
            key=lambda item: item["f1"]
        )

        rule = "Fallback: highest validation F1"

    return selected, results, rule


# ============================================================
# 1. LOAD ORIGINAL DATA AND SEQUENCE FILES
# ============================================================

print("\n[1/7] Loading datasets...")

df = pd.read_csv(DATA_PATH)

df = df[df[TARGET].isin([0, 1])].copy()
df = df.reset_index(drop=True)

df[TARGET] = df[TARGET].astype(np.int8)

train_npz = np.load(
    os.path.join(SEQUENCE_DIR, "train.npz")
)

val_npz = np.load(
    os.path.join(SEQUENCE_DIR, "validation.npz")
)

test_npz = np.load(
    os.path.join(SEQUENCE_DIR, "test.npz")
)

X_train_lstm = train_npz["X"].astype(np.float32)
y_train_lstm = train_npz["y"].astype(np.int64)

X_val_lstm = val_npz["X"].astype(np.float32)
y_val_lstm = val_npz["y"].astype(np.int64)

X_test_lstm = test_npz["X"].astype(np.float32)
y_test_lstm = test_npz["y"].astype(np.int64)

print("Train sequences:", len(y_train_lstm))
print("Validation sequences:", len(y_val_lstm))
print("Test sequences:", len(y_test_lstm))


# ============================================================
# 2. RECREATE EXACT GROUP SPLIT
# ============================================================

print("\n[2/7] Recreating group-based split...")

y = df[TARGET].to_numpy()

groups = pd.MultiIndex.from_frame(
    df[GROUP_COLUMNS]
).factorize()[0]

outer_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_val_idx, test_idx = next(
    outer_splitter.split(df, y, groups)
)

inner_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
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

print("Train rows:", len(train_idx))
print("Validation rows:", len(val_idx))
print("Test rows:", len(test_idx))


# ============================================================
# 3. RECREATE SEQUENCE ENDPOINT INDICES
# ============================================================

print("\n[3/7] Recreating sequence endpoints...")


def get_sequence_endpoints(indices):

    subset = df.iloc[indices].copy()
    subset["_original_index"] = indices

    endpoints = []

    for _, group in subset.groupby(
        GROUP_COLUMNS,
        sort=False
    ):

        group = group.sort_values("time")

        original_indices = group[
            "_original_index"
        ].to_numpy()

        times = group["time"].to_numpy(
            dtype=np.float64
        )

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

                window_positions = segment[
                    start:start + SEQ_LEN
                ]

                endpoint_index = original_indices[
                    window_positions[-1]
                ]

                endpoints.append(endpoint_index)

    return np.asarray(endpoints, dtype=np.int64)


train_endpoints = get_sequence_endpoints(train_idx)
val_endpoints = get_sequence_endpoints(val_idx)
test_endpoints = get_sequence_endpoints(test_idx)

y_train_endpoints = df.iloc[train_endpoints][
    TARGET
].to_numpy(dtype=np.int64)

y_val_endpoints = df.iloc[val_endpoints][
    TARGET
].to_numpy(dtype=np.int64)

y_test_endpoints = df.iloc[test_endpoints][
    TARGET
].to_numpy(dtype=np.int64)


# Verify sequence order and labels
assert len(train_endpoints) == len(y_train_lstm)
assert len(val_endpoints) == len(y_val_lstm)
assert len(test_endpoints) == len(y_test_lstm)

assert np.array_equal(y_train_endpoints, y_train_lstm)
assert np.array_equal(y_val_endpoints, y_val_lstm)
assert np.array_equal(y_test_endpoints, y_test_lstm)

print("Sequence endpoint verification: PASSED")
print("Same test sequences and labels: VERIFIED")


# ============================================================
# 4. TRAIN RANDOM FOREST ON SAME TRAIN ENDPOINTS
# ============================================================

print("\n[4/7] Training Random Forest on sequence endpoints...")

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

rf_preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
])

rf_model = Pipeline([
    ("preprocessor", rf_preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        verbose=1
    ))
])

feature_columns = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)

X_train_rf = df.iloc[train_endpoints][
    feature_columns
]

X_val_rf = df.iloc[val_endpoints][
    feature_columns
]

X_test_rf = df.iloc[test_endpoints][
    feature_columns
]

rf_model.fit(
    X_train_rf,
    y_train_endpoints
)

print("Random Forest training completed.")


# ============================================================
# 5. RANDOM FOREST VALIDATION AND THRESHOLD
# ============================================================

print("\n[5/7] Selecting Random Forest threshold...")

rf_val_probabilities = rf_model.predict_proba(
    X_val_rf
)[:, 1]

rf_val_selected, rf_threshold_results, rf_rule = (
    select_threshold(
        y_val_endpoints,
        rf_val_probabilities
    )
)

rf_threshold = rf_val_selected["threshold"]

print("RF threshold:", rf_threshold)
print("RF validation precision:", rf_val_selected["precision"])
print("RF validation recall:", rf_val_selected["recall"])
print("RF validation F1:", rf_val_selected["f1"])


# ============================================================
# 6. LOAD LSTM AND PREDICT
# ============================================================

print("\n[6/7] Loading LSTM model...")

checkpoint = torch.load(
    LSTM_MODEL_PATH,
    map_location=DEVICE
)

lstm_model = PedestrianRiskLSTM(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"]
).to(DEVICE)

lstm_model.load_state_dict(
    checkpoint["model_state_dict"]
)

lstm_model.eval()


def lstm_predict(X):

    dataset = TensorDataset(
        torch.from_numpy(X)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    probabilities = []

    with torch.no_grad():

        for (batch,) in loader:

            logits = lstm_model(
                batch.to(DEVICE)
            )

            probs = torch.sigmoid(logits)

            probabilities.extend(
                probs.cpu().numpy()
            )

    return np.asarray(probabilities)


lstm_val_probabilities = lstm_predict(X_val_lstm)
lstm_test_probabilities = lstm_predict(X_test_lstm)

lstm_val_selected, lstm_threshold_results, lstm_rule = (
    select_threshold(
        y_val_lstm,
        lstm_val_probabilities
    )
)

lstm_threshold = lstm_val_selected["threshold"]

print("LSTM threshold:", lstm_threshold)
print("LSTM validation precision:", lstm_val_selected["precision"])
print("LSTM validation recall:", lstm_val_selected["recall"])
print("LSTM validation F1:", lstm_val_selected["f1"])


# ============================================================
# 7. SAME TEST SET COMPARISON
# ============================================================

print("\n[7/7] Evaluating both models on identical test sequences...")

rf_test_probabilities = rf_model.predict_proba(
    X_test_rf
)[:, 1]

rf_test_metrics = evaluate(
    y_test_endpoints,
    rf_test_probabilities,
    rf_threshold
)

lstm_test_metrics = evaluate(
    y_test_lstm,
    lstm_test_probabilities,
    lstm_threshold
)

comparison = pd.DataFrame([
    {
        "Model": "Random Forest",
        **rf_test_metrics
    },
    {
        "Model": "LSTM",
        **lstm_test_metrics
    }
])

print("\n========== FAIR MODEL COMPARISON ==========")

print(
    comparison[
        [
            "Model",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "roc_auc",
            "TP",
            "FP",
            "FN",
            "TN"
        ]
    ].to_string(index=False)
)

comparison.to_csv(
    os.path.join(
        RESULTS_DIR,
        "fair_rf_lstm_comparison.csv"
    ),
    index=False
)

joblib.dump(
    rf_model,
    os.path.join(
        MODEL_DIR,
        "random_forest_fair_model.joblib"
    )
)

with open(
    os.path.join(
        RESULTS_DIR,
        "fair_rf_lstm_comparison.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "test_sequence_count": len(y_test_lstm),
            "test_positive_count": int(y_test_lstm.sum()),
            "test_negative_count": int(
                (y_test_lstm == 0).sum()
            ),
            "random_forest": {
                "threshold_rule": rf_rule,
                "validation": rf_val_selected,
                "test": rf_test_metrics
            },
            "lstm": {
                "threshold_rule": lstm_rule,
                "validation": lstm_val_selected,
                "test": lstm_test_metrics
            }
        },
        file,
        indent=4
    )

print("\nFair comparison completed!")
print("CSV: results/fair_rf_lstm_comparison.csv")
print("JSON: results/fair_rf_lstm_comparison.json")
print("RF model: models/random_forest_fair_model.joblib")
