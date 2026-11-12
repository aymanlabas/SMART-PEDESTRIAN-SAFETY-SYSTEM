
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# CONFIGURATION
# =====================================================

DATASET = "datasets/pedestrian_vehicle_risk_dataset_v3.csv"

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
    "is_intersection"
]

CATEGORICAL_FEATURES = [
    "zone_type",
    "nearest_intersection_id"
]

TARGET = "risk_label"
GROUP_COLUMNS = ["vehicle_id", "pedestrian_id"]

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# =====================================================
# 1. LOAD DATA
# =====================================================

print("=" * 55)
print("RANDOM FOREST - THRESHOLD OPTIMIZATION")
print("=" * 55)

print("\nLoading dataset...")

columns = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + GROUP_COLUMNS
    + [TARGET]
)

df = pd.read_csv(DATASET, usecols=columns)
df = df.dropna(subset=[TARGET]).copy()

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET].astype(int)

groups = (
    df["vehicle_id"].astype(str)
    + "__"
    + df["pedestrian_id"].astype(str)
)

print(f"Total records: {len(df):,}")
print("\nClass distribution:")
print(y.value_counts())

# =====================================================
# 2. TRAIN / VALIDATION / TEST SPLIT
# =====================================================

print("\nSplitting data by vehicle-pedestrian pairs...")

outer_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_val_idx, test_idx = next(
    outer_splitter.split(X, y, groups)
)

X_train_val = X.iloc[train_val_idx]
y_train_val = y.iloc[train_val_idx]
groups_train_val = groups.iloc[train_val_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]
groups_test = groups.iloc[test_idx]

inner_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)

train_idx, val_idx = next(
    inner_splitter.split(
        X_train_val,
        y_train_val,
        groups_train_val
    )
)

X_train = X_train_val.iloc[train_idx]
y_train = y_train_val.iloc[train_idx]
groups_train = groups_train_val.iloc[train_idx]

X_val = X_train_val.iloc[val_idx]
y_val = y_train_val.iloc[val_idx]
groups_val = groups_train_val.iloc[val_idx]

assert set(groups_train).isdisjoint(set(groups_val))
assert set(groups_train).isdisjoint(set(groups_test))
assert set(groups_val).isdisjoint(set(groups_test))

print(f"Training records   : {len(X_train):,}")
print(f"Validation records : {len(X_val):,}")
print(f"Testing records    : {len(X_test):,}")

print("\nPositive cases:")
print(f"Train      : {y_train.sum():,}")
print(f"Validation : {y_val.sum():,}")
print(f"Test       : {y_test.sum():,}")

# =====================================================
# 3. PREPROCESSING
# =====================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
])

# =====================================================
# 4. RANDOM FOREST
# =====================================================

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=18,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf)
])

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("\nTraining completed!")

# =====================================================
# 5. OPTIMIZE THRESHOLD ON VALIDATION SET
# =====================================================

print("\nOptimizing threshold using validation data...")

val_probabilities = model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.01)

threshold_results = []

for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val, val_predictions, zero_division=0
    )

    recall = recall_score(
        y_val, val_predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, val_predictions, zero_division=0
    )

    threshold_results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

threshold_df = pd.DataFrame(threshold_results)


# Choose the best precision while maintaining Recall >= 65%

MIN_RECALL = 0.65

eligible = threshold_df[
    threshold_df["recall"] >= MIN_RECALL
]

if not eligible.empty:
    best_row = eligible.sort_values(
        ["precision", "f1_score"],
        ascending=[False, False]
    ).iloc[0]

    print("\nThreshold selected with Recall constraint:")
    print(f"Minimum Recall required: {MIN_RECALL:.2%}")

else:
    print("\nWARNING: No threshold reached the required Recall.")
    print("Selecting the threshold with the highest Recall.")

    best_row = threshold_df.sort_values(
        ["recall", "precision"],
        ascending=[False, False]
    ).iloc[0]

best_threshold = float(best_row["threshold"])


print("\nBest threshold:")
print(f"Threshold : {best_threshold:.2f}")
print(f"Precision : {best_row['precision']:.4f}")
print(f"Recall    : {best_row['recall']:.4f}")
print(f"F1-score  : {best_row['f1_score']:.4f}")

threshold_df.to_csv(
    "results/threshold_comparison.csv",
    index=False
)

# =====================================================
# 6. FINAL EVALUATION ON TEST SET
# =====================================================

print("\nEvaluating on untouched test data...")

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(
    y_test, test_predictions, zero_division=0
)
recall = recall_score(
    y_test, test_predictions, zero_division=0
)
f1 = f1_score(
    y_test, test_predictions, zero_division=0
)
pr_auc = average_precision_score(y_test, test_probabilities)
roc_auc = roc_auc_score(y_test, test_probabilities)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

print("\n" + "=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)

print(f"Threshold : {best_threshold:.2f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["No Risk", "Risk"],
    zero_division=0
))

# =====================================================
# 7. SAVE RESULTS AND MODEL
# =====================================================

joblib.dump(
    {
        "model": model,
        "threshold": best_threshold
    },
    "models/random_forest_threshold_model.joblib"
)

metrics = {
    "threshold": best_threshold,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "pr_auc": float(pr_auc),
    "roc_auc": float(roc_auc),
    "confusion_matrix": cm.tolist()
}

with open(
    "results/random_forest_threshold_metrics.json",
    "w"
) as file:
    json.dump(metrics, file, indent=4)

print("\nSaved:")
print("models/random_forest_threshold_model.joblib")
print("results/threshold_comparison.csv")
print("results/random_forest_threshold_metrics.json")

print("\nPIPELINE COMPLETED!")
