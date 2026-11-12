
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT = Path(__file__).resolve().parent.parent

DATASET = (
    PROJECT / "datasets" /
    "pedestrian_vehicle_risk_dataset_v3.csv"
)

RESULTS = PROJECT / "results"
MODELS = PROJECT / "models"

RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

MODEL_FILE = MODELS / "random_forest_model.joblib"

RANDOM_STATE = 42


# ==========================================
# 2. FEATURES
# ==========================================

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

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "risk_label"

GROUP_COLUMNS = [
    "vehicle_id",
    "pedestrian_id"
]


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("=" * 55)
print("RANDOM FOREST - PEDESTRIAN SAFETY")
print("=" * 55)

print("\nLoading dataset...")

required_columns = FEATURES + [TARGET] + GROUP_COLUMNS

df = pd.read_csv(
    DATASET,
    usecols=required_columns
)

df = df.dropna(subset=[TARGET]).copy()

df[TARGET] = df[TARGET].astype(int)

print("Total labeled records:", f"{len(df):,}")

print("\nClass distribution:")
print(df[TARGET].value_counts())


# ==========================================
# 4. PREPARE FEATURES AND GROUPS
# ==========================================

X = df[FEATURES].copy()
y = df[TARGET].copy()

# IDs are used ONLY to prevent pair leakage.
# They are not model features.

groups = (
    df["vehicle_id"].astype(str)
    + "__"
    + df["pedestrian_id"].astype(str)
)

for column in NUMERIC_FEATURES:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

for column in CATEGORICAL_FEATURES:
    X[column] = X[column].fillna("Unknown").astype(str)


# ==========================================
# 5. GROUPED TRAIN / TEST SPLIT
# ==========================================

print("\nSplitting data by vehicle-pedestrian pairs...")

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

train_indices, test_indices = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

train_groups = set(groups.iloc[train_indices])
test_groups = set(groups.iloc[test_indices])

assert train_groups.isdisjoint(test_groups)

print("Training records:", f"{len(X_train):,}")
print("Testing records:", f"{len(X_test):,}")

print("Training risk cases:", int(y_train.sum()))
print("Testing risk cases:", int(y_test.sum()))

print("Pair overlap:", len(train_groups & test_groups))


# ==========================================
# 6. PREPROCESSING
# ==========================================

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


# ==========================================
# 7. RANDOM FOREST MODEL
# ==========================================

random_forest = RandomForestClassifier(
    n_estimators=150,
    max_depth=18,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", random_forest)
])


# ==========================================
# 8. TRAIN
# ==========================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("\nTraining completed!")


# ==========================================
# 9. PREDICTIONS
# ==========================================

print("\nEvaluating model...")

y_pred = model.predict(X_test)

y_proba = model.predict_proba(X_test)[:, 1]


# ==========================================
# 10. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

pr_auc = average_precision_score(y_test, y_proba)

roc_auc = roc_auc_score(y_test, y_proba)

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\n" + "=" * 55)
print("RANDOM FOREST RESULTS")
print("=" * 55)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(matrix)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Risk", "Risk"],
        zero_division=0
    )
)


# ==========================================
# 11. FEATURE IMPORTANCE
# ==========================================

print("\nCalculating feature importance...")

fitted_preprocessor = model.named_steps["preprocessor"]

feature_names = fitted_preprocessor.get_feature_names_out()

importances = model.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(
    "importance",
    ascending=False
)

importance_file = RESULTS / "random_forest_feature_importance.csv"

importance_df.to_csv(
    importance_file,
    index=False
)

print("\nTop 15 important features:")
print(importance_df.head(15).to_string(index=False))


# ==========================================
# 12. SAVE MODEL
# ==========================================

print("\nSaving model...")

joblib.dump(model, MODEL_FILE)

print("Model saved:", MODEL_FILE)


# ==========================================
# 13. SAVE METRICS
# ==========================================

metrics_file = RESULTS / "random_forest_metrics.txt"

with open(metrics_file, "w", encoding="utf-8") as file:

    file.write("RANDOM FOREST EVALUATION\n")
    file.write("=" * 40 + "\n")

    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n")
    file.write(f"PR-AUC: {pr_auc:.4f}\n")
    file.write(f"ROC-AUC: {roc_auc:.4f}\n")

    file.write("\nConfusion Matrix:\n")
    file.write(str(matrix))

    file.write("\n\nClassification Report:\n")
    file.write(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Risk", "Risk"],
            zero_division=0
        )
    )

print("\nMetrics saved:", metrics_file)

print("\n" + "=" * 55)
print("RANDOM FOREST PIPELINE COMPLETED")
print("=" * 55)
