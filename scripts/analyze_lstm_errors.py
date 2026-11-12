
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "LSTM_64_2.pt"
TEST_NPZ = ROOT / "datasets" / "lstm" / "test.npz"
METADATA_PATH = ROOT / "datasets" / "lstm" / "test_sequence_metadata.csv"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.07262967526912689
BATCH_SIZE = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LSTM MODEL
# ============================================================

class PedestrianLSTM(nn.Module):

    def __init__(
        self,
        input_size=35,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        return self.fc(last_output).squeeze(1)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model():

    print("\n[1/5] Loading LSTM model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove DataParallel prefix if present
    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
    }

    # Infer architecture from saved weights
    input_size = state_dict["lstm.weight_ih_l0"].shape[1]
    hidden_size = state_dict["lstm.weight_hh_l0"].shape[1]

    num_layers = len([
        key for key in state_dict
        if key.startswith("lstm.weight_ih_l")
    ])

    model = PedestrianLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print(f"Model: {MODEL_PATH.name}")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"LSTM layers: {num_layers}")
    print(f"Device: {DEVICE}")

    return model


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    print("\n[2/5] Loading Test dataset and metadata...")

    data = np.load(TEST_NPZ)

    X_test = data["X"].astype(np.float32)
    y_test = data["y"].astype(np.int64)

    metadata = pd.read_csv(METADATA_PATH)

    if len(X_test) != len(metadata):
        raise ValueError(
            f"Sequence count mismatch: NPZ={len(X_test)}, "
            f"Metadata={len(metadata)}"
        )

    if not np.array_equal(
        y_test,
        metadata["risk_label"].to_numpy(dtype=np.int64),
    ):
        raise ValueError(
            "Labels do not match metadata order. "
            "Stop analysis and verify sequence alignment."
        )

    print(f"Test sequences: {len(X_test)}")
    print(f"Risk: {int(y_test.sum())}")
    print(f"No Risk: {int((y_test == 0).sum())}")
    print(f"Input shape: {X_test.shape}")

    return X_test, y_test, metadata


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def predict(model, X_test):

    print("\n[3/5] Generating predictions...")

    probabilities = []

    with torch.no_grad():

        for start in range(0, len(X_test), BATCH_SIZE):

            end = start + BATCH_SIZE

            batch = torch.tensor(
                X_test[start:end],
                dtype=torch.float32,
                device=DEVICE,
            )

            logits = model(batch)
            probs = torch.sigmoid(logits)

            probabilities.extend(
                probs.cpu().numpy().tolist()
            )

    return np.asarray(probabilities)


# ============================================================
# EVALUATION AND ERROR ANALYSIS
# ============================================================

def evaluate(y_true, probabilities, metadata):

    print("\n[4/5] Evaluating predictions...")

    y_pred = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "model": "LSTM_64_2",
        "threshold": THRESHOLD,
        "total_sequences": int(len(y_true)),
        "risk_sequences": int(y_true.sum()),
        "no_risk_sequences": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    results = metadata.copy()

    results["actual_risk"] = y_true
    results["risk_probability"] = probabilities
    results["predicted_risk"] = y_pred

    def classify_error(row):

        if row["actual_risk"] == 0 and row["predicted_risk"] == 1:
            return "False Positive"

        if row["actual_risk"] == 1 and row["predicted_risk"] == 0:
            return "False Negative"

        if row["actual_risk"] == 1 and row["predicted_risk"] == 1:
            return "True Positive"

        return "True Negative"

    results["prediction_result"] = results.apply(
        classify_error,
        axis=1,
    )

    errors = results[
        results["prediction_result"].isin(
            ["False Positive", "False Negative"]
        )
    ].copy()

    # Save all predictions
    predictions_path = RESULTS_DIR / "test_predictions.csv"
    results.to_csv(predictions_path, index=False)

    # Save only errors
    errors_path = RESULTS_DIR / "test_errors.csv"
    errors.to_csv(errors_path, index=False)

    # Error analysis by zone
    zone_analysis = (
        results.groupby("zone_type", dropna=False)
        .agg(
            total_sequences=("actual_risk", "size"),
            actual_risk=("actual_risk", "sum"),
            predicted_risk=("predicted_risk", "sum"),
            false_positives=(
                "prediction_result",
                lambda x: (x == "False Positive").sum(),
            ),
            false_negatives=(
                "prediction_result",
                lambda x: (x == "False Negative").sum(),
            ),
        )
        .reset_index()
    )

    zone_analysis["error_count"] = (
        zone_analysis["false_positives"]
        + zone_analysis["false_negatives"]
    )

    zone_analysis.to_csv(
        RESULTS_DIR / "error_analysis_by_zone.csv",
        index=False,
    )

    # Summary JSON
    with open(
        RESULTS_DIR / "error_analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(metrics, file, indent=4)

    return metrics, errors, zone_analysis


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LSTM_64_2 - TEST ERROR ANALYSIS")
    print("=" * 60)

    model = load_model()

    X_test, y_test, metadata = load_test_data()

    probabilities = predict(model, X_test)

    metrics, errors, zone_analysis = evaluate(
        y_test,
        probabilities,
        metadata,
    )

    print("\n[5/5] Analysis completed.")

    print("\n========== TEST METRICS ==========")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\n========== ERROR DETAILS ==========")
    print(f"False Positives: {(errors['prediction_result'] == 'False Positive').sum()}")
    print(f"False Negatives: {(errors['prediction_result'] == 'False Negative').sum()}")

    print("\n========== ERRORS BY ZONE ==========")
    print(zone_analysis.to_string(index=False))

    print("\n========== GENERATED FILES ==========")
    print("results/test_predictions.csv")
    print("results/test_errors.csv")
    print("results/error_analysis_by_zone.csv")
    print("results/error_analysis_summary.json")


if __name__ == "__main__":
    main()
