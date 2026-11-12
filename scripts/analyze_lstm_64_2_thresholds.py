
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


# =====================================================
# CONFIGURATION
# =====================================================

DEVICE = torch.device("cpu")
BATCH_SIZE = 256

MODEL_PATH = "models/LSTM_64_2.pt"
VALIDATION_PATH = "datasets/lstm/validation.npz"

CSV_OUTPUT = "results/lstm_64_2_threshold_analysis.csv"
JSON_OUTPUT = "results/lstm_64_2_threshold_analysis.json"

MIN_RECALL_TARGETS = [0.65, 0.70, 0.75, 0.80]


# =====================================================
# MODEL
# Must match the training architecture
# =====================================================

class LSTMClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        return self.fc(last_output).squeeze(1)


# =====================================================
# LOAD CHECKPOINT
# =====================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

input_size = int(checkpoint["input_size"])
hidden_size = int(checkpoint["hidden_size"])
num_layers = 2

model = LSTMClassifier(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# =====================================================
# LOAD VALIDATION DATA
# =====================================================

data = np.load(VALIDATION_PATH)

X_val = torch.tensor(data["X"], dtype=torch.float32)
y_val = data["y"].astype(int)

dataset = TensorDataset(
    X_val,
    torch.tensor(y_val, dtype=torch.float32)
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =====================================================
# PREDICT VALIDATION PROBABILITIES
# =====================================================

probabilities = []

with torch.no_grad():

    for X_batch, _ in loader:

        X_batch = X_batch.to(DEVICE)

        logits = model(X_batch)
        probs = torch.sigmoid(logits)

        probabilities.extend(probs.cpu().numpy())

probabilities = np.asarray(probabilities)

if len(probabilities) != len(y_val):
    raise ValueError("Prediction count does not match validation labels.")

if len(np.unique(y_val)) != 2:
    raise ValueError("Validation dataset must contain both classes.")


# =====================================================
# PRECISION-RECALL CURVE
# =====================================================

precision_values, recall_values, thresholds = (
    precision_recall_curve(y_val, probabilities)
)

# Last precision/recall point has no corresponding threshold.
precision_values = precision_values[:-1]
recall_values = recall_values[:-1]

f1_values = (
    2 * precision_values * recall_values
    / (precision_values + recall_values + 1e-12)
)

pr_auc = average_precision_score(y_val, probabilities)
roc_auc = roc_auc_score(y_val, probabilities)


# =====================================================
# BUILD THRESHOLD ANALYSIS
# =====================================================

rows = []

for i, threshold in enumerate(thresholds):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    rows.append({
        "threshold": float(threshold),
        "precision": float(precision_values[i]),
        "recall": float(recall_values[i]),
        "f1": float(f1_values[i]),
        "accuracy": float(accuracy_score(y_val, predictions))
    })

df = pd.DataFrame(rows)

df.to_csv(CSV_OUTPUT, index=False)


# =====================================================
# SELECT THRESHOLDS BY MINIMUM RECALL
# =====================================================

selected_results = []

for min_recall in MIN_RECALL_TARGETS:

    candidates = df[df["recall"] >= min_recall]

    if candidates.empty:
        print(
            f"No threshold satisfies recall >= {min_recall:.0%}"
        )
        continue

    # Highest precision while respecting minimum recall
    best = candidates.sort_values(
        ["precision", "f1"],
        ascending=False
    ).iloc[0]

    selected_results.append({
        "minimum_recall_target": min_recall,
        "threshold": float(best["threshold"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
        "accuracy": float(best["accuracy"])
    })


# =====================================================
# BEST F1 THRESHOLD
# =====================================================

best_f1 = df.loc[df["f1"].idxmax()]

best_f1_result = {
    "threshold": float(best_f1["threshold"]),
    "precision": float(best_f1["precision"]),
    "recall": float(best_f1["recall"]),
    "f1": float(best_f1["f1"]),
    "accuracy": float(best_f1["accuracy"])
}


# =====================================================
# DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 65)
print("LSTM_64_2 - VALIDATION THRESHOLD ANALYSIS")
print("=" * 65)

print("Model:", MODEL_PATH)
print("Architecture: 2 layers")
print("Hidden size:", hidden_size)
print("Validation sequences:", len(y_val))
print("Validation Risk:", int(np.sum(y_val == 1)))
print("Validation No Risk:", int(np.sum(y_val == 0)))

print(f"\nPR-AUC : {pr_auc:.6f}")
print(f"ROC-AUC: {roc_auc:.6f}")

print("\nBEST THRESHOLD BY MINIMUM RECALL")
print("-" * 65)

for result in selected_results:

    print(
        f"Recall >= {result['minimum_recall_target']:.0%} | "
        f"Threshold: {result['threshold']:.6f} | "
        f"Precision: {result['precision']:.4f} | "
        f"Recall: {result['recall']:.4f} | "
        f"F1: {result['f1']:.4f}"
    )

print("\nBEST F1 THRESHOLD")
print("-" * 65)

print(
    f"Threshold: {best_f1_result['threshold']:.6f} | "
    f"Precision: {best_f1_result['precision']:.4f} | "
    f"Recall: {best_f1_result['recall']:.4f} | "
    f"F1: {best_f1_result['f1']:.4f}"
)


# =====================================================
# SAVE SUMMARY
# =====================================================

summary = {
    "model": MODEL_PATH,
    "architecture": "LSTM_64_2",
    "dataset": VALIDATION_PATH,
    "validation_sequences": int(len(y_val)),
    "validation_risk": int(np.sum(y_val == 1)),
    "validation_no_risk": int(np.sum(y_val == 0)),
    "pr_auc": float(pr_auc),
    "roc_auc": float(roc_auc),
    "thresholds_by_minimum_recall": selected_results,
    "best_f1_threshold": best_f1_result,
    "threshold_selection_dataset": "validation_only"
}

with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("\nSaved CSV:", CSV_OUTPUT)
print("Saved JSON:", JSON_OUTPUT)
print("\nThreshold analysis completed.")
