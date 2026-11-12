
import json
import numpy as np
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
    classification_report
)

from torch.utils.data import DataLoader, TensorDataset


# =====================================================
# CONFIGURATION
# =====================================================

DEVICE = torch.device("cpu")
BATCH_SIZE = 256

CONFIG_PATH = "results/lstm_architecture_comparison.csv"
MODEL_PATH = "models/LSTM_64_2.pt"
TEST_PATH = "datasets/lstm/test.npz"

RESULTS_PATH = "results/lstm_64_2_test_metrics.json"

# Threshold selected using VALIDATION only
THRESHOLD = 0.07262967526912689


# =====================================================
# LSTM MODEL
# =====================================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers
    ):
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

input_size = checkpoint["input_size"]
hidden_size = checkpoint["hidden_size"]

# Architecture selected during training
num_layers = 2

model = LSTMClassifier(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# =====================================================
# LOAD TEST DATA
# =====================================================

data = np.load(TEST_PATH)

X_test = torch.tensor(
    data["X"],
    dtype=torch.float32
)

y_test = data["y"].astype(int)

test_dataset = TensorDataset(
    X_test,
    torch.tensor(y_test, dtype=torch.float32)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =====================================================
# PREDICTION
# =====================================================

probabilities = []

with torch.no_grad():

    for X_batch, _ in test_loader:

        X_batch = X_batch.to(DEVICE)

        logits = model(X_batch)

        probs = torch.sigmoid(logits)

        probabilities.extend(
            probs.cpu().numpy()
        )

probabilities = np.array(probabilities)


# =====================================================
# CLASSIFICATION
# =====================================================

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(
    y_test, predictions
)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

pr_auc = average_precision_score(
    y_test, probabilities
)

roc_auc = roc_auc_score(
    y_test, probabilities
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)


# =====================================================
# DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 60)
print("FINAL LSTM_64_2 TEST EVALUATION")
print("=" * 60)

print("Model:", MODEL_PATH)
print("Architecture: 2 LSTM layers")
print("Input size:", input_size)
print("Hidden size:", hidden_size)
print("Threshold:", round(THRESHOLD, 6))

if "pos_weight" in checkpoint:
    print("pos_weight:", checkpoint["pos_weight"])

if "best_epoch" in checkpoint:
    print("Best epoch:", checkpoint["best_epoch"])

print("\nTEST DATA")
print("Total sequences:", len(y_test))
print("Actual Risk:", int(np.sum(y_test == 1)))
print("Actual No Risk:", int(np.sum(y_test == 0)))

print("\nMETRICS")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nCONFUSION MATRIX")
print("Rows = Actual | Columns = Predicted")
print("              Pred No Risk   Pred Risk")
print(f"Actual No Risk     {cm[0][0]:6d}       {cm[0][1]:6d}")
print(f"Actual Risk        {cm[1][0]:6d}       {cm[1][1]:6d}")

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["No Risk", "Risk"],
        zero_division=0
    )
)


# =====================================================
# SAVE RESULTS
# =====================================================

results = {

    "model": MODEL_PATH,

    "architecture": "LSTM_64_2",

    "input_size": int(input_size),

    "hidden_size": int(hidden_size),

    "num_layers": num_layers,

    "threshold": float(THRESHOLD),

    "test_sequences": int(len(y_test)),

    "actual_risk": int(np.sum(y_test == 1)),

    "actual_no_risk": int(np.sum(y_test == 0)),

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1": float(f1),

    "pr_auc": float(pr_auc),

    "roc_auc": float(roc_auc),

    "confusion_matrix": cm.tolist()

}

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print("\nResults saved:", RESULTS_PATH)

print("\nEvaluation completed successfully.")
