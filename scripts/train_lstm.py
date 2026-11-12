
import os
import json
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
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
# 1. CONFIGURATION
# ============================================================

SEED = 42

DATA_DIR = "datasets/lstm"
MODEL_DIR = "models"
RESULTS_DIR = "results"

SEQ_LEN = 20
HIDDEN_SIZE = 64
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE = 4

MIN_RECALL = 0.65

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(min(4, os.cpu_count() or 1))

DEVICE = torch.device("cpu")

print("Device:", DEVICE)
print("PyTorch:", torch.__version__)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n[1/6] Loading prepared sequences...")

train_data = np.load(os.path.join(DATA_DIR, "train.npz"))
val_data = np.load(os.path.join(DATA_DIR, "validation.npz"))
test_data = np.load(os.path.join(DATA_DIR, "test.npz"))

X_train = train_data["X"].astype(np.float32)
y_train = train_data["y"].astype(np.float32)

X_val = val_data["X"].astype(np.float32)
y_val = val_data["y"].astype(np.float32)

X_test = test_data["X"].astype(np.float32)
y_test = test_data["y"].astype(np.float32)

input_size = X_train.shape[2]

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
print("Input features:", input_size)


# ============================================================
# 3. CONVERT TO PYTORCH TENSORS
# ============================================================

print("\n[2/6] Preparing PyTorch tensors...")

train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train)
)

val_dataset = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(y_val)
)

test_dataset = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(y_test)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 4. LSTM MODEL
# ============================================================

print("\n[3/6] Building LSTM model...")


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

        output, (hidden, cell) = self.lstm(x)

        # Last time step
        last_output = output[:, -1, :]

        last_output = self.dropout(last_output)

        return self.fc(last_output).squeeze(1)


model = PedestrianRiskLSTM(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE
).to(DEVICE)

print(model)


# ============================================================
# 5. LOSS FUNCTION AND OPTIMIZER
# ============================================================

positive_count = int(y_train.sum())
negative_count = len(y_train) - positive_count

if positive_count == 0:
    raise ValueError("No positive training sequences found.")

pos_weight_value = negative_count / positive_count

pos_weight = torch.tensor(
    [pos_weight_value],
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nPositive samples:", positive_count)
print("Negative samples:", negative_count)
print("Positive weight:", round(pos_weight_value, 2))


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def predict_probabilities(loader):

    model.eval()

    all_probabilities = []
    all_labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)

            logits = model(X_batch)

            probabilities = torch.sigmoid(logits)

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_labels.extend(
                y_batch.numpy()
            )

    return (
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_probabilities)
    )


def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
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
        "confusion_matrix": cm.tolist(),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    return metrics


# ============================================================
# 7. TRAINING LOOP
# ============================================================

print("\n[4/6] Starting LSTM training...")

best_val_pr_auc = -1
best_epoch = 0
best_state = None
epochs_without_improvement = 0

history = []

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(
            logits,
            y_batch
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    train_loss = total_loss / len(train_dataset)

    # Validation
    val_labels, val_probabilities = predict_probabilities(
        val_loader
    )

    val_pr_auc = average_precision_score(
        val_labels,
        val_probabilities
    )

    val_roc_auc = roc_auc_score(
        val_labels,
        val_probabilities
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_pr_auc": val_pr_auc,
        "validation_roc_auc": val_roc_auc
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.5f} | "
        f"Val PR-AUC: {val_pr_auc:.5f} | "
        f"Val ROC-AUC: {val_roc_auc:.5f}"
    )

    # Save best model based on validation PR-AUC
    if val_pr_auc > best_val_pr_auc:

        best_val_pr_auc = val_pr_auc
        best_epoch = epoch

        best_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping activated.")

        break


if best_state is None:
    raise RuntimeError("Training failed: no best model was saved.")

model.load_state_dict(best_state)

print("\nBest epoch:", best_epoch)
print("Best validation PR-AUC:", round(best_val_pr_auc, 5))


# ============================================================
# 8. SELECT THRESHOLD USING VALIDATION ONLY
# ============================================================

print("\n[5/6] Selecting decision threshold...")

val_labels, val_probabilities = predict_probabilities(
    val_loader
)

threshold_results = []

for threshold in np.arange(0.05, 0.951, 0.01):

    metrics = calculate_metrics(
        val_labels,
        val_probabilities,
        threshold
    )

    threshold_results.append(metrics)


# Prefer highest precision with recall >= 0.65
eligible = [
    result for result in threshold_results
    if result["recall"] >= MIN_RECALL
]

if eligible:

    selected = max(
        eligible,
        key=lambda item: (
            item["precision"],
            item["f1"]
        )
    )

    selection_rule = (
        "Highest validation precision with recall >= 0.65"
    )

else:

    selected = max(
        threshold_results,
        key=lambda item: item["f1"]
    )

    selection_rule = (
        "Fallback: highest validation F1"
    )


selected_threshold = selected["threshold"]

print("Selection rule:", selection_rule)
print("Selected threshold:", round(selected_threshold, 2))
print("Validation precision:", round(selected["precision"], 4))
print("Validation recall:", round(selected["recall"], 4))
print("Validation F1:", round(selected["f1"], 4))


# ============================================================
# 9. FINAL TEST EVALUATION
# ============================================================

print("\n[6/6] Evaluating on untouched test set...")

test_labels, test_probabilities = predict_probabilities(
    test_loader
)

test_metrics = calculate_metrics(
    test_labels,
    test_probabilities,
    selected_threshold
)

print("\n========== LSTM TEST RESULTS ==========")

for key, value in test_metrics.items():

    if key != "confusion_matrix":
        print(f"{key}: {value}")

print("\nConfusion Matrix:")
print(np.array(test_metrics["confusion_matrix"]))


# ============================================================
# 10. SAVE MODEL AND RESULTS
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "lstm_model.pt"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "sequence_length": SEQ_LEN,
        "threshold": selected_threshold
    },
    model_path
)

history_path = os.path.join(
    RESULTS_DIR,
    "lstm_training_history.csv"
)

pd.DataFrame(history).to_csv(
    history_path,
    index=False
)

threshold_path = os.path.join(
    RESULTS_DIR,
    "lstm_threshold_comparison.csv"
)

pd.DataFrame(threshold_results).to_csv(
    threshold_path,
    index=False
)

metrics_path = os.path.join(
    RESULTS_DIR,
    "lstm_metrics.json"
)

final_results = {
    "best_epoch": best_epoch,
    "best_validation_pr_auc": best_val_pr_auc,
    "threshold_selection_rule": selection_rule,
    "selected_threshold": selected_threshold,
    "validation_metrics": selected,
    "test_metrics": test_metrics,
    "model_configuration": {
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "sequence_length": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "pos_weight": pos_weight_value
    }
}

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_results,
        file,
        indent=4
    )

print("\nTraining completed successfully!")

print("Model:", model_path)
print("Metrics:", metrics_path)
print("History:", history_path)
print("Threshold comparison:", threshold_path)
