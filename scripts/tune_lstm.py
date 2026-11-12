
import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score
)

# =========================================================
# CONFIGURATION
# =========================================================

SEED = 42
BATCH_SIZE = 256
EPOCHS = 20
PATIENCE = 4
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64

POS_WEIGHTS = [40.0, 80.0, 120.0, 172.67]
MIN_REQUIRED_RECALL = 0.65

DEVICE = torch.device("cpu")

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Device:", DEVICE)

# =========================================================
# LOAD DATA
# =========================================================

def load_dataset(name):
    data = np.load(f"datasets/lstm/{name}.npz")

    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32)

    return TensorDataset(X, y)


train_dataset = load_dataset("train")
val_dataset = load_dataset("validation")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

input_size = train_dataset.tensors[0].shape[2]

print("\nTraining sequences:", len(train_dataset))
print("Validation sequences:", len(val_dataset))
print("Input features:", input_size)

# =========================================================
# LSTM MODEL
# =========================================================

class LSTMClassifier(nn.Module):

    def __init__(self, input_size, hidden_size):
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

        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)

        return self.fc(last_output).squeeze(1)


# =========================================================
# VALIDATION
# =========================================================

def predict_probabilities(model, loader):

    model.eval()
    probabilities = []
    labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)

            logits = model(X_batch)
            probs = torch.sigmoid(logits)

            probabilities.extend(probs.cpu().numpy())
            labels.extend(y_batch.numpy())

    return np.array(probabilities), np.array(labels)


def select_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    candidates = []

    for i, threshold in enumerate(thresholds):

        if recall[i] >= MIN_REQUIRED_RECALL:

            candidates.append({
                "threshold": float(threshold),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(
                    2 * precision[i] * recall[i] /
                    (precision[i] + recall[i] + 1e-10)
                )
            })

    if not candidates:
        raise RuntimeError(
            "No threshold satisfies the required recall."
        )

    # Highest precision, then highest F1
    best = max(
        candidates,
        key=lambda x: (x["precision"], x["f1"])
    )

    return best


# =========================================================
# TRAIN ONE MODEL
# =========================================================

def train_one_model(pos_weight):

    print("\n" + "=" * 60)
    print("TRAINING WITH POS_WEIGHT =", pos_weight)
    print("=" * 60)

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight],
            dtype=torch.float32
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_pr_auc = -1
    best_epoch = 0
    patience_counter = 0
    best_state = None

    history = []

    for epoch in range(1, EPOCHS + 1):

        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item() * len(y_batch)

        train_loss = total_loss / len(train_dataset)

        val_probs, val_labels = predict_probabilities(
            model,
            val_loader
        )

        val_pr_auc = average_precision_score(
            val_labels,
            val_probs
        )

        val_roc_auc = roc_auc_score(
            val_labels,
            val_probs
        )

        history.append({
            "pos_weight": pos_weight,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_pr_auc": val_pr_auc,
            "val_roc_auc": val_roc_auc
        })

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {train_loss:.5f} | "
            f"Val PR-AUC: {val_pr_auc:.5f} | "
            f"Val ROC-AUC: {val_roc_auc:.5f}"
        )

        if val_pr_auc > best_val_pr_auc:

            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            patience_counter = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    # Restore best epoch
    model.load_state_dict(best_state)

    val_probs, val_labels = predict_probabilities(
        model,
        val_loader
    )

    threshold_result = select_threshold(
        val_labels,
        val_probs
    )

    model_path = (
        f"models/lstm_posweight_{str(pos_weight).replace('.', '_')}.pt"
    )

    torch.save({
        "model_state_dict": best_state,
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "sequence_length": 20,
        "pos_weight": pos_weight,
        "best_epoch": best_epoch,
        "threshold": threshold_result["threshold"]
    }, model_path)

    result = {
        "pos_weight": pos_weight,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_pr_auc,
        "val_roc_auc": roc_auc_score(val_labels, val_probs),
        "threshold": threshold_result["threshold"],
        "val_precision": threshold_result["precision"],
        "val_recall": threshold_result["recall"],
        "val_f1": threshold_result["f1"],
        "model_path": model_path
    }

    print("\nValidation result:")
    print(json.dumps(result, indent=2))

    return result, history


# =========================================================
# RUN EXPERIMENTS
# =========================================================

all_results = []
all_history = []

for weight in POS_WEIGHTS:

    result, history = train_one_model(weight)

    all_results.append(result)
    all_history.extend(history)


# =========================================================
# SELECT BEST CONFIGURATION USING VALIDATION ONLY
# =========================================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by=["val_precision", "val_f1"],
    ascending=False
)

results_df.to_csv(
    "results/lstm_posweight_comparison.csv",
    index=False
)

pd.DataFrame(all_history).to_csv(
    "results/lstm_posweight_training_history.csv",
    index=False
)

best = results_df.iloc[0].to_dict()

with open(
    "results/lstm_best_configuration.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(best, f, indent=4)

print("\n" + "=" * 60)
print("FINAL VALIDATION COMPARISON")
print("=" * 60)

print(
    results_df[
        [
            "pos_weight",
            "best_epoch",
            "best_val_pr_auc",
            "threshold",
            "val_precision",
            "val_recall",
            "val_f1"
        ]
    ].to_string(index=False)
)

print("\nBEST CONFIGURATION:")
print(json.dumps(best, indent=2))

print("\nTest dataset was NOT used.")
print("Experiment completed.")
