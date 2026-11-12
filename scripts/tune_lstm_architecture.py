
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)

# =====================================================
# CONFIGURATION
# =====================================================

SEED = 42
BATCH_SIZE = 256
EPOCHS = 25
PATIENCE = 5
LEARNING_RATE = 0.001
POS_WEIGHT = 40.0
MIN_RECALL = 0.65

DEVICE = torch.device("cpu")

CONFIGS = [
    {"name": "LSTM_64_1", "hidden_size": 64, "num_layers": 1},
    {"name": "LSTM_128_1", "hidden_size": 128, "num_layers": 1},
    {"name": "LSTM_64_2", "hidden_size": 64, "num_layers": 2}
]

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =====================================================
# DATA
# =====================================================

def load_data(name):
    data = np.load(f"datasets/lstm/{name}.npz")

    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32)

    return TensorDataset(X, y)


train_dataset = load_data("train")
val_dataset = load_data("validation")

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


# =====================================================
# MODEL
# =====================================================

class LSTMClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        return self.fc(self.dropout(last_output)).squeeze(1)


# =====================================================
# PREDICTION
# =====================================================

def predict(model, loader):

    model.eval()
    probs = []
    labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:

            logits = model(X_batch.to(DEVICE))
            probabilities = torch.sigmoid(logits)

            probs.extend(probabilities.cpu().numpy())
            labels.extend(y_batch.numpy())

    return np.array(probs), np.array(labels)


# =====================================================
# THRESHOLD SELECTION
# =====================================================

def choose_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    candidates = []

    for i, threshold in enumerate(thresholds):

        if recall[i] >= MIN_RECALL:

            f1 = (
                2 * precision[i] * recall[i] /
                (precision[i] + recall[i] + 1e-10)
            )

            candidates.append({
                "threshold": float(threshold),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1)
            })

    return max(
        candidates,
        key=lambda x: (x["precision"], x["f1"])
    )


# =====================================================
# TRAINING
# =====================================================

all_results = []
all_history = []

for config in CONFIGS:

    set_seed(SEED)

    name = config["name"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]

    print("\n" + "=" * 60)
    print("TRAINING:", name)
    print("=" * 60)

    model = LSTMClassifier(
        input_size,
        hidden_size,
        num_layers
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT])
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_pr_auc = -1
    best_epoch = 0
    best_state = None
    patience_counter = 0

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

        val_probs, val_labels = predict(model, val_loader)

        val_pr_auc = average_precision_score(
            val_labels,
            val_probs
        )

        val_roc_auc = roc_auc_score(
            val_labels,
            val_probs
        )

        all_history.append({
            "architecture": name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_pr_auc": val_pr_auc,
            "val_roc_auc": val_roc_auc
        })

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {train_loss:.5f} | "
            f"PR-AUC: {val_pr_auc:.5f} | "
            f"ROC-AUC: {val_roc_auc:.5f}"
        )

        if val_pr_auc > best_pr_auc:

            best_pr_auc = val_pr_auc
            best_epoch = epoch
            patience_counter = 0

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)

    val_probs, val_labels = predict(model, val_loader)

    threshold_result = choose_threshold(
        val_labels,
        val_probs
    )

    model_path = f"models/{name}.pt"

    torch.save({
        "model_state_dict": best_state,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "sequence_length": 20,
        "pos_weight": POS_WEIGHT,
        "best_epoch": best_epoch,
        "threshold": threshold_result["threshold"]
    }, model_path)

    result = {
        "architecture": name,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "best_epoch": best_epoch,
        "val_pr_auc": best_pr_auc,
        "val_roc_auc": roc_auc_score(val_labels, val_probs),
        "threshold": threshold_result["threshold"],
        "val_precision": threshold_result["precision"],
        "val_recall": threshold_result["recall"],
        "val_f1": threshold_result["f1"],
        "model_path": model_path
    }

    all_results.append(result)

    print("\nValidation result:")
    print(json.dumps(result, indent=2))


# =====================================================
# SAVE COMPARISON
# =====================================================

df = pd.DataFrame(all_results)

df = df.sort_values(
    by=["val_pr_auc", "val_f1"],
    ascending=False
)

df.to_csv(
    "results/lstm_architecture_comparison.csv",
    index=False
)

pd.DataFrame(all_history).to_csv(
    "results/lstm_architecture_history.csv",
    index=False
)

print("\n" + "=" * 60)
print("ARCHITECTURE COMPARISON - VALIDATION")
print("=" * 60)

print(df.to_string(index=False))

print("\nTest dataset was NOT used.")
print("Experiment completed.")
