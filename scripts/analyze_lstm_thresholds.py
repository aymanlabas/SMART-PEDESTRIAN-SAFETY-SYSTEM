
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cpu"
BATCH_SIZE = 256

# Load selected model
with open("results/lstm_best_configuration.json", encoding="utf-8") as f:
    config = json.load(f)

checkpoint = torch.load(
    config["model_path"],
    map_location=DEVICE,
    weights_only=False
)

# Same LSTM architecture
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
        output, _ = self.lstm(x)
        return self.fc(
            self.dropout(output[:, -1, :])
        ).squeeze(1)


# Load VALIDATION only
data = np.load("datasets/lstm/validation.npz")

X = torch.tensor(data["X"], dtype=torch.float32)
y = data["y"].astype(int)

loader = DataLoader(
    TensorDataset(X, torch.tensor(y, dtype=torch.float32)),
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = LSTMClassifier(
    checkpoint["input_size"],
    checkpoint["hidden_size"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Predict validation probabilities
probabilities = []

with torch.no_grad():
    for X_batch, _ in loader:
        logits = model(X_batch)
        probs = torch.sigmoid(logits)
        probabilities.extend(probs.numpy())

probabilities = np.array(probabilities)

precision, recall, thresholds = precision_recall_curve(
    y,
    probabilities
)

print("Validation PR-AUC:", average_precision_score(y, probabilities))

# Find best precision for each minimum recall
targets = [0.65, 0.70, 0.75, 0.80]
results = []

for minimum_recall in targets:

    candidates = []

    for i, threshold in enumerate(thresholds):

        if recall[i] >= minimum_recall:
            candidates.append({
                "minimum_recall": minimum_recall,
                "threshold": float(threshold),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(
                    2 * precision[i] * recall[i] /
                    (precision[i] + recall[i] + 1e-10)
                )
            })

    if candidates:
        best = max(
            candidates,
            key=lambda x: (x["precision"], x["f1"])
        )
        results.append(best)

df = pd.DataFrame(results)

print("\nVALIDATION THRESHOLD COMPARISON")
print(df.to_string(index=False))

df.to_csv(
    "results/lstm_threshold_analysis_validation.csv",
    index=False
)

print("\nSaved: results/lstm_threshold_analysis_validation.csv")
print("TEST dataset was NOT used.")
