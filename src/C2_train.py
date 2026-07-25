import os
import json
import pickle
import random
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================================
# CONFIGURATION
# ==========================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREVIOUS_OUTPUT = "output/C1_quantile_forecasting"
LOOKBACK = 96
HORIZON = 288
EPOCHS = 50
BATCH_SIZE = 64
N_TRIALS = 20
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# ==========================================================
# LOAD OUTPUT DIRECTORY
# ==========================================================
OUT_DIR = "output/C2_train"
MODEL_DIR = os.path.join(OUT_DIR, "models")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


# ==========================================================
# DATASET CLASS
# ==========================================================
class ForecastDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


# ==========================================================
# QUANTILE LOSS
# ==========================================================
def quantile_loss(pred, target, q):
    error = target - pred
    return torch.mean(torch.maximum(q * error, (q - 1) * error))


def multi_quantile_loss(prediction, target):
    q10 = prediction[:, :, 0]
    q50 = prediction[:, :, 1]
    q90 = prediction[:, :, 2]
    loss = (
        quantile_loss(q10, target, 0.1)
        + quantile_loss(q50, target, 0.5)
        + quantile_loss(q90, target, 0.9)
    )
    return loss / 3


# ==========================================================
# MODELS
# ==========================================================
class QuantileLSTM(nn.Module):
    def __init__(self, input_size, hidden, layers, dropout, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
            bidirectional=bidirectional,
        )
        multiplier = 2 if bidirectional else 1
        self.fc = nn.Sequential(nn.Linear(hidden * multiplier, HORIZON * 3))

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        out = self.fc(last)
        out = out.reshape(-1, HORIZON, 3)
        return out


# ==========================================================
# TRAIN FUNCTION
# ==========================================================
def train_model(model, train_loader, val_loader, name):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {"train": [], "val": []}
    best_loss = np.inf
    for epoch in range(EPOCHS):
        model.train()
        train_loss = []
        for X, y in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X)
            loss = multi_quantile_loss(pred, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        model.eval()
        val_loss = []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(X)
                loss = multi_quantile_loss(pred, y)
                val_loss.append(loss.item())
        train_mean = np.mean(train_loss)
        val_mean = np.mean(val_loss)
        history["train"].append(train_mean)
        history["val"].append(val_mean)
        print(name, epoch, train_mean, val_mean)
        if val_mean < best_loss:
            best_loss = val_mean
            torch.save(
                {"model_state": model.state_dict(), "val_loss": best_loss},
                os.path.join(MODEL_DIR, name + ".pt"),
            )
    with open(os.path.join(TABLE_DIR, name + "_history.json"), "w") as f:
        json.dump(history, f)
    return best_loss


# ==========================================================
# TRAIN EXPERIMENT
# ==========================================================
def run_experiment(X, y, name):
    split = int(len(X) * 0.8)
    X_train = X[:split]
    X_val = X[split:]
    y_train = y[:split]
    y_val = y[split:]
    train_ds = ForecastDataset(X_train, y_train)
    val_ds = ForecastDataset(X_val, y_val)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True
    )
    input_size = X.shape[2]
    results = []
    for bidir in [False, True]:
        model = QuantileLSTM(
            input_size, hidden=91, layers=1, dropout=0.057, bidirectional=bidir
        )
        # model_name = name + "_BiLSTM" if bidir else "_LSTM"
        model_name = name + "_BiLSTM" if bidir else name + "_LSTM"
        loss = train_model(model, train_loader, val_loader, model_name)
        results.append({"model": model_name, "loss": loss})
    return results


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("Running on:", DEVICE)
    X = np.load(os.path.join(PREVIOUS_OUTPUT, "preprocess", "X.npy"), mmap_mode="r")

    y = np.load(os.path.join(PREVIOUS_OUTPUT, "preprocess", "y.npy"), mmap_mode="r")

    cluster_ids = np.load(
        os.path.join(PREVIOUS_OUTPUT, "preprocess", "cluster_ids.npy")
    )
    all_results = []

    print("Training CLUSTER models")

    clusters = np.unique(cluster_ids)

    for cluster in clusters:

        print("Cluster", cluster)

        mask = cluster_ids == cluster

        Xc = X[mask]
        yc = y[mask]

        result = run_experiment(Xc, yc, f"cluster_{cluster}")

        all_results.extend(result)
    pd.DataFrame(all_results).to_csv(
        os.path.join(TABLE_DIR, "model_comparison.csv"), index=False
    )
    print("DONE")


if __name__ == "__main__":
    main()
