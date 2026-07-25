import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# CONFIGURATION
# ==========================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREVIOUS_OUTPUT = "output/C1_quantile_forecasting"
OUT_DIR = "output/C2_train"
MODEL_DIR = os.path.join(OUT_DIR, "models")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
LOOKBACK = 672
HORIZON = 288
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 1e-5
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


# ==========================================================
# DATASET
# ==========================================================
class ForecastDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(np.asarray(X)).float()
        self.y = torch.from_numpy(np.asarray(y)).float()

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
# MODEL
# ==========================================================
class QuantileLSTM(nn.Module):
    def __init__(self, input_size, hidden, layers, dropout, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
            bidirectional=bidirectional,
        )
        multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden * multiplier, HORIZON * 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        out = self.fc(last)
        out = out.reshape(-1, HORIZON, 3)
        return out


# ==========================================================
# TRAINING
# ==========================================================
def train_model(model, train_loader, val_loader, name):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    history = {"train": [], "val": [], "lr": []}
    best_loss = np.inf
    patience_counter = 0
    for epoch in range(EPOCHS):
        # ----------------------
        # TRAIN
        # ----------------------
        model.train()
        train_losses = []
        for X, y in train_loader:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            pred = model(X)
            loss = multi_quantile_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        # ----------------------
        # VALIDATION
        # ----------------------
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                pred = model(X)
                loss = multi_quantile_loss(pred, y)
                val_losses.append(loss.item())
        train_mean = np.mean(train_losses)
        val_mean = np.mean(val_losses)
        scheduler.step(val_mean)
        lr = optimizer.param_groups[0]["lr"]
        history["train"].append(train_mean)
        history["val"].append(val_mean)
        history["lr"].append(lr)
        print(
            f"{name} | "
            f"epoch {epoch:02d} | "
            f"train {train_mean:.6f} | "
            f"val {val_mean:.6f} | "
            f"lr {lr:.6f}"
        )
        # Save best
        if val_mean < best_loss - MIN_DELTA:
            best_loss = val_mean
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_loss": best_loss,
                    "epoch": epoch,
                },
                os.path.join(MODEL_DIR, name + ".pt"),
            )
        else:
            patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping {name} " f"at epoch {epoch}")
            break
    with open(os.path.join(TABLE_DIR, name + "_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return best_loss


# ==========================================================
# EXPERIMENT
# ==========================================================
def run_experiment(X, y, name):
    split = int(len(X) * 0.8)
    X_train = X[:split]
    X_val = X[split:]
    y_train = y[:split]
    y_val = y[split:]
    print(f"{name}: " f"train={len(X_train)}, " f"val={len(X_val)}")
    train_ds = ForecastDataset(X_train, y_train)
    val_ds = ForecastDataset(X_val, y_val)
    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=use_pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=use_pin
    )
    input_size = X.shape[2]
    results = []
    for bidir in [False, True]:
        model = QuantileLSTM(
            input_size=input_size,
            hidden=91,
            layers=1,
            dropout=0.057,
            bidirectional=bidir,
        )
        model_name = name + "_BiLSTM" if bidir else name + "_LSTM"
        loss = train_model(model, train_loader, val_loader, model_name)
        results.append({"model": model_name, "loss": loss})
    return results


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("Running on:", DEVICE)
    print("Loading preprocessing output...")
    X = np.load(os.path.join(PREVIOUS_OUTPUT, "preprocess", "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(PREVIOUS_OUTPUT, "preprocess", "y.npy"), mmap_mode="r")
    cluster_ids = np.load(
        os.path.join(PREVIOUS_OUTPUT, "preprocess", "cluster_ids.npy")
    )
    # ======================================================
    # CLUSTER CHECK
    # ======================================================
    print("\n===== CLUSTER INFORMATION =====")
    unique, counts = np.unique(cluster_ids, return_counts=True)
    for c, n in zip(unique, counts):
        print(f"Cluster {c}: {n} samples")
    print("Number of clusters:", len(unique))
    print("Total samples:", len(cluster_ids))
    if len(unique) == 1 and unique[0] == -1:
        raise RuntimeError(
            "ERROR: All samples are cluster -1. " "Check cluster_assignments.csv"
        )
    print("==============================\n")
    all_results = []
    clusters = np.unique(cluster_ids)
    for cluster in clusters:
        print("\nTraining cluster:", cluster)
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
