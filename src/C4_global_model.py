import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle

# ==========================================================
# CONFIGURATION
# ==========================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREVIOUS_OUTPUT = "output/C1_forcasting_preprocessing"
OUT_DIR = "output/C4_train_global"

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

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
        return self.X[idx], self.y[idx]


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
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
            bidirectional=bidirectional,
        )

        multiplier = 2 if bidirectional else 1

        self.fc = nn.Linear(hidden * multiplier, HORIZON * 3)

    def forward(self, x):
        out, _ = self.lstm(x)

        # Last timestep
        last = out[:, -1, :]

        out = self.fc(last)

        # [batch, HORIZON, 3]
        # 3 = q10, q50, q90
        out = out.reshape(-1, HORIZON, 3)

        return out


# ==========================================================
# INVERSE-SCALED MAE
# ==========================================================
def inverse_mae(pred, target, mean, std):
    pred_real = pred * std + mean
    target_real = target * std + mean

    return np.mean(np.abs(pred_real - target_real))


# ==========================================================
# TRAINING
# ==========================================================
def train_model(model, train_loader, val_loader, name, target_mean, target_std):
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = {
        "train": [],
        "val": [],
        "lr": [],
        "train_mae": [],
        "val_mae": [],
    }

    best_loss = np.inf
    patience_counter = 0

    for epoch in range(EPOCHS):

        # ==================================================
        # TRAIN
        # ==================================================
        model.train()

        train_losses = []
        train_maes = []

        for X, y in train_loader:

            X = X.to(DEVICE, non_blocking=True)

            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
            pred = model(X)

            # Quantile loss
            loss = multi_quantile_loss(pred, y)

            # q50 prediction
            train_pred_q50 = pred[:, :, 1]

            # MAE in original scale
            train_mae = inverse_mae(
                train_pred_q50.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                target_mean,
                target_std,
            )

            train_maes.append(train_mae)

            # Backpropagation
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_losses.append(loss.item())

        # ==================================================
        # VALIDATION
        # ==================================================
        model.eval()

        val_losses = []
        real_maes = []

        with torch.no_grad():

            for X, y in val_loader:

                X = X.to(DEVICE, non_blocking=True)

                y = y.to(DEVICE, non_blocking=True)

                pred = model(X)

                # Validation quantile loss
                loss = multi_quantile_loss(pred, y)

                val_losses.append(loss.item())

                # q50 prediction
                pred_q50 = pred[:, :, 1]

                # Inverse scaling
                pred_real = pred_q50.cpu().numpy() * target_std + target_mean

                y_real = y.cpu().numpy() * target_std + target_mean

                mae = np.mean(np.abs(pred_real - y_real))

                real_maes.append(mae)

        # ==================================================
        # EPOCH METRICS
        # ==================================================
        train_mean = np.mean(train_losses)
        val_mean = np.mean(val_losses)

        train_mae_real = np.mean(train_maes)
        val_mae_real = np.mean(real_maes)

        # Update scheduler
        scheduler.step(val_mean)

        lr = optimizer.param_groups[0]["lr"]

        history["train"].append(train_mean)

        history["val"].append(val_mean)

        history["lr"].append(lr)

        history["train_mae"].append(train_mae_real)

        history["val_mae"].append(val_mae_real)

        # ==================================================
        # PRINT
        # ==================================================
        print(
            f"{name} | "
            f"epoch {epoch:02d} | "
            f"train_scaled {train_mean:.6f} | "
            f"val_scaled {val_mean:.6f} | "
            f"train_MAE {train_mae_real:.6f} | "
            f"val_MAE {val_mae_real:.6f} | "
            f"lr {lr:.6f}"
        )

        # ==================================================
        # SAVE BEST MODEL
        # ==================================================
        if val_mean < best_loss - MIN_DELTA:

            best_loss = val_mean
            patience_counter = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_loss": best_loss,
                    "epoch": epoch,
                    "target_mean": target_mean,
                    "target_std": target_std,
                },
                os.path.join(MODEL_DIR, name + ".pt"),
            )

        else:
            patience_counter += 1

        # ==================================================
        # EARLY STOPPING
        # ==================================================
        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print(f"Early stopping {name} " f"at epoch {epoch}")

            break

    # ======================================================
    # SAVE HISTORY
    # ======================================================
    with open(os.path.join(TABLE_DIR, name + "_history.json"), "w") as f:

        json.dump(history, f, indent=2)

    return best_loss


# ==========================================================
# GLOBAL EXPERIMENT
# ==========================================================
def run_global_experiment(X, y, target_mean, target_std):
    """
    Train models on the COMPLETE dataset.

    No cluster filtering is performed.
    """

    # ======================================================
    # CHRONOLOGICAL 80/20 SPLIT
    # ======================================================
    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_val = X[split:]

    y_train = y[:split]
    y_val = y[split:]

    print(f"Global dataset: " f"train={len(X_train)}, " f"val={len(X_val)}")

    # ======================================================
    # DATASETS
    # ======================================================
    train_ds = ForecastDataset(X_train, y_train)

    val_ds = ForecastDataset(X_val, y_val)

    # ======================================================
    # DATALOADERS
    # ======================================================
    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=use_pin,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=use_pin,
    )

    # ======================================================
    # INPUT SIZE
    # ======================================================
    input_size = X.shape[2]

    print("Input shape:", X.shape)

    print("Input features:", input_size)

    print("Lookback:", X.shape[1])

    print("Horizon:", y.shape[1])

    # ======================================================
    # RESULTS
    # ======================================================
    results = []

    # ======================================================
    # TRAIN STANDARD LSTM
    # ======================================================
    print("\n========================================")
    print("Training GLOBAL LSTM")
    print("========================================")

    model = QuantileLSTM(
        input_size=input_size,
        hidden=91,
        layers=1,
        dropout=0.057,
        bidirectional=False,
    )

    model_name = "global_LSTM"

    loss = train_model(
        model,
        train_loader,
        val_loader,
        model_name,
        target_mean,
        target_std,
    )

    results.append(
        {
            "model": model_name,
            "loss": loss,
        }
    )

    # ======================================================
    # TRAIN BIDIRECTIONAL LSTM
    # ======================================================
    print("\n========================================")
    print("Training GLOBAL BiLSTM")
    print("========================================")

    model = QuantileLSTM(
        input_size=input_size,
        hidden=91,
        layers=1,
        dropout=0.057,
        bidirectional=True,
    )

    model_name = "global_BiLSTM"

    loss = train_model(
        model,
        train_loader,
        val_loader,
        model_name,
        target_mean,
        target_std,
    )

    results.append(
        {
            "model": model_name,
            "loss": loss,
        }
    )

    return results


# ==========================================================
# MAIN
# ==========================================================
def main():

    print("Running on:", DEVICE)

    # ======================================================
    # LOAD PREPROCESSING OUTPUT
    # ======================================================
    print("\nLoading preprocessing output...")

    print("Path:", PREVIOUS_OUTPUT)

    X = np.load(os.path.join(PREVIOUS_OUTPUT, "preprocess", "X.npy"), mmap_mode="r")

    y = np.load(os.path.join(PREVIOUS_OUTPUT, "preprocess", "y.npy"), mmap_mode="r")

    # ======================================================
    # LOAD SCALER
    # ======================================================
    with open(os.path.join(PREVIOUS_OUTPUT, "preprocess", "scaler.pkl"), "rb") as f:

        scaler = pickle.load(f)

    # ======================================================
    # LOAD FEATURE COLUMNS
    # ======================================================
    with open(
        os.path.join(PREVIOUS_OUTPUT, "preprocess", "feature_columns.pkl"), "rb"
    ) as f:

        feature_columns = pickle.load(f)

    # ======================================================
    # TARGET SCALING PARAMETERS
    # ======================================================
    target_name = "N.PRB.UL.DrbUsed.Avg[%]"

    target_index = feature_columns.index(target_name)

    TARGET_MEAN = scaler.mean_[target_index]
    TARGET_STD = scaler.scale_[target_index]

    print("\n===== DATA INFORMATION =====")

    print("X shape:", X.shape)

    print("y shape:", y.shape)

    print("Number of samples:", len(X))

    print("Number of features:", X.shape[2])

    print("Lookback:", X.shape[1])

    print("Forecast horizon:", y.shape[1])

    print("Target:", target_name)

    print("Target mean:", TARGET_MEAN)

    print("Target std:", TARGET_STD)

    print("============================\n")

    # ======================================================
    # TRAIN GLOBAL MODELS
    # ======================================================
    all_results = run_global_experiment(
        X,
        y,
        TARGET_MEAN,
        TARGET_STD,
    )

    # ======================================================
    # SAVE COMPARISON
    # ======================================================
    results_df = pd.DataFrame(all_results)

    results_df.to_csv(os.path.join(TABLE_DIR, "model_comparison.csv"), index=False)

    print("\n===== FINAL RESULTS =====")
    print(results_df)

    print("\nDONE")


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    main()
