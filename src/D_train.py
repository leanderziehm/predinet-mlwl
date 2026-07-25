import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pipeline_common import (
    C1_LOOKBACK,
    DEVICE,
    FORECAST_FEATURE_COLUMNS,
    HORIZON,
    MODEL_DROPOUT,
    MODEL_HIDDEN,
    MODEL_LAYERS,
    ForecastDataset,
    QuantileLSTM,
    get_target_scaling,
    multi_quantile_loss,
    set_random_seeds,
)

# ==========================================================
# CONFIGURATION
# ==========================================================
PREVIOUS_OUTPUT = "output/C"
OUT_DIR = "output/D"
MODEL_DIR = os.path.join(
    OUT_DIR,
    "models",
)
TABLE_DIR = os.path.join(
    OUT_DIR,
    "tables",
)
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 1e-5
os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)
os.makedirs(
    TABLE_DIR,
    exist_ok=True,
)


# ==========================================================
# TRAINING METRIC
# ==========================================================
def inverse_mae(
    pred,
    target,
    mean,
    std,
):
    pred_real = pred * std + mean
    target_real = target * std + mean
    return np.mean(np.abs(pred_real - target_real))


# ==========================================================
# TRAINING
# ==========================================================
def train_model(
    model,
    train_loader,
    val_loader,
    name,
    target_mean,
    target_std,
):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    history = {
        "train": [],
        "val": [],
        "lr": [],
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
            X = X.to(
                DEVICE,
                non_blocking=True,
            )
            y = y.to(
                DEVICE,
                non_blocking=True,
            )
            optimizer.zero_grad()
            pred = model(X)
            loss = multi_quantile_loss(
                pred,
                y,
            )
            train_pred_q50 = pred[:, :, 1]
            train_mae = inverse_mae(
                train_pred_q50.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                target_mean,
                target_std,
            )
            train_maes.append(train_mae)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
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
                X = X.to(
                    DEVICE,
                    non_blocking=True,
                )
                y = y.to(
                    DEVICE,
                    non_blocking=True,
                )
                pred = model(X)
                loss = multi_quantile_loss(
                    pred,
                    y,
                )
                val_losses.append(loss.item())
                pred_q50 = pred[:, :, 1]
                pred_real = pred_q50.cpu().numpy() * target_std + target_mean
                y_real = y.cpu().numpy() * target_std + target_mean
                mae = np.mean(np.abs(pred_real - y_real))
                real_maes.append(mae)
        train_mean = np.mean(train_losses)
        val_mean = np.mean(val_losses)
        scheduler.step(val_mean)
        lr = optimizer.param_groups[0]["lr"]
        history["train"].append(train_mean)
        history["val"].append(val_mean)
        history["lr"].append(lr)
        train_mae_real = np.mean(train_maes)
        val_mae_real = np.mean(real_maes)
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
                    "model_state": (model.state_dict()),
                    "val_loss": (best_loss),
                    "epoch": epoch,
                    "model_config": {
                        "input_size": (model.lstm.input_size),
                        "hidden": (model.lstm.hidden_size),
                        "layers": (model.lstm.num_layers),
                        "bidirectional": (model.lstm.bidirectional),
                        "horizon": (model.horizon),
                    },
                },
                os.path.join(
                    MODEL_DIR,
                    name + ".pt",
                ),
            )
        else:
            patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping " f"{name} " f"at epoch {epoch}")
            break
    with open(
        os.path.join(
            TABLE_DIR,
            name + "_history.json",
        ),
        "w",
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
        )
    return best_loss


# ==========================================================
# EXPERIMENT
# ==========================================================
def run_experiment(
    X,
    y,
    name,
    target_mean,
    target_std,
):
    split = int(len(X) * 0.8)
    X_train = X[:split]
    X_val = X[split:]
    y_train = y[:split]
    y_val = y[split:]
    if len(X_train) == 0:
        raise RuntimeError(f"{name}: empty training set.")
    if len(X_val) == 0:
        raise RuntimeError(f"{name}: empty validation set.")
    print(f"{name}: " f"train={len(X_train)}, " f"val={len(X_val)}")
    train_ds = ForecastDataset(
        X_train,
        y_train,
    )
    val_ds = ForecastDataset(
        X_val,
        y_val,
    )
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
    input_size = X.shape[2]
    results = []
    # ------------------------------------------------------
    # LSTM + BiLSTM
    # ------------------------------------------------------
    for bidir in [
        False,
        True,
    ]:
        model = QuantileLSTM(
            input_size=input_size,
            hidden=MODEL_HIDDEN,
            layers=MODEL_LAYERS,
            dropout=MODEL_DROPOUT,
            bidirectional=bidir,
            horizon=HORIZON,
        )
        model_name = name + "_BiLSTM" if bidir else name + "_LSTM"
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
    set_random_seeds()
    print(
        "Running on:",
        DEVICE,
    )
    preprocess_dir = os.path.join(
        PREVIOUS_OUTPUT,
        "preprocess",
    )
    # ------------------------------------------------------
    # Load C1 artifacts
    # ------------------------------------------------------
    X = np.load(
        os.path.join(
            preprocess_dir,
            "X.npy",
        ),
        mmap_mode="r",
    )
    y = np.load(
        os.path.join(
            preprocess_dir,
            "y.npy",
        ),
        mmap_mode="r",
    )
    cluster_ids = np.load(
        os.path.join(
            preprocess_dir,
            "cluster_ids.npy",
        )
    )
    with open(
        os.path.join(
            preprocess_dir,
            "scaler.pkl",
        ),
        "rb",
    ) as f:
        scaler = pickle.load(f)
    with open(
        os.path.join(
            preprocess_dir,
            "feature_columns.pkl",
        ),
        "rb",
    ) as f:
        feature_columns = pickle.load(f)
    # ------------------------------------------------------
    # Validate against shared definition
    # ------------------------------------------------------
    if feature_columns != FORECAST_FEATURE_COLUMNS:
        raise RuntimeError(
            "C1 feature_columns.pkl does not "
            "match pipeline_common.py.\n\n"
            "This means the preprocessing artifacts "
            "were generated by a different feature "
            "definition."
        )
    if X.shape[1] != C1_LOOKBACK:
        raise RuntimeError(
            f"Expected lookback " f"{C1_LOOKBACK}, " f"but X has " f"{X.shape[1]}."
        )
    if y.shape[1] != HORIZON:
        raise RuntimeError(
            f"Expected horizon " f"{HORIZON}, " f"but y has " f"{y.shape[1]}."
        )
    target_mean, target_std = get_target_scaling(
        scaler,
        feature_columns,
    )
    # ------------------------------------------------------
    # Cluster check
    # ------------------------------------------------------
    print("\n===== CLUSTER INFORMATION =====")
    unique, counts = np.unique(
        cluster_ids,
        return_counts=True,
    )
    for c, n in zip(
        unique,
        counts,
    ):
        print(f"Cluster {c}: " f"{n} samples")
    print(
        "Number of clusters:",
        len(unique),
    )
    print(
        "Total samples:",
        len(cluster_ids),
    )
    if len(unique) == 1 and unique[0] == -1:
        raise RuntimeError("All samples are cluster -1.")
    print("==============================\n")
    # ------------------------------------------------------
    # Train each cluster
    # ------------------------------------------------------
    all_results = []
    clusters = np.unique(cluster_ids)
    for cluster in clusters:
        print(
            "\nTraining cluster:",
            cluster,
        )
        mask = cluster_ids == cluster
        Xc = X[mask]
        yc = y[mask]
        result = run_experiment(
            Xc,
            yc,
            f"cluster_{cluster}",
            target_mean,
            target_std,
        )
        all_results.extend(result)
    # ------------------------------------------------------
    # Save comparison
    # ------------------------------------------------------
    pd.DataFrame(all_results).to_csv(
        os.path.join(
            TABLE_DIR,
            "model_comparison.csv",
        ),
        index=False,
    )
    print("DONE")


if __name__ == "__main__":
    main()
