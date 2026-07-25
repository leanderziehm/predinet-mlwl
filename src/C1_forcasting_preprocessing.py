import os
import pickle
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ==========================================================
# CONFIGURATION
# ==========================================================
SEED = 42
DATA_PATH = "UL_PRB_data_set.csv"
CLUSTER_PATH = "output/B_cluster/tables/cluster_assignments.csv"
TARGET = "N.PRB.UL.DrbUsed.Avg[%]"
LOOKBACK = 96
HORIZON = 288
np.random.seed(SEED)
random.seed(SEED)


# ==========================================================
# OUTPUT MANAGEMENT
# ==========================================================
def create_output_directory():
    path = os.path.join("output", "C1_quantile_forecasting")
    folders = [
        "models",
        "models/global",
        "models/clusters",
        "preprocess",
        "predictions",
        "tables",
        "plots",
    ]
    for folder in folders:
        os.makedirs(os.path.join(path, folder), exist_ok=True)
    return path


OUT_DIR = create_output_directory()
LOG_FILE = open(os.path.join(OUT_DIR, "run_log.txt"), "w")


def log(message):
    text = f"[{datetime.now().strftime('%H:%M:%S')}] " f"{message}"
    print(text)
    LOG_FILE.write(text + "\n")
    LOG_FILE.flush()


# ==========================================================
# LOAD DATA
# ==========================================================
def load_dataset():
    log("Loading dataset")
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Short name", "Date"])
    log(f"Dataset size: {df.shape}")
    return df


# ==========================================================
# DATA QUALITY CHECK
# ==========================================================
def validate_data(df):
    log("Running data validation")
    duplicates = df.duplicated(["Short name", "Date"]).sum()
    missing = df.isna().sum().sum()
    log(f"Duplicate rows: {duplicates}")
    log(f"Missing values: {missing}")
    cells = df.groupby("Short name").size()
    log(f"Cells: {len(cells)}")
    log(f"Expected timestamps per cell: {cells.unique()}")
    return df


# ==========================================================
# ADD CLUSTERS
# ==========================================================
def add_clusters(df):
    log("Loading cluster assignments")
    clusters = pd.read_csv(CLUSTER_PATH)
    df = df.merge(clusters, left_on="Short name", right_on="series_id", how="left")
    missing = df["cluster"].isna().sum()
    log(f"Rows without cluster: {missing}")
    df["cluster"] = df["cluster"].astype(int)
    return df


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
def create_features(df):
    log("Creating features")
    # Time features
    df["hour"] = df["Date"].dt.hour
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    # Derived feature
    df["throughput_per_user"] = df["N.ThpVol.UL"] / (
        df["N.User.RRCConn.Active.UL.Avg"] + 1
    )
    # Difference
    df["prb_change"] = df.groupby("Short name")[TARGET].diff()
    # Lag features
    lags = [1, 2, 4, 8, 96, 672]
    for lag in lags:
        df[f"prb_lag_{lag}"] = df.groupby("Short name")[TARGET].shift(lag)
    # Rolling statistics
    windows = [4, 16, 96]
    for w in windows:
        df[f"prb_mean_{w}"] = df.groupby("Short name")[TARGET].transform(
            lambda x: x.rolling(w).mean()
        )
        df[f"prb_std_{w}"] = df.groupby("Short name")[TARGET].transform(
            lambda x: x.rolling(w).std()
        )
    df = df.dropna()
    return df


# ==========================================================
# CREATE SEQUENCES
# ==========================================================
def create_sequences(cell_df, feature_columns):
    values = cell_df[feature_columns].values.astype(np.float32)
    target = cell_df[TARGET].values.astype(np.float32)
    n_samples = len(cell_df) - LOOKBACK - HORIZON
    X = np.empty((n_samples, LOOKBACK, len(feature_columns)), dtype=np.float32)
    y = np.empty((n_samples, HORIZON), dtype=np.float32)
    for i in range(n_samples):
        X[i] = values[i : i + LOOKBACK]
        y[i] = target[i + LOOKBACK : i + LOOKBACK + HORIZON]
    cluster = int(cell_df["cluster"].iloc[0])
    meta = {
        "cell": cell_df["Short name"].iloc[0],
        "cluster": cluster,
        "start": cell_df["Date"].values[LOOKBACK : LOOKBACK + n_samples],
    }
    return X, y, meta, cluster


# ==========================================================
# PREPARE DATASET
# ==========================================================
def prepare_sequences(df):
    log("Preparing sequences")
    feature_columns = [
        TARGET,
        "N.ThpVol.UL",
        "N.User.RRCConn.Active.UL.Avg",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "throughput_per_user",
        "prb_change",
        "prb_lag_1",
        "prb_lag_2",
        "prb_lag_4",
        "prb_lag_8",
        "prb_lag_96",
        "prb_lag_672",
        "prb_mean_4",
        "prb_std_4",
        "prb_mean_16",
        "prb_std_16",
        "prb_mean_96",
        "prb_std_96",
        "cluster",
    ]
    with open(os.path.join(OUT_DIR, "preprocess", "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_columns, f)
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns]).astype(np.float32)
    with open(os.path.join(OUT_DIR, "preprocess", "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    X_all = []
    y_all = []
    cluster_all = []
    metadata = []
    for cell, cell_df in df.groupby("Short name"):
        Xc, yc, meta, cluster = create_sequences(cell_df, feature_columns)
        X_all.append(Xc)
        y_all.append(yc)
        cluster_all.extend([cluster] * len(Xc))
        metadata.append(meta)
    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    cluster_ids = np.array(cluster_all, dtype=np.int32)
    log(f"Total sequences: {len(X)}")
    return (X, y, cluster_ids, metadata)


# ==========================================================
# MAIN
# ==========================================================
def main_preprocessing():
    df = load_dataset()
    validate_data(df)
    df = add_clusters(df)
    df = create_features(df)
    X, y, cluster_ids, metadata = prepare_sequences(df)
    np.save(os.path.join(OUT_DIR, "preprocess", "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "preprocess", "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "preprocess", "cluster_ids.npy"), cluster_ids)
    with open(os.path.join(OUT_DIR, "preprocess", "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    log("Preprocessing completed")
    log(f"Output saved to {OUT_DIR}")


if __name__ == "__main__":
    main_preprocessing()
