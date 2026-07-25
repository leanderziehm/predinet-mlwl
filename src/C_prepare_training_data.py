import os
import pickle
import random
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pipeline_common import (
    C1_LOOKBACK,
    DATA_PATH,
    FORECAST_FEATURE_COLUMNS,
    HORIZON,
    SERIES_COL,
    TARGET,
    TIME_COL,
    create_forecasting_features,
    create_sequences,
    load_time_series_dataframe,
    set_random_seeds,
)

# ==========================================================
# CONFIGURATION
# ==========================================================
CLUSTER_PATH = "output/B/" "tables/cluster_assignments.csv"
SEED = 42


# ==========================================================
# OUTPUT
# ==========================================================
def create_output_directory():
    path = os.path.join(
        "output",
        "C",
    )
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
        os.makedirs(
            os.path.join(
                path,
                folder,
            ),
            exist_ok=True,
        )
    return path


OUT_DIR = create_output_directory()
LOG_FILE = open(
    os.path.join(
        OUT_DIR,
        "run_log.txt",
    ),
    "w",
)


def log(message):
    text = f"[{datetime.now().strftime('%H:%M:%S')}] " f"{message}"
    print(text)
    LOG_FILE.write(text + "\n")
    LOG_FILE.flush()


# ==========================================================
# DATA QUALITY
# ==========================================================
def validate_data(df):
    log("Running data validation")
    duplicates = df.duplicated(
        [
            SERIES_COL,
            TIME_COL,
        ]
    ).sum()
    missing = df.isna().sum().sum()
    log(f"Duplicate rows: " f"{duplicates}")
    log(f"Missing values: " f"{missing}")
    cells = df.groupby(SERIES_COL).size()
    log(f"Cells: {len(cells)}")
    log("Expected timestamps per cell: " f"{cells.unique()}")
    return df


# ==========================================================
# ADD CLUSTERS
# ==========================================================
def add_clusters(df):
    log("Loading cluster assignments")
    if not os.path.exists(CLUSTER_PATH):
        raise FileNotFoundError(f"Cluster assignments not found: " f"{CLUSTER_PATH}")
    clusters = pd.read_csv(CLUSTER_PATH)
    required = [
        "series_id",
        "cluster",
    ]
    missing = [c for c in required if c not in clusters.columns]
    if missing:
        raise RuntimeError(f"Cluster assignment file " f"is missing: {missing}")
    print("\n========== B_CLUSTER FILE ==========")
    print(clusters.head())
    print("\nCluster distribution:")
    print(clusters["cluster"].value_counts().sort_index())
    overlap = set(df[SERIES_COL]).intersection(set(clusters["series_id"]))
    print("\nID overlap:")
    print(
        len(overlap),
        "cells",
    )
    df = df.merge(
        clusters,
        left_on=SERIES_COL,
        right_on="series_id",
        how="left",
    )
    missing_rows = df["cluster"].isna().sum()
    log(f"Rows without cluster: " f"{missing_rows}")
    if missing_rows > 0:
        print("\nCells without clusters:")
        print(
            df.loc[
                df["cluster"].isna(),
                SERIES_COL,
            ].unique()[:20]
        )
        raise RuntimeError("Cluster merge failed. " "Missing cluster assignments.")
    df["cluster"] = df["cluster"].astype(int)
    return df


# ==========================================================
# PREPARE SEQUENCES
# ==========================================================
def prepare_sequences(df):
    log("Preparing sequences")
    feature_columns = FORECAST_FEATURE_COLUMNS
    print("\nFeature columns:")
    print(feature_columns)
    # ------------------------------------------------------
    # Fit scaler on forecasting features
    # ------------------------------------------------------
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns]).astype(np.float32)
    # ------------------------------------------------------
    # Save shared feature definition
    # ------------------------------------------------------
    with open(
        os.path.join(
            OUT_DIR,
            "preprocess",
            "feature_columns.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(
            feature_columns,
            f,
        )
    # ------------------------------------------------------
    # Save scaler
    # ------------------------------------------------------
    with open(
        os.path.join(
            OUT_DIR,
            "preprocess",
            "scaler.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(
            scaler,
            f,
        )
    # ------------------------------------------------------
    # Create sequences
    # ------------------------------------------------------
    X_all = []
    y_all = []
    cluster_all = []
    metadata = []
    cluster_cell_counter = {}
    for cell, cell_df in df.groupby(SERIES_COL):
        cell_df = cell_df.sort_values(TIME_COL).reset_index(drop=True)
        cluster_values = cell_df["cluster"].unique()
        if len(cluster_values) != 1:
            raise RuntimeError(
                f"{cell}: multiple " f"clusters found: " f"{cluster_values}"
            )
        cluster = int(cluster_values[0])
        cluster_cell_counter[cluster] = (
            cluster_cell_counter.get(
                cluster,
                0,
            )
            + 1
        )
        Xc, yc, meta, _ = create_sequences(
            cell_df,
            feature_columns=feature_columns,
            lookback=C1_LOOKBACK,
            horizon=HORIZON,
        )
        print(
            f"Cell={cell} | "
            f"cluster={cluster} | "
            f"rows={len(cell_df)} | "
            f"sequences={len(Xc)}"
        )
        if cluster < 0:
            raise RuntimeError(f"Negative cluster found " f"for {cell}")
        if len(Xc) == 0:
            print(f"WARNING: {cell} " f"has no usable sequences.")
            continue
        X_all.append(Xc)
        y_all.append(yc)
        cluster_all.extend([cluster] * len(Xc))
        metadata.append(meta)
    if not X_all:
        raise RuntimeError("No forecasting sequences " "were created.")
    X = np.concatenate(
        X_all,
        axis=0,
    )
    y = np.concatenate(
        y_all,
        axis=0,
    )
    cluster_ids = np.asarray(
        cluster_all,
        dtype=np.int32,
    )
    print("\nCluster cell counts:")
    print(cluster_cell_counter)
    print(
        "\nX shape:",
        X.shape,
    )
    print(
        "y shape:",
        y.shape,
    )
    print(
        "cluster_ids shape:",
        cluster_ids.shape,
    )
    print("\nFinal cluster distribution:")
    print(pd.Series(cluster_ids).value_counts().sort_index())
    if -1 in cluster_ids:
        raise RuntimeError("cluster_ids contains -1")
    return (
        X,
        y,
        cluster_ids,
        metadata,
    )


# ==========================================================
# MAIN
# ==========================================================
def main_preprocessing():
    set_random_seeds(SEED)
    # ------------------------------------------------------
    # Load
    # ------------------------------------------------------
    log("Loading dataset")
    df = load_time_series_dataframe(DATA_PATH)
    log(f"Dataset size: " f"{df.shape}")
    # ------------------------------------------------------
    # Validate
    # ------------------------------------------------------
    df = validate_data(df)
    # ------------------------------------------------------
    # Add B clusters
    # ------------------------------------------------------
    df = add_clusters(df)
    # ------------------------------------------------------
    # SHARED forecasting feature function
    # ------------------------------------------------------
    log("Creating forecasting features")
    df = create_forecasting_features(
        df,
        dropna=True,
    )
    # ------------------------------------------------------
    # Create sequences
    # ------------------------------------------------------
    (
        X,
        y,
        cluster_ids,
        metadata,
    ) = prepare_sequences(df)
    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------
    preprocess_dir = os.path.join(
        OUT_DIR,
        "preprocess",
    )
    np.save(
        os.path.join(
            preprocess_dir,
            "X.npy",
        ),
        X,
    )
    np.save(
        os.path.join(
            preprocess_dir,
            "y.npy",
        ),
        y,
    )
    np.save(
        os.path.join(
            preprocess_dir,
            "cluster_ids.npy",
        ),
        cluster_ids,
    )
    with open(
        os.path.join(
            preprocess_dir,
            "metadata.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(
            metadata,
            f,
        )
    log("Preprocessing completed")
    log(f"Output saved to " f"{OUT_DIR}")
    LOG_FILE.close()


if __name__ == "__main__":
    main_preprocessing()
