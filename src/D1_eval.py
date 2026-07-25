import os
import json
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# ==========================================================
# CONFIGURATION
# ==========================================================

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREVIOUS_OUTPUT = "output/C1_forcasting_preprocessing"
C2_OUTPUT = "output/C2_train"

MODEL_DIR = os.path.join(C2_OUTPUT, "models")

OUT_DIR = "output/C3_evaluation"
TABLE_DIR = os.path.join(OUT_DIR, "tables")
PREDICTION_DIR = os.path.join(OUT_DIR, "predictions")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)

LOOKBACK = 672
HORIZON = 288

TARGET_COLUMN = "N.PRB.UL.DrbUsed.Avg[%]"

# Cells explicitly requested for evaluation on original dataset
KNOWN_EVAL_CELLS = [
    "cell021",
    "cell198",
    "cell192",
    "cell214",
]

ORIGINAL_DATA_FILE = "UL_PRB_data_set.csv"
UNSEEN_DATA_FILE = "selected_cells_unseen.csv"


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================================
# MODEL
# ==========================================================


class QuantileLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden,
        layers,
        dropout,
        bidirectional=False,
        horizon=288,
    ):
        super().__init__()

        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
            bidirectional=bidirectional,
        )

        multiplier = 2 if bidirectional else 1

        self.fc = nn.Linear(
            hidden * multiplier,
            horizon * 3,
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        last = out[:, -1, :]

        out = self.fc(last)

        out = out.reshape(
            -1,
            self.horizon,
            3,
        )

        return out


# ==========================================================
# HELPERS
# ==========================================================


def find_cell_column(df):
    """
    Try to identify the cell identifier column.
    """

    candidates = [
        "cell",
        "Cell",
        "CELL",
        "cell_id",
        "Cell_ID",
        "cellId",
        "CellId",
        "site",
        "Site",
        "name",
        "Name",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # Fallback: inspect columns containing 'cell'
    for col in df.columns:
        if "cell" in col.lower():
            return col

    raise ValueError(
        "Could not identify cell column. " f"Available columns: {list(df.columns)}"
    )


def find_time_column(df):
    """
    Try to identify the timestamp column.
    """

    candidates = [
        "timestamp",
        "Timestamp",
        "time",
        "Time",
        "datetime",
        "Datetime",
        "date",
        "Date",
        "TimeStamp",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # Try columns that contain likely time-related names
    for col in df.columns:
        low = col.lower()

        if any(key in low for key in ["timestamp", "datetime", "time", "date"]):
            return col

    return None


def sort_dataframe(df):
    """
    Sort by cell and timestamp when possible.
    """

    cell_col = find_cell_column(df)
    time_col = find_time_column(df)

    if time_col is not None:

        try:
            df = df.copy()

            df[time_col] = pd.to_datetime(
                df[time_col],
                errors="coerce",
            )

            df = df.sort_values([cell_col, time_col])

        except Exception:
            df = df.sort_values([cell_col])

    else:
        df = df.sort_values([cell_col])

    return df


def get_scaler_info():

    scaler_path = os.path.join(
        PREVIOUS_OUTPUT,
        "preprocess",
        "scaler.pkl",
    )

    feature_path = os.path.join(
        PREVIOUS_OUTPUT,
        "preprocess",
        "feature_columns.pkl",
    )

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(feature_path, "rb") as f:
        feature_columns = pickle.load(f)

    if TARGET_COLUMN not in feature_columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' " "was not found in feature_columns.pkl"
        )

    target_index = feature_columns.index(TARGET_COLUMN)

    target_mean = scaler.mean_[target_index]
    target_std = scaler.scale_[target_index]

    return (
        scaler,
        feature_columns,
        target_index,
        target_mean,
        target_std,
    )


def inverse_target(values, mean, std):

    return values * std + mean


# ==========================================================
# METRICS
# ==========================================================


def calculate_metrics(
    actual,
    q10,
    q50,
    q90,
    epsilon=1e-8,
):
    """
    Calculate evaluation metrics.

    MAPE:
        Mean absolute percentage error using q50.

    Interval range:
        Mean q90 - q10.

    Coverage:
        Percentage of actual values inside [q10, q90].
    """

    actual = np.asarray(actual, dtype=float)
    q10 = np.asarray(q10, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q90 = np.asarray(q90, dtype=float)

    # ------------------------------------------------------
    # MAPE
    # ------------------------------------------------------

    denominator = np.maximum(
        np.abs(actual),
        epsilon,
    )

    mape = np.mean(np.abs(actual - q50) / denominator) * 100.0

    # ------------------------------------------------------
    # Quantile interval range
    # ------------------------------------------------------

    interval_range = np.mean(q90 - q10)

    # ------------------------------------------------------
    # Coverage
    # ------------------------------------------------------

    covered = (actual >= q10) & (actual <= q90)

    coverage = np.mean(covered) * 100.0

    return {
        "MAPE_percent": mape,
        "mean_quantile_interval_range": interval_range,
        "coverage_percent": coverage,
    }


# ==========================================================
# MODEL LOADING
# ==========================================================


def load_model(model_path, input_size):

    # Names in C2 were trained with:
    #
    # hidden = 91
    # layers = 1
    # dropout = 0.057

    is_bilstm = "_BiLSTM" in os.path.basename(model_path)

    model = QuantileLSTM(
        input_size=input_size,
        hidden=91,
        layers=1,
        dropout=0.057,
        bidirectional=is_bilstm,
        horizon=HORIZON,
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state"])

    model.to(DEVICE)
    model.eval()

    return model


# ==========================================================
# PREDICT
# ==========================================================


def predict(model, X):

    X_tensor = torch.from_numpy(np.asarray(X)).float()

    X_tensor = X_tensor.to(DEVICE)

    with torch.no_grad():

        prediction = model(X_tensor)

    return prediction.cpu().numpy()


# ==========================================================
# CREATE ONE FORECAST WINDOW
# ==========================================================


def create_last_window(
    X_cell,
):
    """
    X_cell must contain the chronological
    feature sequence for one cell.

    Last 672 timestamps are used as input.
    """

    if len(X_cell) < LOOKBACK:

        raise ValueError(
            f"Not enough observations. " f"Need {LOOKBACK}, got {len(X_cell)}."
        )

    return X_cell[-LOOKBACK:]


# ==========================================================
# ORIGINAL DATASET EVALUATION
# ==========================================================


def evaluate_original_dataset():

    print("\n==============================================")
    print("EVALUATION ON ORIGINAL DATASET")
    print("==============================================")

    X_path = os.path.join(
        PREVIOUS_OUTPUT,
        "preprocess",
        "X.npy",
    )

    y_path = os.path.join(
        PREVIOUS_OUTPUT,
        "preprocess",
        "y.npy",
    )

    cluster_path = os.path.join(
        PREVIOUS_OUTPUT,
        "preprocess",
        "cluster_ids.npy",
    )

    X = np.load(
        X_path,
        mmap_mode="r",
    )

    y = np.load(
        y_path,
        mmap_mode="r",
    )

    cluster_ids = np.load(cluster_path)

    (
        scaler,
        feature_columns,
        target_index,
        target_mean,
        target_std,
    ) = get_scaler_info()

    input_size = X.shape[2]

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Input features:", input_size)

    # ------------------------------------------------------
    # Load original CSV
    # ------------------------------------------------------

    original = pd.read_csv(ORIGINAL_DATA_FILE)

    original = sort_dataframe(original)

    cell_col = find_cell_column(original)

    print(
        "Cell column:",
        cell_col,
    )

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    cell_results = []
    cluster_results = []

    # ------------------------------------------------------
    # Evaluate requested cells
    # ------------------------------------------------------

    for cell in KNOWN_EVAL_CELLS:

        print(f"\nEvaluating original cell: {cell}")

        # Find all training windows associated
        # with this cell from the original raw data.
        #
        # IMPORTANT:
        # This requires the preprocessing stage to preserve
        # the relationship between X/y samples and cells.
        #
        # Therefore we identify windows through the target
        # values whenever possible.

        cell_rows = original[original[cell_col].astype(str) == str(cell)]

        if len(cell_rows) == 0:

            print(f"WARNING: {cell} not found " "in original CSV.")

            continue

        # --------------------------------------------------
        # Select windows from X/y.
        #
        # Because X/y already contain the forecasting
        # windows, cluster_ids alone do not identify cells.
        #
        # For a robust cell-level evaluation, C1 should
        # ideally have saved cell_ids.npy.
        # --------------------------------------------------

        cell_ids_path = os.path.join(
            PREVIOUS_OUTPUT,
            "preprocess",
            "cell_ids.npy",
        )

        if not os.path.exists(cell_ids_path):

            raise FileNotFoundError(
                "\ncell_ids.npy was not found in "
                "C1 preprocessing output.\n\n"
                "Your C2 pipeline currently saves "
                "cluster_ids.npy but not cell_ids.npy. "
                "For the requested cell-level evaluation "
                "you need to save the cell ID belonging "
                "to every X/y forecasting window in C1.\n\n"
                "Please add cell_ids.npy to C1 and rerun "
                "preprocessing/training if necessary."
            )

        cell_ids = np.load(
            cell_ids_path,
            allow_pickle=True,
        )

        mask = cell_ids.astype(str) == str(cell)

        X_cell = X[mask]
        y_cell = y[mask]
        cluster_cell = cluster_ids[mask]

        if len(X_cell) == 0:

            print(f"WARNING: No forecasting windows " f"found for {cell}")

            continue

        # --------------------------------------------------
        # The last forecasting window corresponds to the
        # final 288 timestamps available for evaluation.
        # --------------------------------------------------

        X_eval = X_cell[-1:]
        y_eval = y_cell[-1:]

        # Determine cluster from the window
        cluster = cluster_cell[-1]

        print(f"Cell {cell}: " f"windows={len(X_cell)}, " f"cluster={cluster}")

        # --------------------------------------------------
        # Load corresponding models
        # --------------------------------------------------

        model_names = [
            f"cluster_{cluster}_LSTM",
            f"cluster_{cluster}_BiLSTM",
        ]

        for model_name in model_names:

            model_path = os.path.join(
                MODEL_DIR,
                model_name + ".pt",
            )

            if not os.path.exists(model_path):

                print(f"WARNING: Missing model " f"{model_path}")

                continue

            model = load_model(
                model_path,
                input_size,
            )

            prediction = predict(
                model,
                X_eval,
            )

            prediction = prediction[0]

            q10 = prediction[:, 0]
            q50 = prediction[:, 1]
            q90 = prediction[:, 2]

            actual = y_eval[0]

            # --------------------------------------------------
            # Convert target from scaled to real units
            # --------------------------------------------------

            actual_real = inverse_target(
                actual,
                target_mean,
                target_std,
            )

            q10_real = inverse_target(
                q10,
                target_mean,
                target_std,
            )

            q50_real = inverse_target(
                q50,
                target_mean,
                target_std,
            )

            q90_real = inverse_target(
                q90,
                target_mean,
                target_std,
            )

            metrics = calculate_metrics(
                actual_real,
                q10_real,
                q50_real,
                q90_real,
            )

            result = {
                "dataset": "original",
                "cell": cell,
                "cluster": int(cluster),
                "model": model_name,
                "horizon": HORIZON,
                **metrics,
            }

            cell_results.append(result)

            # --------------------------------------------------
            # Save detailed forecast
            # --------------------------------------------------

            forecast_df = pd.DataFrame(
                {
                    "cell": cell,
                    "cluster": cluster,
                    "model": model_name,
                    "horizon_step": np.arange(
                        1,
                        HORIZON + 1,
                    ),
                    "actual": actual_real,
                    "q10": q10_real,
                    "q50": q50_real,
                    "q90": q90_real,
                }
            )

            forecast_path = os.path.join(
                PREDICTION_DIR,
                f"{cell}_{model_name}_forecast.csv",
            )

            forecast_df.to_csv(
                forecast_path,
                index=False,
            )

    # ------------------------------------------------------
    # Cell results
    # ------------------------------------------------------

    cell_results_df = pd.DataFrame(cell_results)

    cell_results_df.to_csv(
        os.path.join(
            TABLE_DIR,
            "original_cell_results.csv",
        ),
        index=False,
    )

    # ------------------------------------------------------
    # Overall average across cells
    # ------------------------------------------------------

    if not cell_results_df.empty:

        overall = pd.DataFrame(
            [
                {
                    "dataset": "original",
                    "group": "all_cells",
                    "number_of_cells": cell_results_df["cell"].nunique(),
                    "MAPE_percent": cell_results_df["MAPE_percent"].mean(),
                    "mean_quantile_interval_range": cell_results_df[
                        "mean_quantile_interval_range"
                    ].mean(),
                    "coverage_percent": cell_results_df["coverage_percent"].mean(),
                }
            ]
        )

        overall.to_csv(
            os.path.join(
                TABLE_DIR,
                "original_all_cells_average.csv",
            ),
            index=False,
        )

        # --------------------------------------------------
        # Average per cluster
        # --------------------------------------------------

        cluster_avg = (
            cell_results_df.groupby("cluster")
            .agg(
                number_of_cells=(
                    "cell",
                    "nunique",
                ),
                MAPE_percent=(
                    "MAPE_percent",
                    "mean",
                ),
                mean_quantile_interval_range=(
                    "mean_quantile_interval_range",
                    "mean",
                ),
                coverage_percent=(
                    "coverage_percent",
                    "mean",
                ),
            )
            .reset_index()
        )

        cluster_avg.insert(
            0,
            "dataset",
            "original",
        )

        cluster_avg.to_csv(
            os.path.join(
                TABLE_DIR,
                "original_cluster_average.csv",
            ),
            index=False,
        )

        print("\n===== ORIGINAL DATASET RESULTS =====")

        print(cell_results_df.to_string(index=False))

        print("\n===== ALL CELLS AVERAGE =====")

        print(overall.to_string(index=False))

        print("\n===== CLUSTER AVERAGES =====")

        print(cluster_avg.to_string(index=False))

    return cell_results_df


# ==========================================================
# UNSEEN DATASET
# ==========================================================


def evaluate_unseen_dataset():

    print("\n==============================================")
    print("EVALUATION ON UNSEEN DATASET")
    print("==============================================")

    unseen = pd.read_csv(UNSEEN_DATA_FILE)

    unseen = sort_dataframe(unseen)

    cell_col = find_cell_column(unseen)

    print(
        "Cell column:",
        cell_col,
    )

    print("Unseen cells:", sorted(unseen[cell_col].astype(str).unique().tolist()))

    # ------------------------------------------------------
    # Load C1 preprocessing artifacts
    # ------------------------------------------------------

    (
        scaler,
        feature_columns,
        target_index,
        target_mean,
        target_std,
    ) = get_scaler_info()

    input_size = len(feature_columns)

    # ------------------------------------------------------
    # IMPORTANT
    # ------------------------------------------------------
    #
    # The unseen dataset must be transformed using the
    # SAME scaler used during training.
    #
    # No fitting is performed here.
    # ------------------------------------------------------

    missing = [c for c in feature_columns if c not in unseen.columns]

    if missing:

        raise ValueError(
            "The unseen dataset does not contain "
            "all required features.\n\n"
            f"Missing columns:\n{missing}"
        )

    results = []

    # ------------------------------------------------------
    # Evaluate each unseen cell
    # ------------------------------------------------------

    for cell in sorted(unseen[cell_col].astype(str).unique()):

        print(f"\nEvaluating unseen cell: {cell}")

        cell_df = unseen[unseen[cell_col].astype(str) == str(cell)].copy()

        time_col = find_time_column(cell_df)

        if time_col is not None:

            cell_df = cell_df.sort_values(time_col)

        if len(cell_df) < (LOOKBACK + HORIZON):

            raise ValueError(
                f"{cell} has only "
                f"{len(cell_df)} rows. "
                f"Need at least "
                f"{LOOKBACK + HORIZON}."
            )

        # --------------------------------------------------
        # Last 288 are the actual evaluation horizon.
        #
        # The 672 rows immediately preceding them are the
        # input sequence.
        # --------------------------------------------------

        input_df = cell_df.iloc[-(LOOKBACK + HORIZON) : -HORIZON]

        actual_df = cell_df.iloc[-HORIZON:]

        # --------------------------------------------------
        # Transform using TRAINING scaler
        # --------------------------------------------------

        X_input_raw = input_df[feature_columns].to_numpy(dtype=np.float32)

        actual_raw = actual_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

        X_input_scaled = scaler.transform(X_input_raw).astype(np.float32)

        actual_scaled = (actual_raw - target_mean) / target_std

        X_eval = np.expand_dims(
            X_input_scaled,
            axis=0,
        )

        # --------------------------------------------------
        # Determine which trained model to use
        # --------------------------------------------------
        #
        # We do NOT retrain.
        #
        # Since the unseen cell has no cluster assignment,
        # we determine its closest cluster based on the
        # final input window's feature profile.
        #
        # A stronger approach is to reuse the exact C1
        # clustering model/artifact if it was saved.
        # --------------------------------------------------

        cluster = determine_unseen_cluster(X_input_scaled)

        print(f"Assigned cluster: {cluster}")

        model_names = [
            f"cluster_{cluster}_LSTM",
            f"cluster_{cluster}_BiLSTM",
        ]

        for model_name in model_names:

            model_path = os.path.join(
                MODEL_DIR,
                model_name + ".pt",
            )

            if not os.path.exists(model_path):

                print(f"WARNING: Model not found: " f"{model_path}")

                continue

            model = load_model(
                model_path,
                input_size,
            )

            prediction = predict(
                model,
                X_eval,
            )[0]

            q10 = prediction[:, 0]
            q50 = prediction[:, 1]
            q90 = prediction[:, 2]

            # --------------------------------------------------
            # Inverse scale predictions
            # --------------------------------------------------

            q10_real = inverse_target(
                q10,
                target_mean,
                target_std,
            )

            q50_real = inverse_target(
                q50,
                target_mean,
                target_std,
            )

            q90_real = inverse_target(
                q90,
                target_mean,
                target_std,
            )

            metrics = calculate_metrics(
                actual_raw,
                q10_real,
                q50_real,
                q90_real,
            )

            result = {
                "dataset": "unseen",
                "cell": cell,
                "cluster": int(cluster),
                "model": model_name,
                "horizon": HORIZON,
                **metrics,
            }

            results.append(result)

            # --------------------------------------------------
            # Save forecast
            # --------------------------------------------------

            forecast_df = pd.DataFrame(
                {
                    "cell": cell,
                    "cluster": cluster,
                    "model": model_name,
                    "horizon_step": np.arange(
                        1,
                        HORIZON + 1,
                    ),
                    "actual": actual_raw,
                    "q10": q10_real,
                    "q50": q50_real,
                    "q90": q90_real,
                }
            )

            if time_col is not None:

                forecast_df.insert(
                    1,
                    "timestamp",
                    actual_df[time_col].values,
                )

            forecast_path = os.path.join(
                PREDICTION_DIR,
                f"unseen_{cell}_{model_name}_forecast.csv",
            )

            forecast_df.to_csv(
                forecast_path,
                index=False,
            )

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        os.path.join(
            TABLE_DIR,
            "unseen_cell_results.csv",
        ),
        index=False,
    )

    # ------------------------------------------------------
    # Average across all 8 unseen cells
    # ------------------------------------------------------

    if not results_df.empty:

        overall = pd.DataFrame(
            [
                {
                    "dataset": "unseen",
                    "group": "all_unseen_cells",
                    "number_of_cells": results_df["cell"].nunique(),
                    "MAPE_percent": results_df["MAPE_percent"].mean(),
                    "mean_quantile_interval_range": results_df[
                        "mean_quantile_interval_range"
                    ].mean(),
                    "coverage_percent": results_df["coverage_percent"].mean(),
                }
            ]
        )

        overall.to_csv(
            os.path.join(
                TABLE_DIR,
                "unseen_all_cells_average.csv",
            ),
            index=False,
        )

        print("\n===== UNSEEN CELL RESULTS =====")

        print(results_df.to_string(index=False))

        print("\n===== UNSEEN CELLS AVERAGE =====")

        print(overall.to_string(index=False))

    return results_df


# ==========================================================
# CLUSTER ASSIGNMENT FOR UNSEEN DATA
# ==========================================================


def determine_unseen_cluster(X_input_scaled):
    """
    Assign an unseen cell to one of the existing clusters.

    IMPORTANT:
    The preferred solution is to save the clustering model
    from C1 and reuse it here.

    This function first tries to load such an artifact.

    If no clustering model is available, it stops instead
    of silently inventing a cluster assignment.
    """

    possible_files = [
        os.path.join(
            PREVIOUS_OUTPUT,
            "preprocess",
            "cluster_model.pkl",
        ),
        os.path.join(
            PREVIOUS_OUTPUT,
            "models",
            "cluster_model.pkl",
        ),
        os.path.join(
            PREVIOUS_OUTPUT,
            "cluster_model.pkl",
        ),
    ]

    cluster_model_path = None

    for path in possible_files:

        if os.path.exists(path):

            cluster_model_path = path
            break

    if cluster_model_path is None:

        raise FileNotFoundError(
            "\nCannot assign unseen cell to a trained "
            "cluster because the C1 clustering model "
            "was not saved.\n\n"
            "Your C2 output contains cluster_ids.npy, "
            "but this only tells us the cluster of the "
            "training windows. It cannot assign a new "
            "unseen cell to a cluster.\n\n"
            "Save the fitted clustering object in C1, "
            "for example as:\n"
            "output/C1_forcasting_preprocessing/"
            "preprocess/cluster_model.pkl\n\n"
            "Then this C3 script can reuse it without "
            "retraining the forecasting model."
        )

    with open(
        cluster_model_path,
        "rb",
    ) as f:

        cluster_model = pickle.load(f)

    # ------------------------------------------------------
    # Represent the unseen cell by the mean feature vector
    # over the input window.
    #
    # If your C1 clustering used another representation,
    # replace this section with exactly that representation.
    # ------------------------------------------------------

    representation = np.mean(
        X_input_scaled,
        axis=0,
    )

    representation = representation.reshape(
        1,
        -1,
    )

    cluster = cluster_model.predict(representation)[0]

    return int(cluster)


# ==========================================================
# MAIN
# ==========================================================


def main():

    print("==============================================")
    print("C3 FORECAST EVALUATION")
    print("==============================================")
    print("Device:", DEVICE)
    print("Lookback:", LOOKBACK)
    print("Horizon:", HORIZON)

    # ------------------------------------------------------
    # Original dataset
    # ------------------------------------------------------

    original_results = evaluate_original_dataset()

    # ------------------------------------------------------
    # Unseen dataset
    # ------------------------------------------------------

    unseen_results = evaluate_unseen_dataset()

    # ------------------------------------------------------
    # Combined report
    # ------------------------------------------------------

    combined = pd.concat(
        [
            original_results,
            unseen_results,
        ],
        ignore_index=True,
    )

    combined.to_csv(
        os.path.join(
            TABLE_DIR,
            "C3_all_results.csv",
        ),
        index=False,
    )

    print("\n==============================================")
    print("C3 EVALUATION COMPLETE")
    print("==============================================")

    print(
        "Reports saved to:",
        TABLE_DIR,
    )

    print(
        "Forecasts saved to:",
        PREDICTION_DIR,
    )


if __name__ == "__main__":
    main()
