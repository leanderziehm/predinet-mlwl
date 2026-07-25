import os
import pickle
import numpy as np
import pandas as pd
from pipeline_common import (
    C1_LOOKBACK,
    DEVICE,
    FORECAST_FEATURE_COLUMNS,
    HORIZON,
    N_NEIGHBORS,
    SERIES_COL,
    TARGET,
    TIME_COL,
    assign_cluster_by_neighbors,
    calculate_metrics,
    create_forecasting_features,
    extract_b_features_from_dataframe,
    inverse_transform_target,
    load_cluster_model,
    set_random_seeds,
    validate_feature_columns,
)

# ==========================================================
# CONFIGURATION
# ==========================================================
C1_DIR = "output/PC_tree"
PREPROCESS_DIR = os.path.join(
    C1_DIR,
    "preprocess",
)
C2_DIR = "output/PD_four"
MODEL_DIR = os.path.join(
    C2_DIR,
    "models",
)
ORIGINAL_DATA = "UL_PRB_data_set.csv"
UNSEEN_DATA = "selected_cells_unseen.csv"
# ----------------------------------------------------------
# B artifacts
# ----------------------------------------------------------
B_DIR = "output/PB_two"
B_MODEL_PATH = os.path.join(
    B_DIR,
    "models",
    "cluster_pipeline.pkl",
)
B_FEATURE_FILE = "output/PA_one/tables/" "cell_forecastability_features.csv"
B_ASSIGNMENTS_FILE = "output/PB_two/tables/" "cluster_assignments.csv"
# ----------------------------------------------------------
# Output
# ----------------------------------------------------------
OUT_DIR = "output/PE_five"
TABLE_DIR = os.path.join(
    OUT_DIR,
    "tables",
)
PREDICTION_DIR = os.path.join(
    OUT_DIR,
    "predictions",
)
CLUSTER_DIR = os.path.join(
    OUT_DIR,
    "cluster_assignment",
)
for directory in [
    TABLE_DIR,
    PREDICTION_DIR,
    CLUSTER_DIR,
]:
    os.makedirs(
        directory,
        exist_ok=True,
    )
# ----------------------------------------------------------
# Original known cells
# ----------------------------------------------------------
KNOWN_CELLS = [
    "cell021",
    "cell198",
    "cell192",
    "cell214",
]


# ==========================================================
# LOAD C1 ARTIFACTS
# ==========================================================
def load_c1_artifacts():
    print("\nLoading C1 artifacts...")
    X = np.load(
        os.path.join(
            PREPROCESS_DIR,
            "X.npy",
        ),
        mmap_mode="r",
    )
    y = np.load(
        os.path.join(
            PREPROCESS_DIR,
            "y.npy",
        ),
        mmap_mode="r",
    )
    cluster_ids = np.load(
        os.path.join(
            PREPROCESS_DIR,
            "cluster_ids.npy",
        )
    )
    with open(
        os.path.join(
            PREPROCESS_DIR,
            "metadata.pkl",
        ),
        "rb",
    ) as f:
        metadata = pickle.load(f)
    with open(
        os.path.join(
            PREPROCESS_DIR,
            "scaler.pkl",
        ),
        "rb",
    ) as f:
        scaler = pickle.load(f)
    with open(
        os.path.join(
            PREPROCESS_DIR,
            "feature_columns.pkl",
        ),
        "rb",
    ) as f:
        feature_columns = pickle.load(f)
    print(
        "X:",
        X.shape,
    )
    print(
        "y:",
        y.shape,
    )
    print(
        "cluster_ids:",
        cluster_ids.shape,
    )
    print(
        "metadata entries:",
        len(metadata),
    )
    print(
        "features:",
        len(feature_columns),
    )
    if feature_columns != FORECAST_FEATURE_COLUMNS:
        raise RuntimeError(
            "C1 feature_columns.pkl does not " "match pipeline_common.py."
        )
    if X.shape[1] != C1_LOOKBACK:
        raise RuntimeError(f"X lookback={X.shape[1]}, " f"expected {C1_LOOKBACK}.")
    if y.shape[1] != HORIZON:
        raise RuntimeError(f"y horizon={y.shape[1]}, " f"expected {HORIZON}.")
    if len(X) != len(y):
        raise RuntimeError("X and y have different lengths.")
    if len(X) != len(cluster_ids):
        raise RuntimeError("X and cluster_ids have " "different lengths.")
    return (
        X,
        y,
        cluster_ids,
        metadata,
        scaler,
        feature_columns,
    )


# ==========================================================
# RECONSTRUCT CELL -> SEQUENCE MAPPING
# ==========================================================
def reconstruct_cell_mapping(
    X,
    y,
    cluster_ids,
    metadata,
):
    mapping = {}
    offset = 0
    for meta in metadata:
        cell = str(meta["cell"])
        cluster = int(meta["cluster"])
        n_sequences = len(meta["start"])
        start_idx = offset
        end_idx = offset + n_sequences
        mapping[cell] = {
            "cluster": cluster,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "X": X[start_idx:end_idx],
            "y": y[start_idx:end_idx],
            "starts": meta["start"],
        }
        actual_clusters = np.unique(cluster_ids[start_idx:end_idx])
        if len(actual_clusters) != 1:
            raise RuntimeError(
                f"{cell}: multiple " f"clusters found: " f"{actual_clusters}"
            )
        if int(actual_clusters[0]) != cluster:
            raise RuntimeError(
                f"{cell}: metadata cluster="
                f"{cluster}, actual="
                f"{actual_clusters[0]}"
            )
        offset = end_idx
    if offset != len(X):
        raise RuntimeError(
            f"Metadata reconstruction "
            f"consumed {offset} sequences "
            f"but X contains {len(X)}."
        )
    print(f"\nReconstructed " f"{len(mapping)} cells " f"from metadata.pkl")
    return mapping


# ==========================================================
# LOAD B REFERENCE
# ==========================================================
def load_b_reference_data():
    print("\nLoading B clustering reference data...")
    if not os.path.exists(B_FEATURE_FILE):
        raise FileNotFoundError(B_FEATURE_FILE)
    if not os.path.exists(B_ASSIGNMENTS_FILE):
        raise FileNotFoundError(B_ASSIGNMENTS_FILE)
    if not os.path.exists(B_MODEL_PATH):
        raise FileNotFoundError(B_MODEL_PATH)
    features = pd.read_csv(B_FEATURE_FILE)
    assignments = pd.read_csv(B_ASSIGNMENTS_FILE)
    if "series_id" not in (features.columns):
        raise RuntimeError("B feature table does not " "contain series_id.")
    if "series_id" not in (assignments.columns):
        raise RuntimeError("B assignment table does not " "contain series_id.")
    if "cluster" not in (assignments.columns):
        raise RuntimeError("B assignment table does not " "contain cluster.")
    reference = features.merge(
        assignments[
            [
                "series_id",
                "cluster",
            ]
        ],
        on="series_id",
        how="inner",
        validate="one_to_one",
    )
    if len(reference) != len(features):
        missing = set(features["series_id"]) - set(assignments["series_id"])
        raise RuntimeError(
            "Not every B feature row has "
            "a cluster assignment.\n"
            f"Missing: {sorted(missing)}"
        )
    with open(
        B_MODEL_PATH,
        "rb",
    ) as f:
        artifact = pickle.load(f)
    required = [
        "scaler",
        "pca",
        "model",
    ]
    missing_artifacts = [key for key in required if key not in artifact]
    if missing_artifacts:
        raise RuntimeError("B artifact is missing: " f"{missing_artifacts}")
    b_scaler = artifact["scaler"]
    b_pca = artifact["pca"]
    b_model = artifact["model"]
    b_feature_columns = [column for column in features.columns if column != "series_id"]
    print(
        "B reference cells:",
        len(reference),
    )
    print(
        "B feature count:",
        len(b_feature_columns),
    )
    print(
        "B PCA dimensions:",
        getattr(
            b_pca,
            "n_components_",
            "unknown",
        ),
    )
    print(
        "Existing clusters:",
        sorted(reference["cluster"].unique()),
    )
    return (
        reference,
        b_feature_columns,
        b_scaler,
        b_pca,
        b_model,
    )


# ==========================================================
# ORIGINAL DATASET EVALUATION
# ==========================================================
def evaluate_original(
    cell_mapping,
    feature_columns,
    scaler,
):
    print("\n" + "=" * 70)
    print("ORIGINAL DATASET EVALUATION")
    print("=" * 70)
    input_size = len(feature_columns)
    results = []
    for cell in KNOWN_CELLS:
        print(f"\nEvaluating {cell}")
        if cell not in cell_mapping:
            print(f"WARNING: {cell} " f"not found in C1 metadata.")
            continue
        info = cell_mapping[cell]
        cluster = info["cluster"]
        X_cell = info["X"]
        y_cell = info["y"]
        if len(X_cell) == 0:
            print(f"WARNING: no sequences " f"for {cell}")
            continue
        # --------------------------------------------------
        # Final sequence
        # --------------------------------------------------
        X_eval = X_cell[-1:]
        y_eval = y_cell[-1:]
        actual_real = inverse_transform_target(
            y_eval[0],
            scaler,
            feature_columns,
        )
        # --------------------------------------------------
        # LSTM + BiLSTM
        # --------------------------------------------------
        for model_type in [
            "LSTM",
            "BiLSTM",
        ]:
            model, model_name = load_cluster_model(
                MODEL_DIR,
                cluster,
                model_type,
                input_size,
                HORIZON,
                DEVICE,
            )
            print(f"Running {model_name}")
            from pipeline_common import predict

            prediction = predict(
                model,
                X_eval,
                DEVICE,
            )[0]
            q10_real = inverse_transform_target(
                prediction[:, 0],
                scaler,
                feature_columns,
            )
            q50_real = inverse_transform_target(
                prediction[:, 1],
                scaler,
                feature_columns,
            )
            q90_real = inverse_transform_target(
                prediction[:, 2],
                scaler,
                feature_columns,
            )
            metrics = calculate_metrics(
                actual_real,
                q10_real,
                q50_real,
                q90_real,
            )
            results.append(
                {
                    "dataset": "original",
                    "cell": cell,
                    "cluster": cluster,
                    "model": model_name,
                    "horizon": HORIZON,
                    **metrics,
                }
            )
            forecast = pd.DataFrame(
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
            forecast.to_csv(
                os.path.join(
                    PREDICTION_DIR,
                    f"original_{cell}_" f"{model_name}.csv",
                ),
                index=False,
            )
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(
            TABLE_DIR,
            "original_cell_results.csv",
        ),
        index=False,
    )
    return results_df


# ==========================================================
# ORIGINAL AGGREGATION
# ==========================================================
def aggregate_original(
    results_df,
):
    if results_df.empty:
        return
    all_cells = pd.DataFrame(
        [
            {
                "dataset": "original",
                "group": "all_cells",
                "number_of_cells": (results_df["cell"].nunique()),
                "MAPE_percent": (results_df["MAPE_percent"].mean()),
                "mean_quantile_interval_range": (
                    results_df["mean_quantile_interval_range"].mean()
                ),
                "coverage_percent": (results_df["coverage_percent"].mean()),
            }
        ]
    )
    all_cells.to_csv(
        os.path.join(
            TABLE_DIR,
            "original_all_cells_average.csv",
        ),
        index=False,
    )
    cluster_avg = (
        results_df.groupby("cluster")
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
    print("\n===== ORIGINAL CELL RESULTS =====")
    print(results_df.to_string(index=False))
    print("\n===== ORIGINAL ALL-CELL AVERAGE =====")
    print(all_cells.to_string(index=False))
    print("\n===== ORIGINAL CLUSTER AVERAGES =====")
    print(cluster_avg.to_string(index=False))


# ==========================================================
# UNSEEN EVALUATION
# ==========================================================
def evaluate_unseen(
    feature_columns,
    scaler,
    b_reference,
    b_feature_columns,
    b_scaler,
    b_pca,
):
    print("\n" + "=" * 70)
    print("UNSEEN DATASET EVALUATION")
    print("=" * 70)
    if not os.path.exists(UNSEEN_DATA):
        raise FileNotFoundError(UNSEEN_DATA)
    unseen_raw = pd.read_csv(UNSEEN_DATA)
    unseen_raw[TIME_COL] = pd.to_datetime(unseen_raw[TIME_COL])
    # ======================================================
    # B FEATURES
    # ======================================================
    print("\nExtracting B forecastability features...")
    unseen_b_features = extract_b_features_from_dataframe(unseen_raw)
    unseen_b_features.to_csv(
        os.path.join(
            CLUSTER_DIR,
            "unseen_b_features.csv",
        ),
        index=False,
    )
    # ======================================================
    # FORECASTING FEATURES
    #
    # THIS IS THE SAME FUNCTION USED BY C1.
    # ======================================================
    unseen = create_forecasting_features(
        unseen_raw,
        dropna=False,
    )
    validate_feature_columns(
        unseen,
        feature_columns,
        "Unseen dataset",
    )
    results = []
    cluster_assignments = []
    for cell, cell_df in unseen.groupby(SERIES_COL):
        cell_df = cell_df.sort_values(TIME_COL).reset_index(drop=True)
        print(f"\nUnseen cell: {cell}")
        # --------------------------------------------------
        # Need enough raw observations
        # --------------------------------------------------
        minimum_required = 672 + HORIZON
        if len(cell_df) < (minimum_required):
            raise RuntimeError(
                f"{cell} has "
                f"{len(cell_df)} rows. "
                f"Need at least "
                f"{minimum_required}."
            )
        # --------------------------------------------------
        # Remove invalid rows exactly like C1
        # --------------------------------------------------
        cell_df = cell_df.dropna(subset=feature_columns).reset_index(drop=True)
        if len(cell_df) < (C1_LOOKBACK + HORIZON):
            raise RuntimeError(
                f"{cell}: insufficient " f"valid rows after " f"feature engineering."
            )
        # ==================================================
        # B CLUSTER ASSIGNMENT
        # ==================================================
        unseen_b_row = unseen_b_features[
            unseen_b_features["series_id"].astype(str) == str(cell)
        ]
        if len(unseen_b_row) != 1:
            raise RuntimeError(
                f"{cell}: expected "
                f"one B feature row, "
                f"found {len(unseen_b_row)}."
            )
        reference_features = b_reference[["series_id"] + b_feature_columns]
        (
            assigned_cluster,
            diagnostics,
            score_df,
        ) = assign_cluster_by_neighbors(
            unseen_features=unseen_b_row,
            reference_features=reference_features,
            feature_columns=b_feature_columns,
            scaler=b_scaler,
            pca=b_pca,
            reference_clusters=(b_reference["cluster"].to_numpy()),
            cell_name=cell,
            n_neighbors=N_NEIGHBORS,
        )
        diagnostics.to_csv(
            os.path.join(
                CLUSTER_DIR,
                f"{cell}_nearest_neighbors.csv",
            ),
            index=False,
        )
        score_df.to_csv(
            os.path.join(
                CLUSTER_DIR,
                f"{cell}_cluster_scores.csv",
            ),
            index=False,
        )
        print(f"{cell} -> cluster " f"{assigned_cluster}")
        cluster_assignments.append(
            {
                "cell": cell,
                "cluster": assigned_cluster,
            }
        )
        # ==================================================
        # FORECASTING INPUT
        # ==================================================
        input_df = cell_df.iloc[-(C1_LOOKBACK + HORIZON) : -HORIZON]
        actual_df = cell_df.iloc[-HORIZON:]
        if len(input_df) != (C1_LOOKBACK):
            raise RuntimeError(
                f"{cell}: expected "
                f"{C1_LOOKBACK} input "
                f"rows, got "
                f"{len(input_df)}."
            )
        if len(actual_df) != (HORIZON):
            raise RuntimeError(
                f"{cell}: expected "
                f"{HORIZON} actual "
                f"rows, got "
                f"{len(actual_df)}."
            )
        # --------------------------------------------------
        # Scale using the SAME C1 scaler
        # --------------------------------------------------
        X_raw = input_df[feature_columns].to_numpy(dtype=np.float32)
        X_scaled = scaler.transform(
            pd.DataFrame(
                X_raw,
                columns=feature_columns,
            )
        ).astype(np.float32)
        X_eval = X_scaled[
            np.newaxis,
            :,
            :,
        ]
        actual_real = actual_df[TARGET].to_numpy(dtype=np.float32)
        input_size = len(feature_columns)
        # ==================================================
        # CLUSTER MODEL
        # ==================================================
        for model_type in [
            "LSTM",
            "BiLSTM",
        ]:
            model, model_name = load_cluster_model(
                MODEL_DIR,
                assigned_cluster,
                model_type,
                input_size,
                HORIZON,
                DEVICE,
            )
            print(f"Running {model_name}")
            from pipeline_common import predict

            prediction = predict(
                model,
                X_eval,
                DEVICE,
            )[0]
            q10_real = inverse_transform_target(
                prediction[:, 0],
                scaler,
                feature_columns,
            )
            q50_real = inverse_transform_target(
                prediction[:, 1],
                scaler,
                feature_columns,
            )
            q90_real = inverse_transform_target(
                prediction[:, 2],
                scaler,
                feature_columns,
            )
            metrics = calculate_metrics(
                actual_real,
                q10_real,
                q50_real,
                q90_real,
            )
            results.append(
                {
                    "dataset": "unseen",
                    "cell": cell,
                    "cluster": assigned_cluster,
                    "model": model_name,
                    "horizon": HORIZON,
                    **metrics,
                }
            )
            forecast = pd.DataFrame(
                {
                    "cell": cell,
                    "cluster": assigned_cluster,
                    "model": model_name,
                    "horizon_step": np.arange(
                        1,
                        HORIZON + 1,
                    ),
                    "timestamp": (actual_df[TIME_COL].values),
                    "actual": actual_real,
                    "q10": q10_real,
                    "q50": q50_real,
                    "q90": q90_real,
                }
            )
            forecast.to_csv(
                os.path.join(
                    PREDICTION_DIR,
                    f"unseen_{cell}_" f"{model_name}.csv",
                ),
                index=False,
            )
    # ======================================================
    # SAVE RESULTS
    # ======================================================
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(
            TABLE_DIR,
            "unseen_cell_results.csv",
        ),
        index=False,
    )
    assignments_df = pd.DataFrame(cluster_assignments)
    assignments_df.to_csv(
        os.path.join(
            TABLE_DIR,
            "unseen_cluster_assignments.csv",
        ),
        index=False,
    )
    # ======================================================
    # AGGREGATION
    # ======================================================
    if not results_df.empty:
        overall = pd.DataFrame(
            [
                {
                    "dataset": "unseen",
                    "group": ("all_unseen_cells"),
                    "number_of_cells": (results_df["cell"].nunique()),
                    "MAPE_percent": (results_df["MAPE_percent"].mean()),
                    "mean_quantile_interval_range": (
                        results_df["mean_quantile_interval_range"].mean()
                    ),
                    "coverage_percent": (results_df["coverage_percent"].mean()),
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
        cluster_avg = (
            results_df.groupby("cluster")
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
            "unseen",
        )
        cluster_avg.to_csv(
            os.path.join(
                TABLE_DIR,
                "unseen_cluster_average.csv",
            ),
            index=False,
        )
        print("\n===== UNSEEN RESULTS =====")
        print(results_df.to_string(index=False))
        print("\n===== UNSEEN ALL-CELL AVERAGE =====")
        print(overall.to_string(index=False))
        print("\n===== UNSEEN CLUSTER AVERAGES =====")
        print(cluster_avg.to_string(index=False))
    return results_df


# ==========================================================
# MAIN
# ==========================================================
def main():
    set_random_seeds()
    print("=" * 70)
    print("C3 EVALUATION")
    print("=" * 70)
    print(
        "Device:",
        DEVICE,
    )
    # ------------------------------------------------------
    # C1
    # ------------------------------------------------------
    (
        X,
        y,
        cluster_ids,
        metadata,
        scaler,
        feature_columns,
    ) = load_c1_artifacts()
    # ------------------------------------------------------
    # Mapping
    # ------------------------------------------------------
    cell_mapping = reconstruct_cell_mapping(
        X,
        y,
        cluster_ids,
        metadata,
    )
    # ------------------------------------------------------
    # B
    # ------------------------------------------------------
    (
        b_reference,
        b_feature_columns,
        b_scaler,
        b_pca,
        b_model,
    ) = load_b_reference_data()
    print(
        "\nB clustering model:",
        type(b_model).__name__,
    )
    print("B model remains frozen.")
    # ------------------------------------------------------
    # Original
    # ------------------------------------------------------
    original_results = evaluate_original(
        cell_mapping,
        feature_columns,
        scaler,
    )
    aggregate_original(original_results)
    # ------------------------------------------------------
    # Unseen
    # ------------------------------------------------------
    unseen_results = evaluate_unseen(
        feature_columns,
        scaler,
        b_reference,
        b_feature_columns,
        b_scaler,
        b_pca,
    )
    # ------------------------------------------------------
    # Combined
    # ------------------------------------------------------
    frames = []
    if not original_results.empty:
        frames.append(original_results)
    if not unseen_results.empty:
        frames.append(unseen_results)
    if frames:
        combined = pd.concat(
            frames,
            ignore_index=True,
        )
        combined.to_csv(
            os.path.join(
                TABLE_DIR,
                "C3_all_results.csv",
            ),
            index=False,
        )
    print("\n" + "=" * 70)
    print("C3 COMPLETE")
    print("=" * 70)
    print(
        "Tables:",
        TABLE_DIR,
    )
    print(
        "Predictions:",
        PREDICTION_DIR,
    )
    print(
        "Cluster diagnostics:",
        CLUSTER_DIR,
    )


if __name__ == "__main__":
    main()
