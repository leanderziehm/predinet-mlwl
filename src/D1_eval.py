import os
import pickle
import random
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import entropy
from scipy.signal import periodogram
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.neighbors import NearestNeighbors

# ==========================================================
# CONFIGURATION
# ==========================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------------
# C1 artifacts
# ----------------------------------------------------------
C1_DIR = "output/C1_forcasting_preprocessing"
PREPROCESS_DIR = os.path.join(
    C1_DIR,
    "preprocess",
)
# ----------------------------------------------------------
# C2 artifacts
# ----------------------------------------------------------
C2_DIR = "degs/C2_train"
MODEL_DIR = os.path.join(
    C2_DIR,
    "models",
)
# ----------------------------------------------------------
# Input datasets
# ----------------------------------------------------------
ORIGINAL_DATA = "UL_PRB_data_set.csv"
UNSEEN_DATA = "selected_cells_unseen.csv"
# ----------------------------------------------------------
# B clustering artifacts
# ----------------------------------------------------------
B_DIR = "output/B_cluster"
B_MODEL_PATH = os.path.join(
    B_DIR,
    "models",
    "cluster_pipeline.pkl",
)
B_FEATURE_FILE = os.path.join(
    "output",
    "A_stl",
    "tables",
    "cell_forecastability_features.csv",
)
B_ASSIGNMENTS_FILE = os.path.join(
    B_DIR,
    "tables",
    "cluster_assignments.csv",
)
# ----------------------------------------------------------
# Output
# ----------------------------------------------------------
OUT_DIR = "output/C3_evaluation"
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
os.makedirs(
    TABLE_DIR,
    exist_ok=True,
)
os.makedirs(
    PREDICTION_DIR,
    exist_ok=True,
)
os.makedirs(
    CLUSTER_DIR,
    exist_ok=True,
)
# ==========================================================
# C1 / FORECASTING CONFIGURATION
# ==========================================================
C1_LOOKBACK = 96
HORIZON = 288
TARGET = "N.PRB.UL.DrbUsed.Avg[%]"
KNOWN_CELLS = [
    "cell021",
    "cell198",
    "cell192",
    "cell214",
]
# ==========================================================
# B FEATURE EXTRACTION CONFIGURATION
#
# These values are copied from your A_stl feature script.
# Do NOT change them unless A_stl changes.
# ==========================================================
B_VALUE_COL = "N.PRB.UL.DrbUsed.Avg[%]"
B_SERIES_COL = "Short name"
B_TIME_COL = "Date"
B_FREQ = "15min"
B_SEASONAL_PERIOD = 96
B_K = 96
# ==========================================================
# OUT-OF-SAMPLE CLUSTER ASSIGNMENT
# ==========================================================
#
# SpectralClustering itself has no predict() method.
#
# Therefore:
#
#   1. B's original clusters remain untouched.
#   2. We reconstruct the exact B feature table.
#   3. We transform the original B points through:
#
#          B RobustScaler -> B PCA
#
#   4. We transform the unseen cell through the same:
#
#          B RobustScaler -> B PCA
#
#   5. We assign the unseen cell to the existing cluster
#      represented by its nearest B/PCA neighbors.
#
# This is an out-of-sample assignment.
#
# It does NOT refit SpectralClustering.
# ==========================================================
N_NEIGHBORS = 15
# ==========================================================
# RANDOM SEEDS
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
        self.horizon = horizon

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
    print("X:", X.shape)
    print("y:", y.shape)
    print("cluster_ids:", cluster_ids.shape)
    print("metadata entries:", len(metadata))
    print("features:", len(feature_columns))
    assert len(X) == len(y)
    assert len(X) == len(cluster_ids)
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
                f"{cell}: multiple clusters found "
                f"in cluster_ids.npy: {actual_clusters}"
            )
        if int(actual_clusters[0]) != cluster:
            raise RuntimeError(
                f"{cell}: metadata cluster={cluster}, "
                f"cluster_ids cluster={actual_clusters[0]}"
            )
        offset = end_idx
    if offset != len(X):
        raise RuntimeError(
            f"Metadata reconstruction consumed "
            f"{offset} sequences but X contains "
            f"{len(X)} sequences."
        )
    print(f"\nReconstructed {len(mapping)} cells " f"from metadata.pkl")
    return mapping


# ==========================================================
# MODEL LOADING
# ==========================================================
def load_model(
    model_name,
    input_size,
):
    model_path = os.path.join(
        MODEL_DIR,
        model_name + ".pt",
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    bidirectional = "_BiLSTM" in model_name
    model = QuantileLSTM(
        input_size=input_size,
        hidden=91,
        layers=1,
        dropout=0.057,
        bidirectional=bidirectional,
        horizon=HORIZON,
    )
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    return model


# ==========================================================
# PREDICTION
# ==========================================================
def predict(
    model,
    X,
):
    X_tensor = torch.from_numpy(np.asarray(X)).float()
    X_tensor = X_tensor.to(DEVICE)
    with torch.no_grad():
        prediction = model(X_tensor)
    return prediction.cpu().numpy()


# ==========================================================
# METRICS
# ==========================================================
def calculate_metrics(
    actual,
    q10,
    q50,
    q90,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )
    q10 = np.asarray(
        q10,
        dtype=float,
    )
    q50 = np.asarray(
        q50,
        dtype=float,
    )
    q90 = np.asarray(
        q90,
        dtype=float,
    )
    valid = np.abs(actual) > 1e-8
    if np.any(valid):
        mape = (
            np.mean(np.abs(actual[valid] - q50[valid]) / np.abs(actual[valid])) * 100.0
        )
    else:
        mape = np.nan
    interval_range = np.mean(q90 - q10)
    coverage = np.mean((actual >= q10) & (actual <= q90)) * 100.0
    return {
        "MAPE_percent": mape,
        "mean_quantile_interval_range": interval_range,
        "coverage_percent": coverage,
    }


# ==========================================================
# TARGET INVERSE TRANSFORMATION
# ==========================================================
def inverse_transform_target(
    values,
    scaler,
    feature_columns,
):
    target_index = feature_columns.index(TARGET)
    target_mean = scaler.mean_[target_index]
    target_scale = scaler.scale_[target_index]
    return np.asarray(values) * target_scale + target_mean


# ==========================================================
# B FEATURE EXTRACTION
#
# EXACTLY MATCHES THE A_stl SCRIPT PROVIDED BY USER
# ==========================================================
def b_preprocess_data(
    df,
):
    df = df.copy()
    df[B_TIME_COL] = pd.to_datetime(df[B_TIME_COL])
    df = df.sort_values(
        [
            B_SERIES_COL,
            B_TIME_COL,
        ]
    )
    df = df[
        [
            B_SERIES_COL,
            B_TIME_COL,
            B_VALUE_COL,
        ]
    ]
    df = df.drop_duplicates(
        subset=[
            B_SERIES_COL,
            B_TIME_COL,
        ]
    )
    return df


def b_align_series(
    group,
):
    series = group.set_index(B_TIME_COL)[B_VALUE_COL].sort_index()
    full_index = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq=B_FREQ,
    )
    series = series.reindex(full_index)
    return series


def b_autocorrelation_features(
    y,
    K,
):
    result = {
        "acf1": np.nan,
        "acf_mean_k": np.nan,
        "acf_max_k": np.nan,
        "seasonal_acf_s": np.nan,
        "pacf1": np.nan,
    }
    if len(y) <= K:
        return result
    y = np.asarray(y)
    y = y - np.mean(y)
    denominator = np.sum(y**2)
    if denominator == 0:
        return result
    acfs = []
    for lag in range(
        1,
        K + 1,
    ):
        numerator = np.sum(y[lag:] * y[:-lag])
        acfs.append(numerator / denominator)
    acfs = np.array(acfs)
    result["acf1"] = acfs[0]
    result["acf_mean_k"] = np.mean(np.abs(acfs))
    result["acf_max_k"] = np.max(np.abs(acfs))
    if len(acfs) >= B_SEASONAL_PERIOD:
        result["seasonal_acf_s"] = acfs[B_SEASONAL_PERIOD - 1]
    try:
        pacf_values = pacf(
            y,
            nlags=1,
        )
        result["pacf1"] = pacf_values[1]
    except Exception:
        pass
    return result


def b_entropy_features(
    y,
):
    result = {
        "entropy": np.nan,
        "spectral_entropy": np.nan,
        "spectral_predictability": np.nan,
    }
    if len(y) < 10:
        return result
    hist, _ = np.histogram(
        y,
        bins=20,
        density=True,
    )
    hist = hist[hist > 0]
    if len(hist):
        result["entropy"] = entropy(hist / hist.sum())
    try:
        freq, power = periodogram(y)
        power = power[1:]
        power = power / np.sum(power)
        spec_entropy = entropy(power)
        result["spectral_entropy"] = spec_entropy
        result["spectral_predictability"] = 1 - spec_entropy / np.log(len(power))
    except Exception:
        pass
    return result


def b_intermittency_features(
    y,
    timestamps,
):
    result = {
        "interarrival_mean": np.nan,
        "interarrival_std": np.nan,
    }
    events = timestamps[y > 0]
    if len(events) > 1:
        gaps = events[1:] - events[:-1]
        gaps = gaps / np.timedelta64(
            15,
            "m",
        )
        result["interarrival_mean"] = np.mean(gaps)
        result["interarrival_std"] = np.std(gaps)
    return result


def b_mase_feature(
    y,
):
    result = {"MASE_naive": np.nan}
    if len(y) < 2:
        return result
    naive_error = np.mean(np.abs(np.diff(y)))
    if naive_error == 0:
        return result
    prediction_error = np.mean(np.abs(y[1:] - y[:-1]))
    result["MASE_naive"] = prediction_error / naive_error
    return result


def b_trend_features(
    y,
):
    result = {
        "trend_slope": np.nan,
        # "seasonal_strength": np.nan,
    }
    try:
        x = np.arange(len(y))
        slope, _ = np.polyfit(
            x,
            y,
            1,
        )
        result["trend_slope"] = slope
    except Exception:
        pass
    try:
        decomposition = seasonal_decompose(
            y,
            period=B_SEASONAL_PERIOD,
            model="additive",
            extrapolate_trend="freq",
        )
        resid_var = np.var(decomposition.resid.dropna())
        seasonal_var = np.var(decomposition.seasonal.dropna())
        # result[
        #     "seasonal_strength"
        # ] = (
        #     seasonal_var
        #     / (
        #         seasonal_var
        #         + resid_var
        #     )
        # )
    except Exception:
        pass
    return result


def b_extract_features(
    series_id,
    y,
    timestamps,
):
    n = len(y)
    missing_rate = np.mean(pd.isna(y))
    y_filled = pd.Series(y).interpolate().fillna(0).values
    features = {
        "series_id": series_id,
        "n_obs": n,
        "missing_rate": missing_rate,
        "zero_fraction": np.mean(y_filled == 0),
        "mean": np.mean(y_filled),
        "std": np.std(y_filled),
        "cv": (
            np.std(y_filled) / np.mean(y_filled) if np.mean(y_filled) != 0 else np.nan
        ),
    }
    features.update(
        b_autocorrelation_features(
            y_filled,
            B_K,
        )
    )
    features.update(b_entropy_features(y_filled))
    features.update(
        b_intermittency_features(
            y_filled,
            timestamps,
        )
    )
    features.update(b_mase_feature(y_filled))
    features.update(b_trend_features(y_filled))
    return features


def extract_b_features_from_dataframe(
    df,
):
    df = b_preprocess_data(df)
    all_features = []
    for cell, group in df.groupby(B_SERIES_COL):
        series = b_align_series(group)
        features = b_extract_features(
            cell,
            series.values,
            series.index.values,
        )
        all_features.append(features)
    return pd.DataFrame(all_features)


# ==========================================================
# LOAD B TRAINING FEATURE TABLE
# ==========================================================
def load_b_reference_data():
    print("\nLoading B clustering reference data...")
    if not os.path.exists(B_FEATURE_FILE):
        raise FileNotFoundError("B feature file not found:\n" f"{B_FEATURE_FILE}")
    if not os.path.exists(B_ASSIGNMENTS_FILE):
        raise FileNotFoundError(
            "B cluster assignments not found:\n" f"{B_ASSIGNMENTS_FILE}"
        )
    if not os.path.exists(B_MODEL_PATH):
        raise FileNotFoundError("B clustering artifact not found:\n" f"{B_MODEL_PATH}")
    features = pd.read_csv(B_FEATURE_FILE)
    assignments = pd.read_csv(B_ASSIGNMENTS_FILE)
    if "series_id" not in features.columns:
        raise RuntimeError("B feature table does not contain " "'series_id'.")
    if "series_id" not in assignments.columns:
        raise RuntimeError(
            "B cluster assignment table does not " "contain 'series_id'."
        )
    if "cluster" not in assignments.columns:
        raise RuntimeError("B cluster assignment table does not " "contain 'cluster'.")
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
            "Not every B feature row has a "
            "cluster assignment.\n"
            f"Missing assignments: {sorted(missing)}"
        )
    with open(
        B_MODEL_PATH,
        "rb",
    ) as f:
        artifact = pickle.load(f)
    if not isinstance(
        artifact,
        dict,
    ):
        raise RuntimeError("Unexpected B cluster artifact format.")
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
    b_spectral_model = artifact["model"]
    # The B script creates X by dropping only series_id.
    b_feature_columns = [col for col in features.columns if col != "series_id"]
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
        b_spectral_model,
    )


# ==========================================================
# ASSIGN NEW CELL TO EXISTING B CLUSTER
# ==========================================================
def assign_unseen_cluster(
    unseen_b_features,
    reference_b_features,
    b_feature_columns,
    b_scaler,
    b_pca,
    reference_clusters,
    cell_name,
):
    print(f"\nAssigning B cluster for {cell_name}...")
    # ------------------------------------------------------
    # Validate feature columns
    # ------------------------------------------------------
    missing = [col for col in b_feature_columns if col not in unseen_b_features.columns]
    if missing:
        raise RuntimeError(f"{cell_name}: missing B features:\n" f"{missing}")
    # ------------------------------------------------------
    # Make absolutely sure feature order is identical
    # to B training.
    # ------------------------------------------------------
    X_reference = reference_b_features[b_feature_columns].copy()
    X_unseen = unseen_b_features[b_feature_columns].copy()
    # ------------------------------------------------------
    # B_cluster dropped NaN rows before fitting.
    #
    # For an unseen cell we cannot silently fill these,
    # because that would change the B representation.
    # ------------------------------------------------------
    if X_reference.isna().any().any():
        bad_cols = X_reference.columns[X_reference.isna().any()].tolist()
        raise RuntimeError("B reference feature table contains " f"NaNs in: {bad_cols}")
    if X_unseen.isna().any().any():
        bad_cols = X_unseen.columns[X_unseen.isna().any()].tolist()
        raise RuntimeError(
            f"{cell_name}: B feature extraction "
            "produced NaN values in:\n"
            f"{bad_cols}\n\n"
            "This cell cannot be assigned to an "
            "existing B cluster without changing "
            "the B feature representation."
        )
    # ------------------------------------------------------
    # Transform reference and unseen data using the exact
    # scaler fitted by B.
    #
    # DataFrames are deliberately passed here so sklearn
    # feature-name validation remains consistent.
    # ------------------------------------------------------
    X_reference_scaled = b_scaler.transform(X_reference)
    X_unseen_scaled = b_scaler.transform(X_unseen)
    # ------------------------------------------------------
    # Transform into the exact PCA space used by B.
    # ------------------------------------------------------
    X_reference_pca = b_pca.transform(X_reference_scaled)
    X_unseen_pca = b_pca.transform(X_unseen_scaled)
    # ------------------------------------------------------
    # Nearest-neighbor assignment.
    #
    # SpectralClustering has no predict() method.
    #
    # We therefore keep B's six clusters frozen and
    # determine which existing cluster the new cell is
    # closest to in B's PCA space.
    # ------------------------------------------------------
    n_neighbors = min(
        N_NEIGHBORS,
        len(X_reference_pca),
    )
    if n_neighbors < 1:
        raise RuntimeError("No B reference observations available.")
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    )
    nn.fit(X_reference_pca)
    distances, indices = nn.kneighbors(X_unseen_pca)
    distances = distances[0]
    indices = indices[0]
    neighbor_clusters = reference_clusters[indices]
    # ------------------------------------------------------
    # Distance-weighted voting.
    #
    # Closer B cells have more influence.
    # ------------------------------------------------------
    weights = 1.0 / (distances + 1e-8)
    cluster_scores = {}
    for cluster, weight in zip(
        neighbor_clusters,
        weights,
    ):
        cluster = int(cluster)
        cluster_scores[cluster] = (
            cluster_scores.get(
                cluster,
                0.0,
            )
            + weight
        )
    assigned_cluster = max(
        cluster_scores,
        key=cluster_scores.get,
    )
    # ------------------------------------------------------
    # Diagnostic information
    # ------------------------------------------------------
    nearest_rows = []
    for rank, (
        distance,
        index,
        cluster,
    ) in enumerate(
        zip(
            distances,
            indices,
            neighbor_clusters,
        ),
        start=1,
    ):
        nearest_rows.append(
            {
                "cell": cell_name,
                "rank": rank,
                "reference_cell": (reference_b_features.iloc[index]["series_id"]),
                "cluster": int(cluster),
                "distance_in_B_PCA": float(distance),
                "weight": float(1.0 / (distance + 1e-8)),
            }
        )
    diagnostics = pd.DataFrame(nearest_rows)
    diagnostics.to_csv(
        os.path.join(
            CLUSTER_DIR,
            f"{cell_name}_nearest_neighbors.csv",
        ),
        index=False,
    )
    score_rows = []
    total_score = sum(cluster_scores.values())
    for cluster, score in sorted(cluster_scores.items()):
        score_rows.append(
            {
                "cell": cell_name,
                "cluster": cluster,
                "weighted_score": score,
                "weighted_score_percent": (
                    100.0 * score / total_score if total_score > 0 else np.nan
                ),
                "assigned": (cluster == assigned_cluster),
            }
        )
    score_df = pd.DataFrame(score_rows)
    score_df.to_csv(
        os.path.join(
            CLUSTER_DIR,
            f"{cell_name}_cluster_scores.csv",
        ),
        index=False,
    )
    # ------------------------------------------------------
    # Print assignment
    # ------------------------------------------------------
    print(f"Assigned cluster: {assigned_cluster}")
    print("Cluster scores:")
    print(score_df.to_string(index=False))
    print("Nearest reference cells:")
    print(diagnostics.head(min(10, len(diagnostics))).to_string(index=False))
    return int(assigned_cluster)


# ==========================================================
# ORIGINAL DATASET
# ==========================================================
def evaluate_original(
    cell_mapping,
    feature_columns,
    scaler,
):
    print("\n")
    print("=" * 70)
    print("ORIGINAL DATASET EVALUATION")
    print("=" * 70)
    input_size = len(feature_columns)
    results = []
    for cell in KNOWN_CELLS:
        print(f"\nEvaluating {cell}")
        if cell not in cell_mapping:
            print(f"WARNING: {cell} not found " "in C1 metadata.pkl")
            continue
        info = cell_mapping[cell]
        cluster = info["cluster"]
        X_cell = info["X"]
        y_cell = info["y"]
        print(f"Cell: {cell}")
        print(f"Cluster: {cluster}")
        print(f"Number of sequences: " f"{len(X_cell)}")
        if len(X_cell) == 0:
            print(f"WARNING: no sequences for {cell}")
            continue
        # --------------------------------------------------
        # Final C1 sequence
        # --------------------------------------------------
        X_eval = X_cell[-1:]
        y_eval = y_cell[-1:]
        # --------------------------------------------------
        # Convert scaled target back to real units.
        # --------------------------------------------------
        actual_real = inverse_transform_target(
            y_eval[0],
            scaler,
            feature_columns,
        )
        for model_type in [
            "LSTM",
            "BiLSTM",
        ]:
            model_name = f"cluster_{cluster}_{model_type}"
            print(f"Running {model_name}")
            model = load_model(
                model_name,
                input_size,
            )
            prediction = predict(
                model,
                X_eval,
            )[0]
            q10_scaled = prediction[
                :,
                0,
            ]
            q50_scaled = prediction[
                :,
                1,
            ]
            q90_scaled = prediction[
                :,
                2,
            ]
            q10_real = inverse_transform_target(
                q10_scaled,
                scaler,
                feature_columns,
            )
            q50_real = inverse_transform_target(
                q50_scaled,
                scaler,
                feature_columns,
            )
            q90_real = inverse_transform_target(
                q90_scaled,
                scaler,
                feature_columns,
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
                "cluster": cluster,
                "model": model_name,
                "horizon": HORIZON,
                **metrics,
            }
            results.append(result)
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
                    f"original_{cell}_{model_name}.csv",
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
# C1 FEATURE ENGINEERING FOR UNSEEN FORECASTING
# ==========================================================
def prepare_unseen_forecasting_features(
    unseen,
):
    unseen = unseen.copy()
    unseen["Date"] = pd.to_datetime(unseen["Date"])
    unseen = unseen.sort_values(
        [
            "Short name",
            "Date",
        ]
    )
    # ------------------------------------------------------
    # Time features
    # ------------------------------------------------------
    unseen["hour"] = unseen["Date"].dt.hour
    unseen["dayofweek"] = unseen["Date"].dt.dayofweek
    unseen["hour_sin"] = np.sin(2 * np.pi * unseen["hour"] / 24)
    unseen["hour_cos"] = np.cos(2 * np.pi * unseen["hour"] / 24)
    unseen["dow_sin"] = np.sin(2 * np.pi * unseen["dayofweek"] / 7)
    unseen["dow_cos"] = np.cos(2 * np.pi * unseen["dayofweek"] / 7)
    # ------------------------------------------------------
    # Throughput
    # ------------------------------------------------------
    unseen["throughput_per_user"] = unseen["N.ThpVol.UL"] / (
        unseen["N.User.RRCConn.Active.UL.Avg"] + 1
    )
    # ------------------------------------------------------
    # PRB change
    # ------------------------------------------------------
    unseen["prb_change"] = unseen.groupby("Short name")[TARGET].diff()
    # ------------------------------------------------------
    # Lag features
    # ------------------------------------------------------
    for lag in [
        1,
        2,
        4,
        8,
        96,
        672,
    ]:
        unseen[f"prb_lag_{lag}"] = unseen.groupby("Short name")[TARGET].shift(lag)
    # ------------------------------------------------------
    # Rolling features
    # ------------------------------------------------------
    for window in [
        4,
        16,
        96,
    ]:
        unseen[f"prb_mean_{window}"] = unseen.groupby("Short name")[TARGET].transform(
            lambda x: x.rolling(window).mean()
        )
        unseen[f"prb_std_{window}"] = unseen.groupby("Short name")[TARGET].transform(
            lambda x: x.rolling(window).std()
        )
    return unseen


# ==========================================================
# UNSEEN DATASET
# ==========================================================
def evaluate_unseen(
    feature_columns,
    scaler,
    b_reference,
    b_feature_columns,
    b_scaler,
    b_pca,
):
    print("\n")
    print("=" * 70)
    print("UNSEEN DATASET EVALUATION")
    print("=" * 70)
    if not os.path.exists(UNSEEN_DATA):
        raise FileNotFoundError(f"Unseen dataset not found: " f"{UNSEEN_DATA}")
    unseen_raw = pd.read_csv(UNSEEN_DATA)
    # ------------------------------------------------------
    # Validate raw B features
    # ------------------------------------------------------
    required_b_raw = [
        B_SERIES_COL,
        B_TIME_COL,
        B_VALUE_COL,
    ]
    missing_b_raw = [col for col in required_b_raw if col not in unseen_raw.columns]
    if missing_b_raw:
        raise RuntimeError(
            "Unseen dataset is missing columns "
            "required by B feature extraction:\n"
            f"{missing_b_raw}"
        )
    # ------------------------------------------------------
    # Validate raw C1 forecasting features
    # ------------------------------------------------------
    required_forecasting_raw = [
        "N.ThpVol.UL",
        "N.User.RRCConn.Active.UL.Avg",
        TARGET,
        "Date",
        "Short name",
    ]
    missing_forecasting_raw = [
        col for col in required_forecasting_raw if col not in unseen_raw.columns
    ]
    if missing_forecasting_raw:
        raise RuntimeError(
            "Unseen dataset is missing raw "
            "forecasting columns:\n"
            f"{missing_forecasting_raw}"
        )
    print("Unseen cells:")
    print(sorted(unseen_raw["Short name"].unique()))
    # ======================================================
    # STEP 1
    # EXACT B FEATURE EXTRACTION
    # ======================================================
    print("\nExtracting B forecastability " "features from unseen cells...")
    unseen_b_features = extract_b_features_from_dataframe(unseen_raw)
    unseen_b_features.to_csv(
        os.path.join(
            CLUSTER_DIR,
            "unseen_b_features.csv",
        ),
        index=False,
    )
    # ------------------------------------------------------
    # Verify B feature columns
    # ------------------------------------------------------
    missing_b_features = [
        col for col in b_feature_columns if col not in unseen_b_features.columns
    ]
    if missing_b_features:
        raise RuntimeError(
            "The B feature extraction in C3 does "
            "not reproduce all B feature columns.\n\n"
            f"Missing:\n{missing_b_features}"
        )
    # ======================================================
    # STEP 2
    # PREPARE C1 FORECASTING FEATURES
    # ======================================================
    unseen = prepare_unseen_forecasting_features(unseen_raw)
    # ------------------------------------------------------
    # Check C1 feature columns
    # ------------------------------------------------------
    missing = [col for col in feature_columns if col not in unseen.columns]
    if missing:
        raise RuntimeError(
            "Unseen dataset is missing "
            "forecasting features required by C1:\n"
            f"{missing}"
        )
    # ======================================================
    # STEP 3
    # EVALUATE EACH UNSEEN CELL
    # ======================================================
    results = []
    cluster_assignments = []
    for cell, cell_df in unseen.groupby("Short name"):
        cell_df = cell_df.sort_values("Date").reset_index(drop=True)
        print(f"\nUnseen cell: {cell}")
        # --------------------------------------------------
        # Check enough raw observations.
        # --------------------------------------------------
        if len(cell_df) < (672 + HORIZON):
            raise RuntimeError(
                f"{cell} has {len(cell_df)} rows. "
                "Not enough data to construct "
                "C1 lag features and a 288-step "
                "evaluation horizon."
            )
        # --------------------------------------------------
        # Drop rows that C1 would drop.
        # --------------------------------------------------
        cell_df = cell_df.dropna(
            subset=[c for c in feature_columns if c in cell_df.columns]
        ).reset_index(drop=True)
        if len(cell_df) < (C1_LOOKBACK + HORIZON):
            raise RuntimeError(
                f"{cell}: insufficient valid rows " "after C1 feature engineering."
            )
        # ==================================================
        # CLUSTER ASSIGNMENT
        # ==================================================
        unseen_b_row = unseen_b_features[
            unseen_b_features["series_id"].astype(str) == str(cell)
        ]
        if len(unseen_b_row) != 1:
            raise RuntimeError(
                f"{cell}: expected exactly one "
                "B feature row, found "
                f"{len(unseen_b_row)}."
            )
        assigned_cluster = assign_unseen_cluster(
            unseen_b_features=unseen_b_row,
            reference_b_features=b_reference[["series_id"] + b_feature_columns],
            b_feature_columns=b_feature_columns,
            b_scaler=b_scaler,
            b_pca=b_pca,
            reference_clusters=b_reference["cluster"].to_numpy(),
            cell_name=cell,
        )
        print(f"{cell} -> existing B cluster " f"{assigned_cluster}")
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
        # --------------------------------------------------
        # Scale C1 input using C1's original scaler.
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
        # FORECAST USING THE ASSIGNED CLUSTER MODEL
        # ==================================================
        for model_type in [
            "LSTM",
            "BiLSTM",
        ]:
            model_name = f"cluster_{assigned_cluster}_" f"{model_type}"
            print(f"Running {model_name}")
            model = load_model(
                model_name,
                input_size,
            )
            prediction = predict(
                model,
                X_eval,
            )[0]
            q10_real = inverse_transform_target(
                prediction[
                    :,
                    0,
                ],
                scaler,
                feature_columns,
            )
            q50_real = inverse_transform_target(
                prediction[
                    :,
                    1,
                ],
                scaler,
                feature_columns,
            )
            q90_real = inverse_transform_target(
                prediction[
                    :,
                    2,
                ],
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
                    "timestamp": (actual_df["Date"].values),
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
    # UNSEEN AGGREGATION
    # ======================================================
    if not results_df.empty:
        overall = pd.DataFrame(
            [
                {
                    "dataset": "unseen",
                    "group": "all_unseen_cells",
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
    print("=" * 70)
    print("C3 EVALUATION")
    print("=" * 70)
    print(
        "Device:",
        DEVICE,
    )
    # ======================================================
    # LOAD C1
    # ======================================================
    (
        X,
        y,
        cluster_ids,
        metadata,
        scaler,
        feature_columns,
    ) = load_c1_artifacts()
    # ======================================================
    # RECONSTRUCT ORIGINAL CELL SEQUENCES
    # ======================================================
    cell_mapping = reconstruct_cell_mapping(
        X,
        y,
        cluster_ids,
        metadata,
    )
    # ======================================================
    # LOAD B REFERENCE DATA + B TRANSFORMERS
    # ======================================================
    (
        b_reference,
        b_feature_columns,
        b_scaler,
        b_pca,
        b_spectral_model,
    ) = load_b_reference_data()
    # ------------------------------------------------------
    # Informational check:
    #
    # The spectral model itself remains frozen.
    # We deliberately do not call fit() or predict() on it.
    # ------------------------------------------------------
    print(
        "\nB clustering model:",
        type(b_spectral_model).__name__,
    )
    print(
        "B spectral clustering is "
        "used as the frozen source of "
        "the existing cluster solution."
    )
    # ======================================================
    # ORIGINAL DATA
    # ======================================================
    original_results = evaluate_original(
        cell_mapping,
        feature_columns,
        scaler,
    )
    aggregate_original(original_results)
    # ======================================================
    # UNSEEN DATA
    # ======================================================
    unseen_results = evaluate_unseen(
        feature_columns,
        scaler,
        b_reference,
        b_feature_columns,
        b_scaler,
        b_pca,
    )
    # ======================================================
    # COMBINED REPORT
    # ======================================================
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
    # ======================================================
    # FINAL
    # ======================================================
    print("\n")
    print("=" * 70)
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
