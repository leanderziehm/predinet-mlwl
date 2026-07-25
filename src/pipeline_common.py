import os
import pickle
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.signal import periodogram
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from statsmodels.tsa.stattools import pacf


# ==========================================================
# GLOBAL CONFIGURATION
# ==========================================================

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------

DATA_PATH = "UL_PRB_data_set.csv"
UNSEEN_DATA_PATH = "selected_cells_unseen.csv"

TARGET = "N.PRB.UL.DrbUsed.Avg[%]"

SERIES_COL = "Short name"
TIME_COL = "Date"

FREQ = "15min"

# ----------------------------------------------------------
# Forecasting
# ----------------------------------------------------------

# C1_LOOKBACK = 96
C1_LOOKBACK = 672 #96*7
HORIZON = 288

FORECAST_LAGS = [
    1,
    2,
    4,
    8,
    96,
    672,
]

FORECAST_ROLLING_WINDOWS = [
    4,
    16,
    96,
    672
]

FORECAST_FEATURE_COLUMNS = [
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
]

# ----------------------------------------------------------
# B / forecastability features
# ----------------------------------------------------------

B_SEASONAL_PERIOD = 96
B_K = 96

B_VALUE_COL = TARGET
B_SERIES_COL = SERIES_COL
B_TIME_COL = TIME_COL

# ----------------------------------------------------------
# Model
# ----------------------------------------------------------

MODEL_HIDDEN = 91
MODEL_LAYERS = 1
MODEL_DROPOUT = 0.057

QUANTILES = (
    0.1,
    0.5,
    0.9,
)

# ----------------------------------------------------------
# Clustering
# ----------------------------------------------------------

N_CLUSTERS = 6
PCA_VARIANCE = 0.95
N_NEIGHBORS = 15


# ==========================================================
# RANDOM SEEDS
# ==========================================================

def set_random_seeds(seed=SEED):
    """
    Set all random seeds used by the pipeline.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================================
# DATA LOADING
# ==========================================================

def load_time_series_dataframe(path=DATA_PATH):
    """
    Load the raw telecom time-series dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if TIME_COL not in df.columns:
        raise RuntimeError(
            f"Dataset is missing required column: {TIME_COL}"
        )

    if SERIES_COL not in df.columns:
        raise RuntimeError(
            f"Dataset is missing required column: {SERIES_COL}"
        )

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    df = df.sort_values(
        [
            SERIES_COL,
            TIME_COL,
        ]
    ).reset_index(drop=True)

    return df


# ==========================================================
# FORECASTING FEATURE ENGINEERING
# ==========================================================

def create_forecasting_features(
    df,
    dropna=True,
):
    """
    Create the exact forecasting features used by C1.

    This function MUST be used by:
        - C1 preprocessing
        - unseen-cell evaluation

    Keeping this in one place prevents train/evaluation
    feature drift.
    """
    df = df.copy()

    required_columns = [
        TARGET,
        "N.ThpVol.UL",
        "N.User.RRCConn.Active.UL.Avg",
        TIME_COL,
        SERIES_COL,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Cannot create forecasting features. "
            f"Missing columns: {missing}"
        )

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    df = df.sort_values(
        [
            SERIES_COL,
            TIME_COL,
        ]
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # Time features
    # ------------------------------------------------------

    df["hour"] = df[TIME_COL].dt.hour
    df["dayofweek"] = df[TIME_COL].dt.dayofweek

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["dow_sin"] = np.sin(
        2 * np.pi * df["dayofweek"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["dayofweek"] / 7
    )

    # ------------------------------------------------------
    # Throughput per user
    # ------------------------------------------------------

    df["throughput_per_user"] = (
        df["N.ThpVol.UL"]
        / (
            df["N.User.RRCConn.Active.UL.Avg"]
            + 1
        )
    )

    # ------------------------------------------------------
    # PRB difference
    # ------------------------------------------------------

    df["prb_change"] = (
        df.groupby(SERIES_COL)[TARGET]
        .diff()
    )

    # ------------------------------------------------------
    # Lag features
    # ------------------------------------------------------

    for lag in FORECAST_LAGS:
        df[f"prb_lag_{lag}"] = (
            df.groupby(SERIES_COL)[TARGET]
            .shift(lag)
        )

    # ------------------------------------------------------
    # Rolling features
    # ------------------------------------------------------

    for window in FORECAST_ROLLING_WINDOWS:

        df[f"prb_mean_{window}"] = (
            df.groupby(SERIES_COL)[TARGET]
            .transform(
                lambda x: x.rolling(window).mean()
            )
        )

        df[f"prb_std_{window}"] = (
            df.groupby(SERIES_COL)[TARGET]
            .transform(
                lambda x: x.rolling(window).std()
            )
        )

    if dropna:
        df = df.dropna().reset_index(drop=True)

    return df


# ==========================================================
# FORECASTING SEQUENCE CREATION
# ==========================================================

def create_sequences(
    cell_df,
    feature_columns=FORECAST_FEATURE_COLUMNS,
    lookback=C1_LOOKBACK,
    horizon=HORIZON,
):
    """
    Create forecasting sequences for one cell.

    X:
        [samples, lookback, features]

    y:
        [samples, horizon]
    """

    if SERIES_COL not in cell_df.columns:
        raise RuntimeError(
            f"Missing column: {SERIES_COL}"
        )

    if TARGET not in cell_df.columns:
        raise RuntimeError(
            f"Missing target column: {TARGET}"
        )

    cell_df = cell_df.sort_values(
        TIME_COL
    ).reset_index(drop=True)

    missing_features = [
        column
        for column in feature_columns
        if column not in cell_df.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Cannot create sequences. "
            f"Missing features: {missing_features}"
        )

    n_samples = (
        len(cell_df)
        - lookback
        - horizon
    )

    if n_samples <= 0:
        return (
            np.empty(
                (
                    0,
                    lookback,
                    len(feature_columns),
                ),
                dtype=np.float32,
            ),
            np.empty(
                (
                    0,
                    horizon,
                ),
                dtype=np.float32,
            ),
            {
                "cell": str(
                    cell_df[SERIES_COL].iloc[0]
                ),
                "cluster": (
                    int(cell_df["cluster"].iloc[0])
                    if "cluster" in cell_df.columns
                    else None
                ),
                "start": np.array([], dtype="datetime64[ns]"),
            },
            (
                int(cell_df["cluster"].iloc[0])
                if "cluster" in cell_df.columns
                else None
            ),
        )

    values = (
        cell_df[feature_columns]
        .to_numpy(dtype=np.float32)
    )

    target = (
        cell_df[TARGET]
        .to_numpy(dtype=np.float32)
    )

    X = np.empty(
        (
            n_samples,
            lookback,
            len(feature_columns),
        ),
        dtype=np.float32,
    )

    y = np.empty(
        (
            n_samples,
            horizon,
        ),
        dtype=np.float32,
    )

    for i in range(n_samples):

        X[i] = values[
            i : i + lookback
        ]

        y[i] = target[
            i + lookback :
            i + lookback + horizon
        ]

    cluster = None

    if "cluster" in cell_df.columns:
        cluster_values = (
            cell_df["cluster"]
            .dropna()
            .unique()
        )

        if len(cluster_values) != 1:
            raise RuntimeError(
                f"{cell_df[SERIES_COL].iloc[0]} "
                f"has multiple clusters: "
                f"{cluster_values}"
            )

        cluster = int(
            cluster_values[0]
        )

    meta = {
        "cell": str(
            cell_df[SERIES_COL].iloc[0]
        ),
        "cluster": cluster,
        "start": cell_df[
            TIME_COL
        ].values[
            lookback :
            lookback + n_samples
        ],
    }

    return X, y, meta, cluster


# ==========================================================
# MODEL
# ==========================================================

class QuantileLSTM(nn.Module):
    """
    Shared model definition used by C2 training and D1
    evaluation.

    IMPORTANT:
    Do not redefine this class in C2 or D1.
    """

    def __init__(
        self,
        input_size,
        hidden=MODEL_HIDDEN,
        layers=MODEL_LAYERS,
        dropout=MODEL_DROPOUT,
        bidirectional=False,
        horizon=HORIZON,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=(
                dropout
                if layers > 1
                else 0
            ),
            bidirectional=bidirectional,
        )

        multiplier = (
            2
            if bidirectional
            else 1
        )

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
# DATASET
# ==========================================================

class ForecastDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        X,
        y,
    ):
        self.X = torch.from_numpy(
            np.asarray(X)
        ).float()

        self.y = torch.from_numpy(
            np.asarray(y)
        ).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y[idx],
        )


# ==========================================================
# QUANTILE LOSS
# ==========================================================

def quantile_loss(
    pred,
    target,
    q,
):
    error = target - pred

    return torch.mean(
        torch.maximum(
            q * error,
            (q - 1) * error,
        )
    )


def multi_quantile_loss(
    prediction,
    target,
):
    q10 = prediction[:, :, 0]
    q50 = prediction[:, :, 1]
    q90 = prediction[:, :, 2]

    loss = (
        quantile_loss(
            q10,
            target,
            0.1,
        )
        + quantile_loss(
            q50,
            target,
            0.5,
        )
        + quantile_loss(
            q90,
            target,
            0.9,
        )
    )

    return loss / 3.0


# ==========================================================
# TARGET SCALING
# ==========================================================

def get_target_scaling(
    scaler,
    feature_columns,
    target=TARGET,
):
    target_index = feature_columns.index(
        target
    )

    target_mean = scaler.mean_[
        target_index
    ]

    target_std = scaler.scale_[
        target_index
    ]

    return (
        target_mean,
        target_std,
    )


def inverse_transform_target(
    values,
    scaler,
    feature_columns,
    target=TARGET,
):
    target_mean, target_std = (
        get_target_scaling(
            scaler,
            feature_columns,
            target,
        )
    )

    return (
        np.asarray(values)
        * target_std
        + target_mean
    )


# ==========================================================
# PREDICTION
# ==========================================================

def predict(
    model,
    X,
    device=DEVICE,
):
    """
    Shared prediction function used by evaluation.
    """

    X_array = np.asarray(
        X,
        dtype=np.float32,
    )

    X_tensor = torch.from_numpy(
        X_array
    ).float().to(device)

    model.eval()

    with torch.no_grad():
        prediction = model(
            X_tensor
        )

    return prediction.cpu().numpy()


# ==========================================================
# MODEL LOADING
# ==========================================================

def load_quantile_model(
    model_path,
    input_size,
    bidirectional=False,
    horizon=HORIZON,
    device=DEVICE,
):
    if not os.path.exists(
        model_path
    ):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = QuantileLSTM(
        input_size=input_size,
        hidden=MODEL_HIDDEN,
        layers=MODEL_LAYERS,
        dropout=MODEL_DROPOUT,
        bidirectional=bidirectional,
        horizon=horizon,
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint does not contain "
            f"'model_state': {model_path}"
        )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.to(device)
    model.eval()

    return model


def load_cluster_model(
    model_dir,
    cluster,
    model_type,
    input_size,
    horizon=HORIZON,
    device=DEVICE,
):
    if model_type not in (
        "LSTM",
        "BiLSTM",
    ):
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    model_name = (
        f"cluster_{cluster}_{model_type}"
    )

    model_path = os.path.join(
        model_dir,
        model_name + ".pt",
    )

    model = load_quantile_model(
        model_path=model_path,
        input_size=input_size,
        bidirectional=(
            model_type == "BiLSTM"
        ),
        horizon=horizon,
        device=device,
    )

    return model, model_name


# ==========================================================
# EVALUATION METRICS
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

    if not (
        actual.shape
        == q10.shape
        == q50.shape
        == q90.shape
    ):
        raise ValueError(
            "actual, q10, q50 and q90 "
            "must have identical shapes."
        )

    valid = (
        np.abs(actual) > 1e-8
    )

    if np.any(valid):
        mape = (
            np.mean(
                np.abs(
                    actual[valid]
                    - q50[valid]
                )
                / np.abs(
                    actual[valid]
                )
            )
            * 100.0
        )
    else:
        mape = np.nan

    interval_range = np.mean(
        q90 - q10
    )

    coverage = (
        np.mean(
            (actual >= q10)
            & (actual <= q90)
        )
        * 100.0
    )

    return {
        "MAPE_percent": mape,
        "mean_quantile_interval_range": (
            interval_range
        ),
        "coverage_percent": coverage,
    }


# ==========================================================
# B FEATURE EXTRACTION
# ==========================================================

def b_preprocess_data(
    df,
):
    df = df.copy()

    df[B_TIME_COL] = pd.to_datetime(
        df[B_TIME_COL]
    )

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
    series = (
        group
        .set_index(B_TIME_COL)[B_VALUE_COL]
        .sort_index()
    )

    full_index = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq=FREQ,
    )

    return series.reindex(
        full_index
    )


def b_autocorrelation_features(
    y,
    K=B_K,
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

    denominator = np.sum(
        y ** 2
    )

    if denominator == 0:
        return result

    acfs = []

    for lag in range(
        1,
        K + 1,
    ):
        numerator = np.sum(
            y[lag:]
            * y[:-lag]
        )

        acfs.append(
            numerator
            / denominator
        )

    acfs = np.asarray(
        acfs
    )

    result["acf1"] = acfs[0]

    result["acf_mean_k"] = (
        np.mean(
            np.abs(acfs)
        )
    )

    result["acf_max_k"] = (
        np.max(
            np.abs(acfs)
        )
    )

    if (
        len(acfs)
        >= B_SEASONAL_PERIOD
    ):
        result[
            "seasonal_acf_s"
        ] = acfs[
            B_SEASONAL_PERIOD - 1
        ]

    try:
        pacf_values = pacf(
            y,
            nlags=1,
        )

        result["pacf1"] = (
            pacf_values[1]
        )

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

    hist = hist[
        hist > 0
    ]

    if len(hist):
        result["entropy"] = (
            entropy(
                hist
                / hist.sum()
            )
        )

    try:
        _, power = periodogram(
            y
        )

        power = power[1:]

        power_sum = np.sum(
            power
        )

        if power_sum <= 0:
            return result

        power = (
            power
            / power_sum
        )

        spec_entropy = (
            entropy(power)
        )

        result[
            "spectral_entropy"
        ] = spec_entropy

        if len(power) > 1:
            result[
                "spectral_predictability"
            ] = (
                1
                - spec_entropy
                / np.log(
                    len(power)
                )
            )

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

    y = np.asarray(y)

    timestamps = np.asarray(
        timestamps
    )

    events = timestamps[
        y > 0
    ]

    if len(events) > 1:

        gaps = (
            events[1:]
            - events[:-1]
        )

        gaps = (
            gaps
            / np.timedelta64(
                15,
                "m",
            )
        )

        result[
            "interarrival_mean"
        ] = np.mean(gaps)

        result[
            "interarrival_std"
        ] = np.std(gaps)

    return result


def b_mase_feature(
    y,
):
    result = {
        "MASE_naive": np.nan
    }

    if len(y) < 2:
        return result

    naive_error = np.mean(
        np.abs(
            np.diff(y)
        )
    )

    if naive_error == 0:
        return result

    prediction_error = np.mean(
        np.abs(
            y[1:]
            - y[:-1]
        )
    )

    result[
        "MASE_naive"
    ] = (
        prediction_error
        / naive_error
    )

    return result


def b_trend_features(
    y,
):
    result = {
        "trend_slope": np.nan,
    }

    try:
        x = np.arange(
            len(y)
        )

        slope, _ = np.polyfit(
            x,
            y,
            1,
        )

        result[
            "trend_slope"
        ] = slope

    except Exception:
        pass

    return result


def b_extract_features(
    series_id,
    y,
    timestamps,
):
    n = len(y)

    missing_rate = np.mean(
        pd.isna(y)
    )

    y_filled = (
        pd.Series(y)
        .interpolate()
        .fillna(0)
        .values
    )

    features = {
        "series_id": series_id,
        "n_obs": n,
        "missing_rate": missing_rate,
        "zero_fraction": (
            np.mean(
                y_filled == 0
            )
        ),
        "mean": np.mean(
            y_filled
        ),
        "std": np.std(
            y_filled
        ),
        "cv": (
            np.std(y_filled)
            / np.mean(y_filled)
            if np.mean(y_filled) != 0
            else np.nan
        ),
    }

    features.update(
        b_autocorrelation_features(
            y_filled
        )
    )

    features.update(
        b_entropy_features(
            y_filled
        )
    )

    features.update(
        b_intermittency_features(
            y_filled,
            timestamps,
        )
    )

    features.update(
        b_mase_feature(
            y_filled
        )
    )

    features.update(
        b_trend_features(
            y_filled
        )
    )

    return features


def extract_b_features_from_dataframe(
    df,
):
    """
    Shared B feature extraction.

    This is used by:
        - A_stl.py
        - B_cluster.py
        - D1_eval.py

    Therefore the features used to train B and the
    features used for unseen assignment cannot drift apart.
    """

    df = b_preprocess_data(
        df
    )

    all_features = []

    for cell, group in df.groupby(
        B_SERIES_COL
    ):

        series = b_align_series(
            group
        )

        features = (
            b_extract_features(
                cell,
                series.values,
                series.index.values,
            )
        )

        all_features.append(
            features
        )

    return pd.DataFrame(
        all_features
    )


# ==========================================================
# B MODEL TRANSFORMATIONS
# ==========================================================

def transform_b_features(
    features_df,
    feature_columns,
    scaler,
    pca,
):
    """
    Apply the exact B scaler + PCA pipeline.
    """

    X = features_df[
        feature_columns
    ].copy()

    if X.isna().any().any():
        bad_columns = (
            X.columns[
                X.isna().any()
            ].tolist()
        )

        raise RuntimeError(
            "B features contain NaN values "
            f"in: {bad_columns}"
        )

    X_scaled = scaler.transform(
        X
    )

    X_pca = pca.transform(
        X_scaled
    )

    return X_scaled, X_pca


# ==========================================================
# B OUT-OF-SAMPLE CLUSTER ASSIGNMENT
# ==========================================================

def assign_cluster_by_neighbors(
    unseen_features,
    reference_features,
    feature_columns,
    scaler,
    pca,
    reference_clusters,
    cell_name,
    n_neighbors=N_NEIGHBORS,
):
    """
    Assign an unseen cell to an existing frozen cluster
    using distance-weighted nearest-neighbor voting in the
    exact PCA space used by B.

    SpectralClustering itself has no predict() method.
    """

    missing = [
        column
        for column in feature_columns
        if column not in unseen_features.columns
    ]

    if missing:
        raise RuntimeError(
            f"{cell_name}: missing B features: "
            f"{missing}"
        )

    X_reference = (
        reference_features[
            feature_columns
        ].copy()
    )

    X_unseen = (
        unseen_features[
            feature_columns
        ].copy()
    )

    if X_reference.isna().any().any():
        bad = (
            X_reference.columns[
                X_reference.isna().any()
            ].tolist()
        )

        raise RuntimeError(
            "B reference feature table contains "
            f"NaNs in: {bad}"
        )

    if X_unseen.isna().any().any():
        bad = (
            X_unseen.columns[
                X_unseen.isna().any()
            ].tolist()
        )

        raise RuntimeError(
            f"{cell_name}: unseen B feature "
            f"table contains NaNs in: {bad}"
        )

    X_reference_pca = transform_b_features(
        reference_features,
        feature_columns,
        scaler,
        pca,
    )[1]

    X_unseen_pca = transform_b_features(
        unseen_features,
        feature_columns,
        scaler,
        pca,
    )[1]

    n_neighbors = min(
        n_neighbors,
        len(X_reference_pca),
    )

    if n_neighbors < 1:
        raise RuntimeError(
            "No B reference observations available."
        )

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    )

    nn.fit(
        X_reference_pca
    )

    distances, indices = (
        nn.kneighbors(
            X_unseen_pca
        )
    )

    distances = distances[0]
    indices = indices[0]

    neighbor_clusters = (
        np.asarray(
            reference_clusters
        )[indices]
    )

    weights = (
        1.0
        / (
            distances
            + 1e-8
        )
    )

    cluster_scores = {}

    for cluster, weight in zip(
        neighbor_clusters,
        weights,
    ):

        cluster = int(
            cluster
        )

        cluster_scores[
            cluster
        ] = (
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
                "reference_cell": (
                    reference_features.iloc[
                        index
                    ]["series_id"]
                ),
                "cluster": int(
                    cluster
                ),
                "distance_in_B_PCA": float(
                    distance
                ),
                "weight": float(
                    1.0
                    / (
                        distance
                        + 1e-8
                    )
                ),
            }
        )

    diagnostics = pd.DataFrame(
        nearest_rows
    )

    score_rows = []

    total_score = sum(
        cluster_scores.values()
    )

    for cluster, score in sorted(
        cluster_scores.items()
    ):

        score_rows.append(
            {
                "cell": cell_name,
                "cluster": cluster,
                "weighted_score": score,
                "weighted_score_percent": (
                    100.0
                    * score
                    / total_score
                    if total_score > 0
                    else np.nan
                ),
                "assigned": (
                    cluster
                    == assigned_cluster
                ),
            }
        )

    score_df = pd.DataFrame(
        score_rows
    )

    return (
        int(assigned_cluster),
        diagnostics,
        score_df,
    )


# ==========================================================
# VALIDATION HELPERS
# ==========================================================

def validate_feature_columns(
    df,
    feature_columns,
    context="dataset",
):
    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{context} is missing "
            f"required features: {missing}"
        )


def validate_single_cluster(
    cell_df,
    cell_name,
):
    if "cluster" not in cell_df.columns:
        raise RuntimeError(
            f"{cell_name}: cluster column missing."
        )

    clusters = (
        cell_df["cluster"]
        .dropna()
        .unique()
    )

    if len(clusters) != 1:
        raise RuntimeError(
            f"{cell_name}: expected exactly "
            f"one cluster, found {clusters}"
        )

    return int(
        clusters[0]
    )