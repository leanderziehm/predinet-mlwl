import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import entropy
from scipy.signal import periodogram
from statsmodels.tsa.stattools import pacf

SEED = 42
# ============================
# Configuration
# ============================
VALUE_COL = "N.PRB.UL.DrbUsed.Avg[%]"
SERIES_COL = "Short name"
TIME_COL = "Date"
FREQ = "15min"
# 96 points = one day
SEASONAL_PERIOD = 96
# ACF window
K = 96


# ============================
# Output handling
# ============================
def make_output_dir():
    script_path = os.path.abspath(__file__)
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("output", script_name)  # , timestamp)
    os.makedirs(out_dir, exist_ok=True)
    for folder in ["tables"]:
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)
    return out_dir


# ============================
# Data preprocessing
# ============================
def preprocess_data(df):
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values([SERIES_COL, TIME_COL])
    df = df[[SERIES_COL, TIME_COL, VALUE_COL]]
    df = df.drop_duplicates(subset=[SERIES_COL, TIME_COL])
    return df


def align_series(group):
    """
    Convert each cell into a regular
    15 minute time series.
    """
    series = group.set_index(TIME_COL)[VALUE_COL].sort_index()
    full_index = pd.date_range(
        start=series.index.min(), end=series.index.max(), freq=FREQ
    )
    series = series.reindex(full_index)
    return series


# ============================
# Autocorrelation features
# ============================
def autocorrelation_features(y, K):
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
    for lag in range(1, K + 1):
        numerator = np.sum(y[lag:] * y[:-lag])
        acfs.append(numerator / denominator)
    acfs = np.array(acfs)
    result["acf1"] = acfs[0]
    result["acf_mean_k"] = np.mean(np.abs(acfs))
    result["acf_max_k"] = np.max(np.abs(acfs))
    if len(acfs) >= SEASONAL_PERIOD:
        result["seasonal_acf_s"] = acfs[SEASONAL_PERIOD - 1]
    try:
        pacf_values = pacf(y, nlags=1)
        result["pacf1"] = pacf_values[1]
    except:
        pass
    return result


# ============================
# Entropy features
# ============================
def entropy_features(y):
    result = {
        "entropy": np.nan,
        "spectral_entropy": np.nan,
        "spectral_predictability": np.nan,
    }
    if len(y) < 10:
        return result
    # Value distribution entropy
    hist, _ = np.histogram(y, bins=20, density=True)
    hist = hist[hist > 0]
    if len(hist):
        result["entropy"] = entropy(hist / hist.sum())
    # Frequency entropy
    try:
        freq, power = periodogram(y)
        power = power[1:]
        power = power / np.sum(power)
        spec_entropy = entropy(power)
        result["spectral_entropy"] = spec_entropy
        result["spectral_predictability"] = 1 - spec_entropy / np.log(len(power))
    except:
        pass
    return result


# ============================
# Intermittency features
# ============================
def intermittency_features(y, timestamps):
    result = {"interarrival_mean": np.nan, "interarrival_std": np.nan}
    events = timestamps[y > 0]
    if len(events) > 1:
        gaps = events[1:] - events[:-1]
        gaps = gaps / np.timedelta64(15, "m")
        result["interarrival_mean"] = np.mean(gaps)
        result["interarrival_std"] = np.std(gaps)
    return result


# ============================
# Forecast difficulty
# ============================
def mase_feature(y):
    result = {"MASE_naive": np.nan}
    if len(y) < 2:
        return result
    naive_error = np.mean(np.abs(np.diff(y)))
    if naive_error == 0:
        return result
    prediction_error = np.mean(np.abs(y[1:] - y[:-1]))
    result["MASE_naive"] = prediction_error / naive_error
    return result


# ============================
# Trend and seasonality
# ============================
def trend_features(y):
    result = {"trend_slope": np.nan}
    try:
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        result["trend_slope"] = slope
    except:
        pass
    return result


# ============================
# Feature extraction
# ============================
def extract_features(series_id, y, timestamps):
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
    features.update(autocorrelation_features(y_filled, K))
    features.update(entropy_features(y_filled))
    features.update(intermittency_features(y_filled, timestamps))
    features.update(mase_feature(y_filled))
    features.update(trend_features(y_filled))
    return features


# ============================
# Main pipeline
# ============================
def main():
    np.random.seed(SEED)
    out_dir = make_output_dir()
    log_file = open(os.path.join(out_dir, "run_log.txt"), "w")

    def log(msg):
        line = f"[{datetime.now()}] {msg}"
        print(line)
        log_file.write(line + "\n")

    log("Loading dataset")
    df = pd.read_csv("UL_PRB_data_set.csv")
    df = preprocess_data(df)
    log(f"Loaded {df[SERIES_COL].nunique()} cells")
    all_features = []
    for cell, group in df.groupby(SERIES_COL):
        series = align_series(group)
        features = extract_features(cell, series.values, series.index.values)
        all_features.append(features)
    feature_df = pd.DataFrame(all_features)
    output_file = os.path.join(out_dir, "tables", "cell_forecastability_features.csv")
    feature_df.to_csv(output_file, index=False)
    summary = feature_df.describe().T
    summary.to_csv(os.path.join(out_dir, "tables", "feature_summary.csv"))
    log(f"Saved feature table: {output_file}")
    log_file.close()


if __name__ == "__main__":
    main()
