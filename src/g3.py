"""
g3.py -- Multi-Cell Uplink PRB Utilization Forecasting (v3)

KEY DESIGN CHANGE FROM g2.py
-----------------------------
g2.py trains a single 1-step-ahead quantile model per cluster and generates
the 3-day (288-step) forecast RECURSIVELY: it feeds its own q50 prediction
back in as a pseudo-observation for the next step's lag features. This
causes two problems visible in the g2 results:
  1. Error compounds across 288 steps -> MAE/RMSE degrade with horizon.
  2. The quantile models were only ever trained on 1-step-ahead residual
     spread, so q10/q90 stay far too narrow at step 200+ -> coverage
     collapses (was measured at 12-55% instead of the target ~80%).

g3.py instead uses DIRECT multi-horizon forecasting (a standard, well
established alternative to recursive forecasting in time series ML,
sometimes called the "direct strategy" -- see e.g. Taieb & Hyndman 2012,
"Recursive and direct multi-step forecasting strategies"): for a set of
horizon blocks (e.g. steps 1-24, 25-96, 97-192, 193-288), we train a
SEPARATE quantile model per (cluster, horizon-block, quantile) that
predicts "PRB usage at lead time h" directly from features available at
forecast-origin time t. No recursion, no compounding, and each block's
quantile spread is learned from its OWN true error distribution at that
lead time -- so coverage should track the nominal 80% band much more
closely.

Model: LightGBM quantile regression (pinball loss), which is fast enough
to fully train per-cluster-per-block-per-quantile on CPU in this dataset's
size range, and generally outperforms sklearn's GradientBoostingRegressor
in both speed and accuracy on tabular data of this shape.

Everything else (data loading, validation, calendar features, STL-based
clustering) is kept consistent with g2.py so the two pipelines are
directly comparable in your report.
"""
import argparse
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------
TARGET_COL = "N.PRB.UL.DrbUsed.Avg[%]"
FEATURE_COLS_RAW = [TARGET_COL, "N.ThpVol.UL", "N.User.RRCConn.Active.UL.Avg"]
CELL_COL = "Short name"
TIME_COL = "Date"

STEPS_PER_DAY = 96
HORIZON = 3 * STEPS_PER_DAY  # 288 steps = 3 days
QUANTILES = [0.10, 0.50, 0.90]

LAGS = [1, 2, 4, 8, 96, 96 * 7]
ROLL_WINDOWS = [4, 96, 96 * 7]

HORIZON_BLOCKS = [
    (1, 24),
    (25, 96),
    (97, 192),
    (193, 288),
]


def make_output_dir(script_path):
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("output", script_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    for sub in ["models", "plots", "tables"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    return out_dir


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_and_validate(data_path, log):
    df = pd.read_csv(data_path, parse_dates=[TIME_COL])
    log(f"Loaded {len(df):,} rows, {df[CELL_COL].nunique()} cells.")
    n_missing = df.isna().sum().sum()
    n_dupes = df.duplicated(subset=[CELL_COL, TIME_COL]).sum()
    log(f"Missing values total: {n_missing} | Duplicate (cell,timestamp) rows: {n_dupes}")
    if n_missing > 0:
        df = df.sort_values([CELL_COL, TIME_COL])
        df[FEATURE_COLS_RAW] = (
            df.groupby(CELL_COL)[FEATURE_COLS_RAW].apply(lambda g: g.ffill().bfill())
        )
    if n_dupes > 0:
        df = df.drop_duplicates(subset=[CELL_COL, TIME_COL], keep="first")
    df = df.sort_values([CELL_COL, TIME_COL]).reset_index(drop=True)
    return df


def add_calendar_features(df):
    dt = df[TIME_COL].dt
    tod = dt.hour * 4 + dt.minute // 15
    dow = dt.dayofweek
    df["tod_sin"] = np.sin(2 * np.pi * tod / STEPS_PER_DAY)
    df["tod_cos"] = np.cos(2 * np.pi * tod / STEPS_PER_DAY)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
    return df


def add_lag_features(df, target_col=TARGET_COL):
    df = df.sort_values([CELL_COL, TIME_COL])
    g = df.groupby(CELL_COL)[target_col]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = g.shift(1).rolling(w).mean().reset_index(level=0, drop=True)
        df[f"roll_std_{w}"] = g.shift(1).rolling(w).std().reset_index(level=0, drop=True)
    return df


def add_direct_targets(df, target_col=TARGET_COL, lead_steps=None):
    """For direct multi-horizon forecasting, create one target column per
    individual lead step h (y_h{h} = target_col h steps ahead), computed
    per cell via a forward shift. Combined with target-step calendar
    features (added in build_training_rows_for_block), this lets each
    block's model resolve intra-day shape instead of collapsing to a
    flat block-average (a flat block-average badly underfits a strongly
    periodic signal like PRB utilization -- see HORIZON_BLOCKS lead steps)."""
    if lead_steps is None:
        lead_steps = sorted({h for (s, e) in HORIZON_BLOCKS for h in (s, e)}
                             | set(range(1, HORIZON + 1, 4)))  # dense enough grid
    df = df.sort_values([CELL_COL, TIME_COL])
    out_frames = []
    for cell, g in df.groupby(CELL_COL):
        g = g.copy()
        for h in lead_steps:
            g[f"y_lead_{h}"] = g[target_col].shift(-h).values
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


# Origin-time features: describe the state of the series AT the forecast
# origin (t=0). These do not change across lead steps.
ORIGIN_FEATURE_COLS = (
    ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend"]
    + [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
)

# Target-time features: describe calendar position AT the future lead
# time being predicted, plus the lead distance itself. Including these
# lets ONE model per (cluster, horizon-block) still resolve intra-day
# oscillation instead of collapsing to the block's flat average -- the
# model learns "given origin state X, what's the value at lead h with
# time-of-day Y", rather than "what's the average over 96 steps".
TARGET_TIME_FEATURE_COLS = ["lead_step", "target_tod_sin", "target_tod_cos",
                            "target_dow_sin", "target_dow_cos", "target_is_weekend"]

FEATURE_COLS_MODEL = ORIGIN_FEATURE_COLS + TARGET_TIME_FEATURE_COLS


def add_target_time_features(df, lead_steps, time_col=TIME_COL):
    """For each row (forecast origin) and each lead step h, compute the
    calendar features of the FUTURE timestamp origin_time + h*15min.
    Returns a long-format DataFrame: one row per (origin_row, lead_step),
    which is what we train each block's model on.
    """
    records = []
    dt = df[time_col]
    for h in lead_steps:
        future_time = dt + pd.Timedelta(minutes=15 * h)
        tod = future_time.dt.hour * 4 + future_time.dt.minute // 15
        dow = future_time.dt.dayofweek
        block = pd.DataFrame({
            "_orig_index": df.index,
            "lead_step": h,
            "target_tod_sin": np.sin(2 * np.pi * tod / STEPS_PER_DAY),
            "target_tod_cos": np.cos(2 * np.pi * tod / STEPS_PER_DAY),
            "target_dow_sin": np.sin(2 * np.pi * dow / 7),
            "target_dow_cos": np.cos(2 * np.pi * dow / 7),
            "target_is_weekend": (dow >= 5).astype(int),
            "y_target": df[f"y_lead_{h}"].values,
        })
        records.append(block)
    return pd.concat(records, ignore_index=True)


def stl_summarize_cell(series, period=STEPS_PER_DAY):
    from statsmodels.tsa.seasonal import STL
    s = series.values.astype(float)
    if len(s) < period * 2:
        reps = int(np.ceil((period * 2) / len(s))) + 1
        s = np.tile(s, reps)[: period * 2 + period]
    res = STL(s, period=period, robust=True).fit()
    trend, seasonal, resid = res.trend, res.seasonal, res.resid
    total_var = np.var(s) + 1e-9
    return {
        "mean_level": np.mean(s),
        "trend_slope": (trend[-1] - trend[0]) / max(len(trend), 1),
        "seasonal_strength": max(0.0, 1 - np.var(resid) / (np.var(seasonal + resid) + 1e-9)),
        "trend_strength": max(0.0, 1 - np.var(resid) / (np.var(trend + resid) + 1e-9)),
        "resid_var_ratio": np.var(resid) / total_var,
        "cv": np.std(s) / (np.mean(s) + 1e-6),
    }


def select_best_k(X, k_range, log):
    best_k, best_score, scores = None, -1, {}
    n_samples = X.shape[0]
    for k in k_range:
        if k < 2 or k > n_samples - 1:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        score = silhouette_score(X, km.labels_)
        scores[k] = score
        log(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    if best_k is None:
        best_k = min(3, max(2, n_samples - 1))
        log(f"  No valid k in sweep range for n_samples={n_samples}; "
            f"falling back to k={best_k}.")
    else:
        log(f"  Selected k={best_k} (silhouette={best_score:.4f})")
    return best_k, scores


def cluster_cells(df, target_col, log, k_range=range(3, 9)):
    log("Running STL decomposition per cell to build clustering features...")
    records = []
    cells = sorted(df[CELL_COL].unique())
    for i, cell in enumerate(cells):
        series = df.loc[df[CELL_COL] == cell, target_col]
        try:
            feats = stl_summarize_cell(series)
        except Exception as e:
            log(f"  STL failed for {cell} ({e}); using fallback flat stats.")
            feats = {
                "mean_level": series.mean(), "trend_slope": 0.0,
                "seasonal_strength": 0.0, "trend_strength": 0.0,
                "resid_var_ratio": 1.0, "cv": series.std() / (series.mean() + 1e-6),
            }
        feats[CELL_COL] = cell
        records.append(feats)
        if (i + 1) % 50 == 0:
            log(f"  ...{i + 1}/{len(cells)} cells decomposed")
    cell_feats = pd.DataFrame(records).set_index(CELL_COL)

    cell_feats["mean_level_w"] = cell_feats["mean_level"]
    cell_feats["cv_w"] = cell_feats["cv"]
    feature_names = [c for c in cell_feats.columns]

    scaler = RobustScaler()
    X = scaler.fit_transform(cell_feats[feature_names])

    log("Selecting k via silhouette score sweep...")
    best_k, sweep_scores = select_best_k(X, k_range, log)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    cell_feats["cluster"] = labels
    log(f"Cluster sizes:\n{cell_feats['cluster'].value_counts().sort_index()}")
    return cell_feats, scaler, kmeans, feature_names, sweep_scores


def train_direct_quantile_models(train_df, log, horizon_blocks=HORIZON_BLOCKS,
                                  quantiles=QUANTILES):
    """
    One LightGBM quantile model per (cluster, horizon-block, quantile),
    trained on a LONG table: each row is (origin features, target-time
    calendar features, lead_step) -> y_target. Because target-time
    calendar features (target_tod_sin/cos etc.) and lead_step are inputs,
    a single block model still resolves intra-day oscillation instead of
    collapsing to the block's flat average.
    """
    models = {}
    for cluster_id, cdf in train_df.groupby("cluster"):
        cdf = cdf.dropna(subset=ORIGIN_FEATURE_COLS)
        block_models = {}
        for (start, end) in horizon_blocks:
            lead_steps = list(range(start, end + 1, 4))  # every hour within block
            if lead_steps[-1] != end:
                lead_steps.append(end)
            long_df = add_target_time_features(cdf, lead_steps)
            # merge back origin features by position
            origin_feats = cdf[ORIGIN_FEATURE_COLS].reset_index(drop=True)
            origin_feats["_orig_index"] = cdf.index.values
            long_df = long_df.merge(origin_feats, on="_orig_index", how="left")
            long_df = long_df.dropna(subset=FEATURE_COLS_MODEL + ["y_target"])
            if len(long_df) < 500:
                log(f"  Cluster {cluster_id} block {start}-{end}: "
                    f"too few rows ({len(long_df)}), skipping.")
                continue
            X = long_df[FEATURE_COLS_MODEL].values
            y = long_df["y_target"].values
            q_models = {}
            for q in quantiles:
                model = lgb.LGBMRegressor(
                    objective="quantile", alpha=q,
                    n_estimators=200, max_depth=5, num_leaves=31,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    min_child_samples=30, random_state=42, verbosity=-1,
                )
                model.fit(X, y)
                q_models[q] = model
            block_models[(start, end)] = q_models
            log(f"  Cluster {cluster_id} block {start}-{end}: "
                f"trained on {len(long_df):,} rows ({len(lead_steps)} lead steps).")
        models[cluster_id] = block_models
    return models


def _find_block_for_step(h, horizon_blocks):
    for (start, end) in horizon_blocks:
        if start <= h <= end:
            return (start, end)
    return None


def forecast_direct(origin_row, origin_time, models_for_cluster,
                     horizon_blocks=HORIZON_BLOCKS, quantiles=QUANTILES,
                     horizon=HORIZON):
    """
    Predicts every one of the `horizon` future steps individually, using
    each step's own block model but feeding in that step's actual future
    calendar features (target_tod_sin/cos, lead_step, etc.) -- so the
    intra-day/weekly shape is preserved rather than flattened per block.
    """
    origin_vals = origin_row[ORIGIN_FEATURE_COLS].values.astype(float)
    preds = {q: np.zeros(horizon) for q in quantiles}

    # group steps by block so we can batch-predict per block (much faster
    # than one .predict() call per step)
    steps_by_block = {}
    for h in range(1, horizon + 1):
        block = _find_block_for_step(h, horizon_blocks)
        if block is None:
            continue
        steps_by_block.setdefault(block, []).append(h)

    for block, steps in steps_by_block.items():
        if block not in models_for_cluster:
            continue
        steps_arr = np.array(steps)
        future_times = origin_time + pd.to_timedelta(steps_arr * 15, unit="m")
        tod = future_times.hour * 4 + future_times.minute // 15
        dow = future_times.dayofweek
        n = len(steps_arr)
        X = np.column_stack([
            np.tile(origin_vals, (n, 1)),
            steps_arr,
            np.sin(2 * np.pi * tod / STEPS_PER_DAY),
            np.cos(2 * np.pi * tod / STEPS_PER_DAY),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            (dow >= 5).astype(int),
        ])
        for q in quantiles:
            p = models_for_cluster[block][q].predict(X)
            preds[q][steps_arr - 1] = p

    q10, q50, q90 = preds[0.10], preds[0.50], preds[0.90]
    lo = np.minimum(np.minimum(q10, q50), q90)
    hi = np.maximum(np.maximum(q10, q50), q90)
    mid = q10 + q50 + q90 - lo - hi
    preds[0.10], preds[0.50], preds[0.90] = lo, mid, hi
    return preds


def build_origin_features_for_cell(cell_hist_df, feature_cols=None):
    if feature_cols is None:
        feature_cols = ORIGIN_FEATURE_COLS
    last_row = cell_hist_df.iloc[-1]
    missing = [c for c in feature_cols if c not in cell_hist_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return last_row[feature_cols]


def compute_metrics(y_true, q10, q50, q90):
    mae = mean_absolute_error(y_true, q50)
    rmse = np.sqrt(mean_squared_error(y_true, q50))
    coverage = np.mean((y_true >= q10) & (y_true <= q90))
    mean_width = np.mean(q90 - q10)
    return {"MAE": mae, "RMSE": rmse, "coverage_10_90": coverage,
            "mean_interval_width": mean_width}


def compute_kpis(y_true, q10, q50, q90, eps=1e-6):
    y_true = np.asarray(y_true, dtype=float)
    q10, q50, q90 = (np.asarray(a, dtype=float) for a in (q10, q50, q90))
    nonzero_mask = np.abs(y_true) > eps
    active_mask = np.abs(y_true) > 5.0
    n_zero = int((~nonzero_mask).sum())
    mape_raw = (np.mean(np.abs(y_true[nonzero_mask] - q50[nonzero_mask])
                         / np.abs(y_true[nonzero_mask])) * 100.0
                if nonzero_mask.any() else np.nan)
    mape_safe = np.mean(np.abs(y_true - q50) / (np.abs(y_true) + eps)) * 100.0
    mape_active = (np.mean(np.abs(y_true[active_mask] - q50[active_mask])
                            / np.abs(y_true[active_mask])) * 100.0
                   if active_mask.any() else np.nan)
    return {
        "MAPE_raw_%": mape_raw,
        "MAPE_safe_%": mape_safe,
        "MAPE_active_%": mape_active,
        "n_zero_actuals": n_zero,
        "quantile_interval_range": np.mean(q90 - q10),
        "coverage_%": np.mean((y_true >= q10) & (y_true <= q90)) * 100.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="UL_PRB_data_set.csv")
    parser.add_argument("--n-eval-cells", type=int, default=15)
    parser.add_argument("--sample-cells", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = make_output_dir(os.path.abspath(__file__) if "__file__" in globals() else "g3.py")
    log_path = os.path.join(out_dir, "run_log.txt")
    log_file = open(log_path, "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"Output directory: {out_dir}")
    np.random.seed(args.seed)

    df = load_and_validate(args.data, log)
    if args.sample_cells > 0:
        keep_cells = sorted(df[CELL_COL].unique())[: args.sample_cells]
        df = df[df[CELL_COL].isin(keep_cells)]
        log(f"Smoke-test mode: restricted to {len(keep_cells)} cells.")

    log("Adding calendar + lag + direct multi-horizon target features...")
    df = add_calendar_features(df)
    df = add_lag_features(df, TARGET_COL)
    df = add_direct_targets(df, TARGET_COL)

    cell_feats, cluster_scaler, kmeans_model, cluster_feature_names, sweep_scores = \
        cluster_cells(df, TARGET_COL, log)
    df = df.merge(cell_feats["cluster"].reset_index(), on=CELL_COL, how="left")
    save_pickle(cluster_scaler, os.path.join(out_dir, "models", "cluster_scaler.pkl"))
    save_pickle(kmeans_model, os.path.join(out_dir, "models", "kmeans_model.pkl"))
    cell_feats.to_csv(os.path.join(out_dir, "tables", "cell_cluster_assignments.csv"))
    pd.Series(sweep_scores, name="silhouette").to_csv(
        os.path.join(out_dir, "tables", "silhouette_k_sweep.csv")
    )

    log("Splitting train/test (last 3 days per cell held out)...")
    df = df.sort_values([CELL_COL, TIME_COL])
    test_parts, train_parts = [], []
    for cell, g in df.groupby(CELL_COL):
        g = g.reset_index(drop=True)
        max_lag = max(LAGS)
        if len(g) <= HORIZON + max_lag + 10:
            train_parts.append(g)
            continue
        train_parts.append(g.iloc[: -HORIZON])
        test_parts.append(g.iloc[-HORIZON:])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    log(f"Train rows: {len(train_df):,} | Test rows (held-out horizon): {len(test_df):,}")

    log("Training per-cluster, per-horizon-block, per-quantile LightGBM models...")
    cluster_models = train_direct_quantile_models(train_df, log)
    save_pickle(cluster_models, os.path.join(out_dir, "models", "cluster_direct_models.pkl"))

    log(f"Evaluating on up to {args.n_eval_cells} cells (direct 288-step forecast)...")
    eval_cells = sorted(train_df[CELL_COL].unique())[: args.n_eval_cells]
    all_results = []
    per_cell_curves = {}

    for cell in eval_cells:
        cell_train_hist = train_df[train_df[CELL_COL] == cell].sort_values(TIME_COL)
        cell_test = test_df[test_df[CELL_COL] == cell].sort_values(TIME_COL) if len(test_df) else pd.DataFrame()
        if len(cell_test) < HORIZON:
            continue
        cluster_id = cell_train_hist["cluster"].iloc[-1]
        if cluster_id not in cluster_models:
            continue

        y_true = cell_test[TARGET_COL].values[:HORIZON]
        origin_row = build_origin_features_for_cell(cell_train_hist, ORIGIN_FEATURE_COLS)
        origin_time = cell_train_hist[TIME_COL].iloc[-1]
        preds = forecast_direct(origin_row, origin_time, cluster_models[cluster_id])

        m_model = compute_metrics(y_true, preds[0.10], preds[0.50], preds[0.90])
        kpis = compute_kpis(y_true, preds[0.10], preds[0.50], preds[0.90])
        all_results.append({"cell": cell, "cluster": cluster_id, "method": "lgbm_direct_multihorizon",
                             **m_model, **kpis})

        future_times = pd.date_range(
            cell_train_hist[TIME_COL].iloc[-1] + pd.Timedelta(minutes=15),
            periods=HORIZON, freq="15min",
        )
        per_cell_curves[cell] = {
            "times": future_times, "y_true": y_true,
            "q10": preds[0.10], "q50": preds[0.50], "q90": preds[0.90],
        }
        log(f"  {cell} (cluster {cluster_id}): MAE={m_model['MAE']:.4f} "
            f"RMSE={m_model['RMSE']:.4f} coverage={m_model['coverage_10_90']:.2f} "
            f"MAPE_active={kpis['MAPE_active_%']:.2f}%")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(out_dir, "tables", "evaluation_results.csv"), index=False)

    if len(results_df):
        summary_cols = ["MAE", "RMSE", "coverage_10_90", "mean_interval_width",
                         "MAPE_active_%", "MAPE_safe_%"]
        summary = results_df[summary_cols].mean().to_frame().T
        summary.to_csv(os.path.join(out_dir, "tables", "evaluation_summary.csv"), index=False)
        log("=== SUMMARY (averaged across evaluated cells) ===")
        log("\n" + summary.to_string(index=False))

        per_cluster_summary = results_df.groupby("cluster")[summary_cols].mean()
        per_cluster_summary.to_csv(os.path.join(out_dir, "tables", "evaluation_summary_per_cluster.csv"))
        log("=== SUMMARY PER CLUSTER ===")
        log("\n" + per_cluster_summary.to_string())
    else:
        log("WARNING: no cells had enough history for full evaluation.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for cell, d in list(per_cell_curves.items())[:5]:
            plt.figure(figsize=(12, 4))
            plt.plot(d["times"], d["y_true"], label="actual", color="black", linewidth=1)
            plt.plot(d["times"], d["q50"], label="q50 (model)", color="tab:blue")
            plt.fill_between(d["times"], d["q10"], d["q90"], color="tab:blue",
                              alpha=0.2, label="q10-q90")
            plt.title(f"g3 direct multi-horizon forecast vs actual: {cell}")
            plt.xlabel("Time")
            plt.ylabel(TARGET_COL)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "plots", f"forecast_{cell}.png"), dpi=110)
            plt.close()

        if sweep_scores:
            plt.figure(figsize=(6, 4))
            pd.Series(sweep_scores).plot(kind="bar")
            plt.title("Silhouette score by k (cluster selection)")
            plt.xlabel("k")
            plt.ylabel("silhouette score")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "plots", "silhouette_sweep.png"), dpi=110)
            plt.close()
        log("Saved plots.")
    except Exception as e:
        log(f"Plotting skipped due to error: {e}")

    save_pickle(FEATURE_COLS_MODEL, os.path.join(out_dir, "models", "feature_columns.pkl"))
    save_pickle(cluster_feature_names, os.path.join(out_dir, "models", "cluster_feature_names.pkl"))
    save_pickle(HORIZON_BLOCKS, os.path.join(out_dir, "models", "horizon_blocks.pkl"))
    log(f"DONE. All artifacts saved under: {out_dir}")
    log_file.close()


if __name__ == "__main__":
    main()