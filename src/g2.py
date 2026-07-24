import argparse
import os
import sys
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------
TARGET_COL = "N.PRB.UL.DrbUsed.Avg[%]"
FEATURE_COLS_RAW = [TARGET_COL, "N.ThpVol.UL", "N.User.RRCConn.Active.UL.Avg"]
CELL_COL = "Short name"
TIME_COL = "Date"

STEPS_PER_DAY = 96          # 15-min granularity -> 96 steps/day
HORIZON = 3 * STEPS_PER_DAY  # 288 steps = 3 days
QUANTILES = [0.10, 0.50, 0.90]
LAGS = [1, 2, 4, 8, 96, 96 * 7]   # 15min,30min,1h,2h,1day,1week lags
ROLL_WINDOWS = [4, 96]            # 1h, 1day rolling means


# ----------------------------------------------------------------------
# UTILITIES: output directory setup
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# STEP 1: LOAD + VALIDATE
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING (calendar + lags)
# ----------------------------------------------------------------------
def add_calendar_features(df):
    dt = df[TIME_COL].dt
    tod = dt.hour * 4 + dt.minute // 15         # 0..95 slot of day
    dow = dt.dayofweek                          # 0..6

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
    return df


# ----------------------------------------------------------------------
# STEP 3: CELL CLUSTERING (STL trend/seasonal/resid -> KMeans)
# ----------------------------------------------------------------------
def stl_summarize_cell(series, period=STEPS_PER_DAY):
    """Return summary descriptors of trend/seasonal/residual components."""
    from statsmodels.tsa.seasonal import STL

    s = series.values.astype(float)
    if len(s) < period * 2:
        # not enough data, pad by repeating
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


def cluster_cells(df, target_col, n_clusters, log):
    log("Running STL decomposition per cell to build clustering features "
        "(trend/seasonal/residual summaries)...")
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
    feature_names = [c for c in cell_feats.columns]

    scaler = RobustScaler()
    X = scaler.fit_transform(cell_feats[feature_names])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    cell_feats["cluster"] = labels
    log(f"Cluster sizes:\n{cell_feats['cluster'].value_counts().sort_index()}")
    return cell_feats, scaler, kmeans, feature_names


# ----------------------------------------------------------------------
# STEP 4: BASELINES
# ----------------------------------------------------------------------
def naive_last_value_forecast(history_last_value, horizon):
    """Repeat the last observed value for the whole horizon."""
    return np.full(horizon, history_last_value)


def seasonal_naive_forecast(history_series, horizon, period=STEPS_PER_DAY):
    """Repeat the last full day-cycle (or week if available) forward."""
    if len(history_series) >= period:
        last_cycle = history_series[-period:]
    else:
        last_cycle = history_series
    reps = int(np.ceil(horizon / len(last_cycle)))
    return np.tile(last_cycle, reps)[:horizon]


# ----------------------------------------------------------------------
# STEP 5: QUANTILE FORECASTING MODEL (per-cluster GBR, direct-block recursive)
# ----------------------------------------------------------------------
FEATURE_COLS_MODEL = (
    ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend"]
    + [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
)


def train_cluster_quantile_models(train_df, log):
    """One GradientBoostingRegressor per quantile, per cluster.
    Trained on 1-step-ahead lag/calendar features (tabular supervised
    learning). Kept deliberately shallow/few-estimators for CPU speed.
    """
    models = {}
    for cluster_id, cdf in train_df.groupby("cluster"):
        cdf = cdf.dropna(subset=FEATURE_COLS_MODEL + [TARGET_COL])
        if len(cdf) < 200:
            log(f"  Cluster {cluster_id}: too few rows ({len(cdf)}), skipping.")
            continue
        X = cdf[FEATURE_COLS_MODEL].values
        y = cdf[TARGET_COL].values

        cluster_models = {}
        for q in QUANTILES:
            gbr = GradientBoostingRegressor(
                loss="quantile", alpha=q,
                n_estimators=80, max_depth=3, learning_rate=0.08,
                subsample=0.8, random_state=42,
            )
            gbr.fit(X, y)
            cluster_models[q] = gbr
        models[cluster_id] = cluster_models
        log(f"  Cluster {cluster_id}: trained on {len(cdf):,} rows.")
    return models


def recursive_forecast_one_cell(cell_hist_df, models_for_cluster, horizon, log_ctx=""):
    """Roll the trained per-quantile models forward `horizon` steps.
    We recursively feed the q50 prediction back in as the 'observed'
    value for future lag features (standard practice for recursive
    multi-step forecasting with tabular models).
    """
    hist = cell_hist_df.copy().reset_index(drop=True)
    last_time = hist[TIME_COL].iloc[-1]
    future_times = pd.date_range(last_time + pd.Timedelta(minutes=15),
                                  periods=horizon, freq="15min")

    extended = hist[[TIME_COL, TARGET_COL]].copy()
    preds = {q: [] for q in QUANTILES}

    for t in future_times:
        row = {TIME_COL: t}
        tod = t.hour * 4 + t.minute // 15
        dow = t.dayofweek
        row["tod_sin"] = np.sin(2 * np.pi * tod / STEPS_PER_DAY)
        row["tod_cos"] = np.cos(2 * np.pi * tod / STEPS_PER_DAY)
        row["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        row["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        row["is_weekend"] = int(dow >= 5)

        vals = extended[TARGET_COL].values
        for lag in LAGS:
            row[f"lag_{lag}"] = vals[-lag] if len(vals) >= lag else vals[0]
        for w in ROLL_WINDOWS:
            row[f"roll_mean_{w}"] = vals[-w:].mean() if len(vals) >= w else vals.mean()

        X_row = np.array([[row[c] for c in FEATURE_COLS_MODEL]])

        step_preds = {}
        for q in QUANTILES:
            p = models_for_cluster[q].predict(X_row)[0]
            step_preds[q] = p
            preds[q].append(p)

        # feed q50 back in as pseudo-observation for next step's lags
        extended = pd.concat(
            [extended, pd.DataFrame({TIME_COL: [t], TARGET_COL: [step_preds[0.5]]})],
            ignore_index=True,
        )

    return future_times, {q: np.array(v) for q, v in preds.items()}


# ----------------------------------------------------------------------
# STEP 6: METRICS
# ----------------------------------------------------------------------
def compute_metrics(y_true, q10, q50, q90):
    mae = mean_absolute_error(y_true, q50)
    rmse = np.sqrt(mean_squared_error(y_true, q50))
    coverage = np.mean((y_true >= q10) & (y_true <= q90))
    mean_width = np.mean(q90 - q10)
    return {"MAE": mae, "RMSE": rmse, "coverage_10_90": coverage, "mean_interval_width": mean_width}


# ----------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="UL_PRB_data_set.csv")
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--n-eval-cells", type=int, default=15,
                         help="How many cells to run the full 288-step evaluation on "
                              "(evaluation is the expensive recursive part).")
    parser.add_argument("--sample-cells", type=int, default=0,
                         help="If >0, only use this many cells total (fast smoke test).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = make_output_dir(os.path.abspath(__file__) if "__file__" in globals() else "pipeline.py")
    log_path = os.path.join(out_dir, "run_log.txt")
    log_file = open(log_path, "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"Output directory: {out_dir}")
    np.random.seed(args.seed)

    # ---------------- STEP 1 ----------------
    df = load_and_validate(args.data, log)

    if args.sample_cells > 0:
        keep_cells = sorted(df[CELL_COL].unique())[: args.sample_cells]
        df = df[df[CELL_COL].isin(keep_cells)]
        log(f"Smoke-test mode: restricted to {len(keep_cells)} cells.")

    # ---------------- STEP 2 ----------------
    log("Adding calendar + lag features...")
    df = add_calendar_features(df)
    df = add_lag_features(df, TARGET_COL)

    # ---------------- STEP 3: CLUSTERING ----------------
    cell_feats, cluster_scaler, kmeans_model, cluster_feature_names = cluster_cells(
        df, TARGET_COL, args.n_clusters, log
    )
    df = df.merge(cell_feats["cluster"].reset_index(), on=CELL_COL, how="left")

    save_pickle(cluster_scaler, os.path.join(out_dir, "models", "cluster_scaler.pkl"))
    save_pickle(kmeans_model, os.path.join(out_dir, "models", "kmeans_model.pkl"))
    cell_feats.to_csv(os.path.join(out_dir, "tables", "cell_cluster_assignments.csv"))

    # ---------------- TRAIN / TEST SPLIT ----------------
    # Last HORIZON steps of each cell = held-out test (the 3-day forecast window).
    log("Splitting train/test (last 3 days per cell held out)...")
    df = df.sort_values([CELL_COL, TIME_COL])
    test_parts, train_parts = [], []
    for cell, g in df.groupby(CELL_COL):
        g = g.reset_index(drop=True)
        if len(g) <= HORIZON + max(LAGS) + 10:
            train_parts.append(g)
            continue
        train_parts.append(g.iloc[: -HORIZON])
        test_parts.append(g.iloc[-HORIZON:])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    log(f"Train rows: {len(train_df):,} | Test rows (held-out horizon): {len(test_df):,}")

    # ---------------- STEP 5: TRAIN QUANTILE MODELS PER CLUSTER ----------------
    log("Training per-cluster quantile regression models (GradientBoosting, q10/q50/q90)...")
    cluster_models = train_cluster_quantile_models(train_df, log)
    save_pickle(cluster_models, os.path.join(out_dir, "models", "cluster_quantile_models.pkl"))

    # ---------------- STEP 4 + 5 + 6: EVALUATE BASELINES + MODEL ----------------
    log(f"Evaluating baselines + model on up to {args.n_eval_cells} cells "
        f"(full {HORIZON}-step recursive forecast each)...")

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
        last_val = cell_train_hist[TARGET_COL].values[-1]
        hist_vals = cell_train_hist[TARGET_COL].values

        # Baselines
        naive_pred = naive_last_value_forecast(last_val, HORIZON)
        seasonal_pred = seasonal_naive_forecast(hist_vals, HORIZON)

        # crude baseline "intervals": +-0 width (point forecast) -> coverage undefined,
        # so we give baselines a fixed empirical residual band from train history for a
        # fair coverage comparison.
        resid_std = np.std(hist_vals[-STEPS_PER_DAY * 7:] - seasonal_naive_forecast(
            hist_vals[:-STEPS_PER_DAY * 7] if len(hist_vals) > STEPS_PER_DAY * 7 else hist_vals,
            STEPS_PER_DAY * 7)) if len(hist_vals) > STEPS_PER_DAY * 7 else np.std(hist_vals) * 0.5

        naive_q10, naive_q90 = naive_pred - 1.2816 * resid_std, naive_pred + 1.2816 * resid_std
        seas_q10, seas_q90 = seasonal_pred - 1.2816 * resid_std, seasonal_pred + 1.2816 * resid_std

        # Model
        future_times, model_preds = recursive_forecast_one_cell(
            cell_train_hist, cluster_models[cluster_id], HORIZON, log_ctx=cell
        )

        m_naive = compute_metrics(y_true, naive_q10, naive_pred, naive_q90)
        m_seas = compute_metrics(y_true, seas_q10, seasonal_pred, seas_q90)
        m_model = compute_metrics(y_true, model_preds[0.10], model_preds[0.50], model_preds[0.90])

        for name, m in [("naive_last_value", m_naive), ("seasonal_naive", m_seas), ("gbr_quantile_model", m_model)]:
            all_results.append({"cell": cell, "cluster": cluster_id, "method": name, **m})

        per_cell_curves[cell] = {
            "times": future_times, "y_true": y_true,
            "q10": model_preds[0.10], "q50": model_preds[0.50], "q90": model_preds[0.90],
            "seasonal_naive": seasonal_pred,
        }
        log(f"  {cell} (cluster {cluster_id}): "
            f"model MAE={m_model['MAE']:.4f} RMSE={m_model['RMSE']:.4f} "
            f"coverage={m_model['coverage_10_90']:.2f} | "
            f"seasonal_naive MAE={m_seas['MAE']:.4f}")

    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(out_dir, "tables", "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)

    if len(results_df):
        summary = results_df.groupby("method")[["MAE", "RMSE", "coverage_10_90", "mean_interval_width"]].mean()
        summary_path = os.path.join(out_dir, "tables", "evaluation_summary.csv")
        summary.to_csv(summary_path)
        log("=== SUMMARY (averaged across evaluated cells) ===")
        log("\n" + summary.to_string())

        per_cluster_summary = results_df.groupby(["cluster", "method"])[
            ["MAE", "RMSE", "coverage_10_90", "mean_interval_width"]
        ].mean()
        per_cluster_summary.to_csv(os.path.join(out_dir, "tables", "evaluation_summary_per_cluster.csv"))
        log("=== SUMMARY PER CLUSTER ===")
        log("\n" + per_cluster_summary.to_string())
    else:
        log("WARNING: no cells had enough history for full evaluation; "
            "try lowering --n-eval-cells or check dataset length.")

    # ---------------- PLOTS ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for cell, d in list(per_cell_curves.items())[:5]:
            plt.figure(figsize=(12, 4))
            plt.plot(d["times"], d["y_true"], label="actual", color="black", linewidth=1)
            plt.plot(d["times"], d["q50"], label="q50 (model)", color="tab:blue")
            plt.fill_between(d["times"], d["q10"], d["q90"], color="tab:blue", alpha=0.2, label="q10-q90")
            plt.plot(d["times"], d["seasonal_naive"], label="seasonal_naive", color="tab:orange", linestyle="--")
            plt.title(f"3-day forecast vs actual: {cell}")
            plt.xlabel("Time")
            plt.ylabel(TARGET_COL)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "plots", f"forecast_{cell}.png"), dpi=110)
            plt.close()

        if len(results_df):
            plt.figure(figsize=(7, 5))
            summary[["MAE", "RMSE"]].plot(kind="bar", ax=plt.gca())
            plt.title("MAE / RMSE by method")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "plots", "metric_comparison.png"), dpi=110)
            plt.close()
        log("Saved plots.")
    except Exception as e:
        log(f"Plotting skipped due to error: {e}")

    # ---------------- SAVE remaining artifacts ----------------
    save_pickle(FEATURE_COLS_MODEL, os.path.join(out_dir, "models", "feature_columns.pkl"))
    save_pickle(cluster_feature_names, os.path.join(out_dir, "models", "cluster_feature_names.pkl"))

    log(f"DONE. All artifacts saved under: {out_dir}")
    log_file.close()


if __name__ == "__main__":
    main()