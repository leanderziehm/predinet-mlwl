import argparse
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Import everything we need directly from g2.py so we reuse the exact
# same constants / feature engineering / forecasting logic. Nothing here
# redefines or changes what's already specified there.
# ----------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import g2
except ImportError:
    raise SystemExit(
        "Could not import g2.py. Place this script in the same directory "
        "as src/g2.py (or adjust sys.path) and re-run."
    )

TARGET_COL = g2.TARGET_COL
CELL_COL = g2.CELL_COL
TIME_COL = g2.TIME_COL
HORIZON = g2.HORIZON
QUANTILES = g2.QUANTILES
LAGS = g2.LAGS
ROLL_WINDOWS = g2.ROLL_WINDOWS
FEATURE_COLS_MODEL = g2.FEATURE_COLS_MODEL


def log(msg):
    print(msg)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------
# KPI DEFINITIONS
# ----------------------------------------------------------------------
def compute_kpis(y_true, q10, q50, q90, eps=1e-6):

    y_true = np.asarray(y_true, dtype=float)
    q10 = np.asarray(q10, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q90 = np.asarray(q90, dtype=float)

    nonzero_mask = np.abs(y_true) > eps
    n_zero = int((~nonzero_mask).sum())
    frac_zero = n_zero / len(y_true) if len(y_true) else np.nan

    if nonzero_mask.any():
        mape_raw = np.mean(
            np.abs(y_true[nonzero_mask] - q50[nonzero_mask]) / np.abs(y_true[nonzero_mask])
        ) * 100.0
    else:
        mape_raw = np.nan

    mape_safe = np.mean(np.abs(y_true - q50) / (np.abs(y_true) + eps)) * 100.0

    interval_range = np.mean(q90 - q10)
    coverage_pct = np.mean((y_true >= q10) & (y_true <= q90)) * 100.0

    return {
        "MAPE_raw_%": mape_raw,
        "MAPE_safe_%": mape_safe,
        "n_zero_actuals": n_zero,
        "frac_zero_actuals": frac_zero,
        "quantile_interval_range": interval_range,
        "coverage_%": coverage_pct,
    }


# ----------------------------------------------------------------------
# TASK A: selected cells from the ORIGINAL dataset (held-out last 3 days)
# ----------------------------------------------------------------------
def run_task_a(data_path, run_dir, selected_cells, out_dir):
    log(f"\n=== TASK A: original dataset, selected cells {selected_cells} ===")

    models_dir = os.path.join(run_dir, "models")
    tables_dir = os.path.join(run_dir, "tables")

    cluster_models = load_pickle(os.path.join(models_dir, "cluster_quantile_models.pkl"))
    cell_feats = pd.read_csv(
        os.path.join(tables_dir, "cell_cluster_assignments.csv"), index_col=CELL_COL
    )

    # Reuse g2.py's own loading/feature pipeline exactly as-is.
    df = g2.load_and_validate(data_path, log)
    df = g2.add_calendar_features(df)
    df = g2.add_lag_features(df, TARGET_COL)
    df = df.merge(cell_feats["cluster"].reset_index(), on=CELL_COL, how="left")

    missing = [c for c in selected_cells if c not in df[CELL_COL].unique()]
    if missing:
        log(f"  WARNING: these requested cells were not found in {data_path}: {missing}")

    per_cell_rows = []
    for cell in selected_cells:
        if cell not in df[CELL_COL].unique():
            continue
        g = df[df[CELL_COL] == cell].sort_values(TIME_COL).reset_index(drop=True)
        if len(g) <= HORIZON:
            log(f"  WARNING: {cell} has <= HORIZON rows, cannot hold out 3 days; skipping.")
            continue

        # Same train/test split convention as g2.py: last HORIZON rows = test.
        train_hist = g.iloc[:-HORIZON]
        test = g.iloc[-HORIZON:]
        cluster_id = train_hist["cluster"].iloc[-1]

        if cluster_id not in cluster_models:
            log(f"  WARNING: {cell} -> cluster {cluster_id} has no trained model; skipping.")
            continue

        y_true = test[TARGET_COL].values[:HORIZON]
        future_times, model_preds = g2.recursive_forecast_one_cell(
            train_hist, cluster_models[cluster_id], HORIZON, log_ctx=cell
        )

        kpis = compute_kpis(y_true, model_preds[0.10], model_preds[0.50], model_preds[0.90])
        row = {"cell": cell, "cluster": cluster_id, **kpis}
        per_cell_rows.append(row)
        log(f"  {cell} (cluster {cluster_id}): "
            f"MAPE_raw={kpis['MAPE_raw_%']:.2f}% "
            f"interval_range={kpis['quantile_interval_range']:.4f} "
            f"coverage={kpis['coverage_%']:.2f}%")

    per_cell_df = pd.DataFrame(per_cell_rows)
    per_cell_path = os.path.join(out_dir, "taskA_per_cell_kpis.csv")
    per_cell_df.to_csv(per_cell_path, index=False)

    kpi_cols = ["MAPE_raw_%", "MAPE_safe_%", "quantile_interval_range", "coverage_%"]

    if len(per_cell_df):
        overall_avg = per_cell_df[kpi_cols].mean().to_frame().T
        overall_avg.insert(0, "scope", "ALL_SELECTED_CELLS")
        overall_path = os.path.join(out_dir, "taskA_overall_avg_kpis.csv")
        overall_avg.to_csv(overall_path, index=False)

        per_cluster_avg = (
            per_cell_df.groupby("cluster")[kpi_cols].mean().reset_index()
        )
        per_cluster_path = os.path.join(out_dir, "taskA_per_cluster_avg_kpis.csv")
        per_cluster_avg.to_csv(per_cluster_path, index=False)

        log("\n  --- Task A: overall average across selected cells ---")
        log(overall_avg.to_string(index=False))
        log("\n  --- Task A: average per cluster (selected cells only) ---")
        log(per_cluster_avg.to_string(index=False))
    else:
        log("  WARNING: no valid cells evaluated for Task A.")

    return per_cell_df


# ----------------------------------------------------------------------
# TASK B: unseen dataset, no retraining
# ----------------------------------------------------------------------
def assign_unseen_cell_to_cluster(series, cluster_scaler, kmeans_model, cluster_feature_names, log):
    feats = g2.stl_summarize_cell(series)
    feats_df = pd.DataFrame([feats])[cluster_feature_names]
    X = cluster_scaler.transform(feats_df)
    cluster_id = int(kmeans_model.predict(X)[0])
    return cluster_id


def run_task_b(unseen_path, run_dir, out_dir):
    log(f"\n=== TASK B: unseen dataset ({unseen_path}), no retraining ===")

    models_dir = os.path.join(run_dir, "models")
    cluster_models = load_pickle(os.path.join(models_dir, "cluster_quantile_models.pkl"))
    cluster_scaler = load_pickle(os.path.join(models_dir, "cluster_scaler.pkl"))
    kmeans_model = load_pickle(os.path.join(models_dir, "kmeans_model.pkl"))
    cluster_feature_names = load_pickle(os.path.join(models_dir, "cluster_feature_names.pkl"))

    # Same loading/feature functions as g2.py -- nothing changed.
    df = g2.load_and_validate(unseen_path, log)
    df = g2.add_calendar_features(df)
    df = g2.add_lag_features(df, TARGET_COL)

    cells = sorted(df[CELL_COL].unique())
    log(f"  Found {len(cells)} unseen cells: {cells}")

    per_cell_rows = []
    for cell in cells:
        g = df[df[CELL_COL] == cell].sort_values(TIME_COL).reset_index(drop=True)
        if len(g) <= HORIZON:
            log(f"  WARNING: {cell} has <= {HORIZON} rows, cannot evaluate last 3 days; skipping.")
            continue

        # "Forecast the last 3 days (288 timestamps) included in the dataset"
        # -> same convention as elsewhere: last HORIZON rows = held-out test,
        # everything before is the history fed into the recursive forecaster.
        train_hist = g.iloc[:-HORIZON]
        test = g.iloc[-HORIZON:]

        cluster_id = assign_unseen_cell_to_cluster(
            train_hist[TARGET_COL], cluster_scaler, kmeans_model, cluster_feature_names, log
        )
        log(f"  {cell}: assigned to nearest existing cluster {cluster_id} (no retraining)")

        if cluster_id not in cluster_models:
            log(f"  WARNING: cluster {cluster_id} has no trained model; skipping {cell}.")
            continue

        y_true = test[TARGET_COL].values[:HORIZON]
        future_times, model_preds = g2.recursive_forecast_one_cell(
            train_hist, cluster_models[cluster_id], HORIZON, log_ctx=cell
        )

        kpis = compute_kpis(y_true, model_preds[0.10], model_preds[0.50], model_preds[0.90])
        row = {"cell": cell, "assigned_cluster": cluster_id, **kpis}
        per_cell_rows.append(row)
        log(f"  {cell} (cluster {cluster_id}): "
            f"MAPE_raw={kpis['MAPE_raw_%']:.2f}% "
            f"interval_range={kpis['quantile_interval_range']:.4f} "
            f"coverage={kpis['coverage_%']:.2f}%")

    per_cell_df = pd.DataFrame(per_cell_rows)
    per_cell_path = os.path.join(out_dir, "taskB_per_cell_kpis.csv")
    per_cell_df.to_csv(per_cell_path, index=False)

    kpi_cols = ["MAPE_raw_%", "MAPE_safe_%", "quantile_interval_range", "coverage_%"]

    if len(per_cell_df):
        overall_avg = per_cell_df[kpi_cols].mean().to_frame().T
        overall_avg.insert(0, "scope", "ALL_8_UNSEEN_CELLS")
        overall_path = os.path.join(out_dir, "taskB_overall_avg_kpis.csv")
        overall_avg.to_csv(overall_path, index=False)

        log("\n  --- Task B: overall average across the 8 unseen cells ---")
        log(overall_avg.to_string(index=False))
    else:
        log("  WARNING: no valid cells evaluated for Task B.")

    return per_cell_df


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="UL_PRB_data_set.csv",
                         help="Path to the original training CSV (same one g2.py used).")
    parser.add_argument("--unseen-data", default="selected_cells_unseen.csv",
                         help="Path to the unseen-cells CSV.")
    parser.add_argument("--run-dir", default="output/g2/20260720_164050",
                         help="Path to the g2.py run directory containing models/ and tables/.")
    parser.add_argument("--out-dir", default=None,
                         help="Where to write the report CSVs. Defaults to <run-dir>/eval_report.")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, "eval_report")
    os.makedirs(out_dir, exist_ok=True)

    selected_cells = ["cell021", "cell198", "cell192", "cell214"]

    task_a_df = run_task_a(args.data, args.run_dir, selected_cells, out_dir)
    task_b_df = run_task_b(args.unseen_data, args.run_dir, out_dir)

    log(f"\nAll KPI report CSVs written to: {out_dir}")
    log("Files:")
    for fn in sorted(os.listdir(out_dir)):
        log(f"  - {os.path.join(out_dir, fn)}")


if __name__ == "__main__":
    main()