import os
import pickle
import numpy as np
import pandas as pd
from g2 import (
    TARGET_COL,
    CELL_COL,
    TIME_COL,
    HORIZON,
    FEATURE_COLS_MODEL,
    add_calendar_features,
    add_lag_features,
    recursive_forecast_one_cell,
    stl_summarize_cell,
)

MODEL_DIR = "output/g2/20260720_164050/models"


def load_models():
    with open(f"{MODEL_DIR}/cluster_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(f"{MODEL_DIR}/cluster_quantile_models.pkl", "rb") as f:
        models = pickle.load(f)
    return scaler, kmeans, models


def compute_kpis(y, q10, q50, q90):
    eps = 1e-6
    mape = np.mean(np.abs(y - q50) / np.maximum(np.abs(y), eps)) * 100
    interval = np.mean(q90 - q10)
    coverage = np.mean((y >= q10) & (y <= q90)) * 100
    return {"MAPE_%": mape, "Mean_interval": interval, "Coverage_%": coverage}


def assign_cluster(cell_df, scaler, kmeans):
    feat = stl_summarize_cell(cell_df[TARGET_COL])
    X = pd.DataFrame([feat])
    X = X[scaler.feature_names_in_]
    X = scaler.transform(X)
    cluster = kmeans.predict(X)[0]
    return cluster


def evaluate_dataset(df, name):
    scaler, kmeans, models = load_models()
    results = []
    for cell, g in df.groupby(CELL_COL):
        g = g.sort_values(TIME_COL)
        cluster = assign_cluster(g, scaler, kmeans)
        train = g.iloc[:-HORIZON]
        test = g.iloc[-HORIZON:]
        if cluster not in models:
            continue
        future, preds = recursive_forecast_one_cell(train, models[cluster], HORIZON)
        y = test[TARGET_COL].values
        kpi = compute_kpis(y, preds[0.10], preds[0.50], preds[0.90])
        results.append({"dataset": name, "cell": cell, "cluster": cluster, **kpi})
    return pd.DataFrame(results)


# ===========================
# TRAIN DATA REQUIRED CELLS
# ===========================
train = pd.read_csv("UL_PRB_data_set.csv", parse_dates=[TIME_COL])
selected = ["cell021", "cell198", "cell192", "cell214"]
selected_df = train[train[CELL_COL].isin(selected)]
r1 = evaluate_dataset(selected_df, "training_selected")
# ===========================
# ALL TRAIN CELLS
# ===========================
r2 = evaluate_dataset(train, "training_all")
# ===========================
# UNSEEN CELLS
# ===========================
unseen = pd.read_csv("selected_cells_unseen.csv", parse_dates=[TIME_COL])
r3 = evaluate_dataset(unseen, "unseen")
results = pd.concat([r1, r2, r3], ignore_index=True)
os.makedirs("evaluation_report", exist_ok=True)
results.to_csv("evaluation_report/cell_results.csv", index=False)
# averages
summary = results.groupby("dataset")[["MAPE_%", "Mean_interval", "Coverage_%"]].mean()
summary.to_csv("evaluation_report/summary.csv")
# cluster report
cluster_summary = (
    results[results.dataset == "training_all"]
    .groupby("cluster")[["MAPE_%", "Mean_interval", "Coverage_%"]]
    .mean()
)
cluster_summary.to_csv("evaluation_report/cluster_summary.csv")
print("\nCELL RESULTS")
print(results)
print("\nSUMMARY")
print(summary)
print("\nCLUSTER SUMMARY")
print(cluster_summary)
