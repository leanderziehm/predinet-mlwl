import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================================
# CONFIG
# ==========================================================
PREVIOUS_OUTPUT = "output/C1_quantile_forecasting"
LOOKBACK = 96
HORIZON = 288
VAL_RATIO = 0.2


# ==========================================================
# METRICS
# ==========================================================
def quantile_loss(pred, target, q):
    error = target - pred
    return np.mean(np.maximum(q * error, (q - 1) * error))


def evaluate(name, pred, true):
    mae = mean_absolute_error(true.flatten(), pred.flatten())
    rmse = np.sqrt(mean_squared_error(true.flatten(), pred.flatten()))
    # fake quantiles:
    # baseline produces deterministic forecast
    qloss = (
        quantile_loss(pred, true, 0.1)
        + quantile_loss(pred, true, 0.5)
        + quantile_loss(pred, true, 0.9)
    ) / 3
    return {"model": name, "MAE": mae, "RMSE": rmse, "Quantile_loss": qloss}


# ==========================================================
# BASELINES
# ==========================================================
def persistence_baseline(X):
    """
    Predict future as last observed value
    """
    last_value = X[:, -1, 0]
    pred = np.repeat(last_value[:, None], HORIZON, axis=1)
    return pred


def daily_seasonal_baseline(X):
    """
    Repeat previous day pattern
    96 points/day
    Forecast = 3 x previous day
    """
    last_day = X[:, -LOOKBACK:, 0]
    pred = np.tile(last_day, (1, 3))
    return pred


def weekly_seasonal_baseline(X):
    """
    Same period last week
    Need 672 points history
    Since current X only has 96 points,
    this baseline requires raw sequences
    with longer lookback.
    """
    raise NotImplementedError("Need LOOKBACK >= 672 for weekly baseline")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("Loading data...")
    X = np.load(os.path.join(PREVIOUS_OUTPUT, "preprocess", "X.npy"))
    y = np.load(os.path.join(PREVIOUS_OUTPUT, "preprocess", "y.npy"))
    print("Samples:", len(X))
    # same split as training
    split = int(len(X) * (1 - VAL_RATIO))
    X_val = X[split:]
    y_val = y[split:]
    print("Validation samples:", len(X_val))
    results = []
    # -----------------------------
    # Persistence
    # -----------------------------
    print("Running persistence...")
    pred = persistence_baseline(X_val)
    results.append(evaluate("Persistence", pred, y_val))
    # -----------------------------
    # Daily seasonal
    # -----------------------------
    print("Running daily seasonal...")
    pred = daily_seasonal_baseline(X_val)
    results.append(evaluate("Daily seasonal", pred, y_val))
    # -----------------------------
    # Save
    # -----------------------------
    df = pd.DataFrame(results)
    print("\nRESULTS")
    print(df.sort_values("Quantile_loss").to_string(index=False))
    os.makedirs("output/C3_baselines", exist_ok=True)
    df.to_csv("output/C3_baselines/results.csv", index=False)
    print("\nSaved:" " output/C3_baselines/results.csv")


if __name__ == "__main__":
    main()

# user@debian:~/dev/predinet-mlwl$  source /home/user/dev/predinet-mlwl/.venv/bin/activate
# (predinet-github) user@debian:~/dev/predinet-mlwl$ /home/user/dev/predinet-mlwl/.venv/bin/python /home/user/dev/predinet-mlwl/src/C1_baselines.py
# Loading data...
# Samples: 1063104
# Validation samples: 212621
# Running persistence...
# Running daily seasonal...

# RESULTS
#          model      MAE     RMSE  Quantile_loss
# Daily seasonal 0.229033 0.714885       0.114516
#    Persistence 0.321199 0.832264       0.160599

# Saved: output/C3_baselines/results.csv
# (predinet-github) user@debian:~/dev/predinet-mlwl$ 