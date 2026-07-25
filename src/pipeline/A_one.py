import os
from datetime import datetime
import numpy as np
import pandas as pd
from pipeline_common import (
    DATA_PATH,
    SERIES_COL,
    extract_b_features_from_dataframe,
    load_time_series_dataframe,
    set_random_seeds,
)


# ==========================================================
# OUTPUT
# ==========================================================
def make_output_dir():
    out_dir = os.path.join(
        "output",
        "PA_one",
    )
    os.makedirs(
        os.path.join(
            out_dir,
            "tables",
        ),
        exist_ok=True,
    )
    return out_dir


# ==========================================================
# MAIN
# ==========================================================
def main():
    set_random_seeds()
    out_dir = make_output_dir()
    log_path = os.path.join(
        out_dir,
        "run_log.txt",
    )
    log_file = open(
        log_path,
        "w",
    )

    def log(message):
        line = f"[{datetime.now()}] " f"{message}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------
    log("Loading dataset")
    df = load_time_series_dataframe(DATA_PATH)
    log(f"Loaded " f"{df[SERIES_COL].nunique()} cells")
    # ------------------------------------------------------
    # Shared B feature extraction
    # ------------------------------------------------------
    log("Extracting forecastability features")
    feature_df = extract_b_features_from_dataframe(df)
    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------
    output_file = os.path.join(
        out_dir,
        "tables",
        "cell_forecastability_features.csv",
    )
    feature_df.to_csv(
        output_file,
        index=False,
    )
    summary = feature_df.describe().T
    summary.to_csv(
        os.path.join(
            out_dir,
            "tables",
            "feature_summary.csv",
        )
    )
    log(f"Saved feature table: " f"{output_file}")
    log_file.close()


if __name__ == "__main__":
    main()
