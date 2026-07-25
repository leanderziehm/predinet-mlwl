from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path("degs/PE_five")
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
REPORT = BASE / "C3_report.tex"

FIGURES.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def load_csv(name):
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)
    print(f"Loaded {path}: {df.shape}")
    return df


def latex_escape(value):
    """Escape characters that have special meaning in LaTeX."""
    if pd.isna(value):
        return ""

    value = str(value)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return value


def format_number(value, decimals=2):
    if pd.isna(value):
        return "--"

    value = float(value)

    # Scientific notation for extremely large values.
    if abs(value) >= 1_000_000:
        return f"{value:.2e}"

    return f"{value:.{decimals}f}"


def dataframe_to_latex(
    df,
    columns,
    headers,
    number_formats=None,
    column_spec=None,
):
    """
    Create a clean LaTeX table manually rather than relying on
    pandas' default formatting.
    """

    if number_formats is None:
        number_formats = {}

    if column_spec is None:
        column_spec = "l" * len(columns)

    lines = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        rf"\begin{{tabular}}{{{column_spec}}}"
    )
    lines.append(r"\toprule")

    lines.append(
        " & ".join(headers) + r" \\"
    )

    lines.append(r"\midrule")

    for _, row in df.iterrows():
        values = []

        for col in columns:
            value = row[col]

            if col in number_formats:
                decimals = number_formats[col]
                values.append(format_number(value, decimals))
            else:
                values.append(latex_escape(value))

        lines.append(
            " & ".join(values) + r" \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def save_figure(fig, filename):
    path = FIGURES / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated figure: {path}")
    return path


def figure_latex(filename, caption, label, width="0.90\\textwidth"):
    return rf"""
\begin{{figure}}[htbp]
    \centering
    \includegraphics[width={width}]{{figures/{filename}}}
    \caption{{{caption}}}
    \label{{fig:{label}}}
\end{{figure}}
"""


# ============================================================
# LOAD DATA
# ============================================================

original_cells = load_csv("original_cell_results.csv")
original_average = load_csv("original_all_cells_average.csv")
original_clusters = load_csv("original_cluster_average.csv")

unseen_cells = load_csv("unseen_cell_results.csv")
unseen_average = load_csv("unseen_all_cells_average.csv")
unseen_clusters = load_csv("unseen_cluster_average.csv")


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

for df in [
    original_cells,
    original_average,
    original_clusters,
    unseen_cells,
    unseen_average,
    unseen_clusters,
]:
    df.columns = [str(c).strip() for c in df.columns]


# ============================================================
# BASIC INFORMATION
# ============================================================

original_cell_names = list(original_cells["cell"].unique())
unseen_cell_names = list(unseen_cells["cell"].unique())

n_original = len(original_cell_names)
n_unseen = len(unseen_cell_names)

horizon = int(original_cells["horizon"].iloc[0])

print()
print("============================================================")
print("C3 REPORT GENERATION")
print("============================================================")
print(f"Original cells: {n_original}")
print(f"Unseen cells:   {n_unseen}")
print(f"Horizon:        {horizon} timestamps")
print(f"Report:         {REPORT}")
print()


# ============================================================
# TABLE 1: ORIGINAL PER-CELL RESULTS
# ============================================================

original_table = original_cells.copy()

original_table["MAPE_percent"] = pd.to_numeric(
    original_table["MAPE_percent"]
)

original_table["mean_quantile_interval_range"] = pd.to_numeric(
    original_table["mean_quantile_interval_range"]
)

original_table["coverage_percent"] = pd.to_numeric(
    original_table["coverage_percent"]
)

original_table = original_table.sort_values(
    ["cell", "model"]
)

original_latex = dataframe_to_latex(
    original_table,
    columns=[
        "cell",
        "cluster",
        "model",
        "MAPE_percent",
        "mean_quantile_interval_range",
        "coverage_percent",
    ],
    headers=[
        "Cell",
        "Cluster",
        "Model",
        "MAPE (\%)",
        "QI range",
        "Coverage (\%)",
    ],
    number_formats={
        "MAPE_percent": 2,
        "mean_quantile_interval_range": 3,
        "coverage_percent": 2,
    },
    column_spec="llrrrr",
)


# ============================================================
# TABLE 2: ORIGINAL OVERALL AVERAGE
# ============================================================

original_avg_latex = dataframe_to_latex(
    original_average,
    columns=[
        "number_of_cells",
        "MAPE_percent",
        "mean_quantile_interval_range",
        "coverage_percent",
    ],
    headers=[
        "Number of cells",
        "MAPE (\%)",
        "QI range",
        "Coverage (\%)",
    ],
    number_formats={
        "number_of_cells": 0,
        "MAPE_percent": 2,
        "mean_quantile_interval_range": 3,
        "coverage_percent": 2,
    },
    column_spec="rrrr",
)


# ============================================================
# TABLE 3: ORIGINAL CLUSTER RESULTS
# ============================================================

original_cluster_latex = dataframe_to_latex(
    original_clusters.sort_values("cluster"),
    columns=[
        "cluster",
        "number_of_cells",
        "MAPE_percent",
        "mean_quantile_interval_range",
        "coverage_percent",
    ],
    headers=[
        "Cluster",
        "Cells",
        "MAPE (\%)",
        "QI range",
        "Coverage (\%)",
    ],
    number_formats={
        "cluster": 0,
        "number_of_cells": 0,
        "MAPE_percent": 2,
        "mean_quantile_interval_range": 3,
        "coverage_percent": 2,
    },
    column_spec="rrrrr",
)


# ============================================================
# TABLE 4: UNSEEN PER-CELL RESULTS
# ============================================================

unseen_table = unseen_cells.copy()

unseen_table["MAPE_percent"] = pd.to_numeric(
    unseen_table["MAPE_percent"]
)

unseen_table["mean_quantile_interval_range"] = pd.to_numeric(
    unseen_table["mean_quantile_interval_range"]
)

unseen_table["coverage_percent"] = pd.to_numeric(
    unseen_table["coverage_percent"]
)

unseen_table = unseen_table.sort_values(
    ["cell", "model"]
)

unseen_latex = dataframe_to_latex(
    unseen_table,
    columns=[
        "cell",
        "cluster",
        "model",
        "MAPE_percent",
        "mean_quantile_interval_range",
        "coverage_percent",
    ],
    headers=[
        "Cell",
        "Cluster",
        "Model",
        "MAPE (\%)",
        "QI range",
        "Coverage (\%)",
    ],
    number_formats={
        "MAPE_percent": 2,
        "mean_quantile_interval_range": 3,
        "coverage_percent": 2,
    },
    column_spec="llrrrr",
)


# ============================================================
# TABLE 5: UNSEEN OVERALL AVERAGE
# ============================================================

unseen_avg_latex = dataframe_to_latex(
    unseen_average,
    columns=[
        "number_of_cells",
        "MAPE_percent",
        "mean_quantile_interval_range",
        "coverage_percent",
    ],
    headers=[
        "Number of cells",
        "MAPE (\%)",
        "QI range",
        "Coverage (\%)",
    ],
    number_formats={
        "number_of_cells": 0,
        "MAPE_percent": 2,
        "mean_quantile_interval_range": 3,
        "coverage_percent": 2,
    },
    column_spec="rrrr",
)


# ============================================================
# TABLE 6: UNSEEN CLUSTER RESULTS
# ============================================================

unseen_cluster_latex = dataframe_to_latex(
    unseen_clusters.sort_values("cluster"),
    columns=[
        "cluster",
        "number_of_cells",
        "MAPE_percent",
        "mean_quantile_interval_range",
        "coverage_percent",
    ],
    headers=[
        "Cluster",
        "Cells",
        "MAPE (\%)",
        "QI range",
        "Coverage (\%)",
    ],
    number_formats={
        "cluster": 0,
        "number_of_cells": 0,
        "MAPE_percent": 2,
        "mean_quantile_interval_range": 3,
        "coverage_percent": 2,
    },
    column_spec="rrrrr",
)


# ============================================================
# FIGURE 1: ORIGINAL MAPE BY CELL
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

pivot = original_cells.pivot(
    index="cell",
    columns="model",
    values="MAPE_percent",
)

pivot.plot(kind="bar", ax=ax)

ax.set_title("Original Dataset: MAPE by Cell")
ax.set_xlabel("Cell")
ax.set_ylabel("MAPE (%)")
ax.set_yscale("log")
ax.grid(axis="y", alpha=0.25)
ax.legend(title="Model")

save_figure(fig, "original_mape_by_cell.png")


# ============================================================
# FIGURE 2: UNSEEN MAPE BY CELL
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

pivot = unseen_cells.pivot(
    index="cell",
    columns="model",
    values="MAPE_percent",
)

pivot.plot(kind="bar", ax=ax)

ax.set_title("Unseen Dataset: MAPE by Cell")
ax.set_xlabel("Cell")
ax.set_ylabel("MAPE (%)")
ax.set_yscale("log")
ax.grid(axis="y", alpha=0.25)
ax.legend(title="Model")

save_figure(fig, "unseen_mape_by_cell.png")


# ============================================================
# FIGURE 3: ORIGINAL COVERAGE
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

pivot = original_cells.pivot(
    index="cell",
    columns="model",
    values="coverage_percent",
)

pivot.plot(kind="bar", ax=ax)

ax.set_title("Original Dataset: Prediction Interval Coverage")
ax.set_xlabel("Cell")
ax.set_ylabel("Coverage (%)")
ax.set_ylim(0, 100)
ax.axhline(90, linestyle="--", linewidth=1)
ax.grid(axis="y", alpha=0.25)
ax.legend(title="Model")

save_figure(fig, "original_coverage_by_cell.png")


# ============================================================
# FIGURE 4: UNSEEN COVERAGE
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

pivot = unseen_cells.pivot(
    index="cell",
    columns="model",
    values="coverage_percent",
)

pivot.plot(kind="bar", ax=ax)

ax.set_title("Unseen Dataset: Prediction Interval Coverage")
ax.set_xlabel("Cell")
ax.set_ylabel("Coverage (%)")
ax.set_ylim(0, 100)
ax.axhline(90, linestyle="--", linewidth=1)
ax.grid(axis="y", alpha=0.25)
ax.legend(title="Model")

save_figure(fig, "unseen_coverage_by_cell.png")


# ============================================================
# FIGURE 5: QUANTILE INTERVAL RANGE
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

pivot = original_cells.pivot(
    index="cell",
    columns="model",
    values="mean_quantile_interval_range",
)

pivot.plot(kind="bar", ax=ax)

ax.set_title("Original Dataset: Mean Quantile Interval Range")
ax.set_xlabel("Cell")
ax.set_ylabel("Mean interval range")
ax.grid(axis="y", alpha=0.25)
ax.legend(title="Model")

save_figure(fig, "original_interval_range_by_cell.png")


# ============================================================
# FIGURE 6: UNSEEN QUANTILE INTERVAL RANGE
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

pivot = unseen_cells.pivot(
    index="cell",
    columns="model",
    values="mean_quantile_interval_range",
)

pivot.plot(kind="bar", ax=ax)

ax.set_title("Unseen Dataset: Mean Quantile Interval Range")
ax.set_xlabel("Cell")
ax.set_ylabel("Mean interval range")
ax.grid(axis="y", alpha=0.25)
ax.legend(title="Model")

save_figure(fig, "unseen_interval_range_by_cell.png")


# ============================================================
# FIGURE 7: ORIGINAL VS UNSEEN OVERALL KPI COMPARISON
# ============================================================

orig = original_average.iloc[0]
unseen = unseen_average.iloc[0]

comparison = pd.DataFrame({
    "Original": [
        orig["MAPE_percent"],
        orig["mean_quantile_interval_range"],
        orig["coverage_percent"],
    ],
    "Unseen": [
        unseen["MAPE_percent"],
        unseen["mean_quantile_interval_range"],
        unseen["coverage_percent"],
    ],
}, index=[
    "MAPE (%)",
    "QI range",
    "Coverage (%)",
])

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12, 4),
)

for i, metric in enumerate(comparison.index):
    comparison.loc[[metric]].T.plot(
        kind="bar",
        ax=axes[i],
        legend=False,
    )

    axes[i].set_title(metric)
    axes[i].set_xlabel("")
    axes[i].grid(axis="y", alpha=0.25)

fig.suptitle("Original vs Unseen Dataset Performance")

save_figure(fig, "original_vs_unseen_kpis.png")


# ============================================================
# FIGURE 8: CLUSTER MAPE
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

cluster_plot = pd.concat([
    original_clusters.assign(dataset="Original"),
    unseen_clusters.assign(dataset="Unseen"),
])

cluster_pivot = cluster_plot.pivot_table(
    index="cluster",
    columns="dataset",
    values="MAPE_percent",
)

cluster_pivot.plot(kind="bar", ax=ax)

ax.set_title("MAPE by Cluster")
ax.set_xlabel("Cluster")
ax.set_ylabel("MAPE (%)")
ax.set_yscale("log")
ax.grid(axis="y", alpha=0.25)
ax.legend(title="Dataset")

save_figure(fig, "cluster_mape_comparison.png")


# ============================================================
# AUTOMATIC SUMMARY VALUES
# ============================================================

def get_value(df, column):
    return float(df.iloc[0][column])


orig_mape = get_value(original_average, "MAPE_percent")
orig_range = get_value(
    original_average,
    "mean_quantile_interval_range"
)
orig_coverage = get_value(
    original_average,
    "coverage_percent"
)

unseen_mape = get_value(unseen_average, "MAPE_percent")
unseen_range = get_value(
    unseen_average,
    "mean_quantile_interval_range"
)
unseen_coverage = get_value(
    unseen_average,
    "coverage_percent"
)


# Find best model based on mean cell MAPE.
model_mape_original = (
    original_cells
    .groupby("model")["MAPE_percent"]
    .mean()
    .sort_values()
)

model_mape_unseen = (
    unseen_cells
    .groupby("model")["MAPE_percent"]
    .mean()
    .sort_values()
)

best_original_model = model_mape_original.index[0]
best_unseen_model = model_mape_unseen.index[0]


# ============================================================
# LATEX DOCUMENT
# ============================================================
# ============================================================
# BUILD LATEX REPORT
# ============================================================

# Prepare unseen cell names outside the f-string expression.
unseen_cells_latex = ", ".join(
    rf"\texttt{{{c}}}" for c in unseen_cell_names
)

latex = rf"""
\documentclass[a4paper,11pt]{{article}}

% ------------------------------------------------------------
% Packages
% ------------------------------------------------------------

\usepackage[margin=2.3cm]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{float}}
\usepackage{{amsmath}}
\usepackage{{siunitx}}
\usepackage{{caption}}
\usepackage{{hyperref}}
\usepackage{{microtype}}

\hypersetup{{
    colorlinks=true,
    linkcolor=black,
    urlcolor=blue,
    citecolor=black
}}

\title{{C3 Evaluation: Forecasting Performance}}
\author{{}}
\date{{}}

\begin{{document}}

\maketitle

% ============================================================
% INTRODUCTION
% ============================================================

\section{{Evaluation Objective}}

The objective of C3 is to evaluate the forecasting models on both
cells from the original dataset used during model development and
cells from an unseen dataset. The evaluation considers the final
three days of each time series, corresponding to
{horizon} timestamps.

For the original dataset, the selected cells are:
\texttt{{cell021}}, \texttt{{cell198}}, \texttt{{cell192}}, and
\texttt{{cell214}}.

The unseen evaluation contains {n_unseen} cells:
{unseen_cells_latex}.

For the unseen dataset, the previously trained forecasting models
and clustering model were kept frozen. No retraining or fine-tuning
was performed.

% ============================================================
% KPI DEFINITIONS
% ============================================================

\section{{Evaluation Metrics}}

Three KPIs are used to evaluate the forecasts.

\subsection{{Mean Absolute Percentage Error}}

The Mean Absolute Percentage Error (MAPE) averaged over the
prediction horizon is calculated as

\begin{{equation}}
\mathrm{{MAPE}}
=
\frac{{100}}{{H}}
\sum_{{t=1}}^{{H}}
\left|
\frac{{y_t-\hat{{y}}_t}}{{y_t}}
\right|,
\end{{equation}}

where $y_t$ denotes the actual value, $\hat{{y}}_t$ the forecast,
and $H={horizon}$ the prediction horizon.

Because MAPE divides by the actual value, very small actual values
can result in extremely large percentage errors. This effect is
particularly relevant for some of the cells evaluated here and is
considered when interpreting the results.

\subsection{{Quantile Interval Range}}

For each prediction timestamp, the quantile prediction interval is
defined by a lower and upper quantile. Its range is

\begin{{equation}}
R_t = Q_{{upper,t}} - Q_{{lower,t}}.
\end{{equation}}

The reported quantile interval range is the average over all
{horizon} prediction timestamps:

\begin{{equation}}
\overline{{R}}
=
\frac{{1}}{{H}}
\sum_{{t=1}}^{{H}} R_t.
\end{{equation}}

\subsection{{Coverage}}

Coverage measures the percentage of actual observations that fall
inside the predicted quantile interval:

\begin{{equation}}
\mathrm{{Coverage}}
=
\frac{{100}}{{H}}
\sum_{{t=1}}^{{H}}
\mathbf{{1}}
\left[
Q_{{lower,t}}
\leq y_t \leq
Q_{{upper,t}}
\right].
\end{{equation}}

% ============================================================
% ORIGINAL DATASET
% ============================================================

\section{{Evaluation on the Original Dataset}}

The four selected cells from the original dataset were evaluated
over the final {horizon} timestamps. Both the cluster-specific
LSTM and BiLSTM models were evaluated.

\subsection{{Per-Cell Results}}

{original_latex}

The per-cell results show substantial differences between the cells.
In particular, the MAPE values for some cells are very large. This
is consistent with the sensitivity of MAPE to actual values close
to zero.

\subsection{{Average Over Original Cells}}

{original_avg_latex}

The average across the selected original cells results in a MAPE of
{format_number(orig_mape, 2)}\%, a mean quantile interval range of
{format_number(orig_range, 3)}, and a coverage of
{format_number(orig_coverage, 2)}\%.

\subsection{{Results by Cluster}}

{original_cluster_latex}

The cluster-level results demonstrate that forecasting performance
varies considerably between the learned clusters. This indicates
that the cluster-specific models capture different forecasting
characteristics of the underlying cells.

\subsection{{Visualisation of Original Results}}

{figure_latex(
    "original_mape_by_cell.png",
    "MAPE for the four selected cells from the original dataset. A logarithmic y-axis is used because of the large variation in MAPE.",
    "original_mape"
)}

{figure_latex(
    "original_coverage_by_cell.png",
    "Prediction interval coverage for the selected original cells.",
    "original_coverage"
)}

{figure_latex(
    "original_interval_range_by_cell.png",
    "Mean quantile prediction interval range for the selected original cells.",
    "original_interval_range"
)}

% ============================================================
% UNSEEN DATASET
% ============================================================

\section{{Evaluation on the Unseen Dataset}}

The second evaluation considers {n_unseen} cells that were not used
during model training. The trained clustering model was used to
assign each unseen cell to an existing cluster. The corresponding
cluster-specific forecasting models were then used without
retraining.

\subsection{{Per-Cell Results}}

{unseen_latex}

The results demonstrate a degradation in forecasting performance
for several unseen cells compared with the original dataset.
However, the degree of degradation varies strongly between cells
and clusters.

\subsection{{Average Over All Unseen Cells}}

{unseen_avg_latex}

Across all {n_unseen} unseen cells, the average MAPE is
{format_number(unseen_mape, 2)}\%, the average quantile interval
range is {format_number(unseen_range, 3)}, and the average coverage
is {format_number(unseen_coverage, 2)}\%.

\subsection{{Results by Cluster}}

{unseen_cluster_latex}

The unseen cluster results further demonstrate that generalisation
performance depends strongly on the cluster to which an unseen cell
is assigned.

\subsection{{Visualisation of Unseen Results}}

{figure_latex(
    "unseen_mape_by_cell.png",
    "MAPE for the eight unseen cells. The logarithmic y-axis highlights the substantial variation between cells.",
    "unseen_mape"
)}

{figure_latex(
    "unseen_coverage_by_cell.png",
    "Prediction interval coverage for the eight unseen cells.",
    "unseen_coverage"
)}

{figure_latex(
    "unseen_interval_range_by_cell.png",
    "Mean quantile prediction interval range for the eight unseen cells.",
    "unseen_interval_range"
)}

% ============================================================
% COMPARISON
% ============================================================

\section{{Original versus Unseen Performance}}

The overall results are compared in Figure~\ref{{fig:overall_comparison}}.

{figure_latex(
    "original_vs_unseen_kpis.png",
    "Comparison of overall KPI averages between the original and unseen datasets.",
    "overall_comparison"
)}

The overall MAPE increases from
{format_number(orig_mape, 2)}\% on the original selected cells to
{format_number(unseen_mape, 2)}\% on the unseen cells. This indicates
a reduction in generalisation performance when the models are applied
to cells not observed during training.

The average quantile interval range changes from
{format_number(orig_range, 3)} to {format_number(unseen_range, 3)}.
The corresponding coverage changes from
{format_number(orig_coverage, 2)}\% to
{format_number(unseen_coverage, 2)}\%.

These results should be interpreted jointly. A wider prediction
interval can increase coverage but may provide less precise
uncertainty estimates. Conversely, a narrow interval can be more
informative but may result in lower coverage.

\subsection{{Cluster Comparison}}

{figure_latex(
    "cluster_mape_comparison.png",
    "Comparison of cluster-level MAPE between the original and unseen datasets.",
    "cluster_mape"
)}

% ============================================================
% LSTM VS BILSTM
% ============================================================

\section{{LSTM versus BiLSTM}}

For additional comparison, the mean cell-level MAPE is considered
separately for the two forecasting architectures.

For the original dataset, the model with the lowest average
cell-level MAPE is
\textbf{{{latex_escape(best_original_model)}}}.

For the unseen dataset, the model with the lowest average cell-level
MAPE is
\textbf{{{latex_escape(best_unseen_model)}}}.

The per-cell results should nevertheless be considered in addition
to these averages because the performance varies substantially
between cells.

% ============================================================
% DISCUSSION
% ============================================================

\section{{Discussion}}

The evaluation demonstrates three main findings.

First, forecasting performance is highly dependent on the individual
cell and its cluster. The original dataset already contains large
differences in MAPE between cells, indicating heterogeneous
forecasting behaviour.

Second, performance generally deteriorates when the trained models
are applied to unseen cells without retraining. This provides a
direct assessment of model generalisation and demonstrates that
cluster-based model selection does not completely eliminate the
distribution shift between training and unseen cells.

Third, MAPE should be interpreted carefully for this dataset. Since
MAPE normalises the absolute error by the actual value, observations
close to zero can dominate the metric and produce extremely large
percentage values. Therefore, the MAPE results should be considered
together with the quantile interval range and coverage rather than
being interpreted in isolation.

The coverage results provide information about the reliability of
the prediction intervals, while the quantile interval range
indicates their average width. Together, these metrics provide a
more complete picture of both point-forecast accuracy and predictive
uncertainty.

% ============================================================
% CONCLUSION
% ============================================================

\section{{Conclusion}}

The C3 evaluation was performed over the final {horizon} timestamps,
representing three days of observations. The original evaluation
covered {n_original} selected cells, while the unseen evaluation
covered {n_unseen} previously unseen cells.

The results show that the forecasting models can produce useful
predictions for some cells, but generalisation to unseen cells is
strongly dependent on the cell and cluster. The substantial
variation in MAPE also highlights the limitations of percentage-based
error metrics when actual values approach zero.

The combination of MAPE, quantile interval range, and coverage
therefore provides a more informative evaluation of forecasting
accuracy and uncertainty than any single metric alone.

\end{{document}}
"""


# ============================================================
# WRITE REPORT
# ============================================================

REPORT.write_text(latex, encoding="utf-8")

print()
print("============================================================")
print("DONE")
print("============================================================")
print(f"LaTeX report: {REPORT}")
print(f"Figures:      {FIGURES}")
print()
print("To compile:")
print(f"  cd {BASE}")
print("  pdflatex C3_report.tex")
print("  pdflatex C3_report.tex")
print()
