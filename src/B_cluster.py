# B6_cluster_interpretation.py
import os
import json
import pickle
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from scipy.stats import kruskal, f_oneway

warnings.filterwarnings("ignore")
# =====================================================
# CONFIG
# =====================================================
FEATURE_FILE = "output/A_stl/tables/cell_forecastability_features.csv"
# original time-series data
# change this if needed
TIME_SERIES_FILE = "UL_PRB_data_set.csv"
OUTPUT_ROOT = "output/B_cluster"
SEED = 42
N_CLUSTERS = 6
PCA_VARIANCE = 0.95


# =====================================================
# OUTPUT
# =====================================================
def create_output():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_ROOT)  # , timestamp)
    folders = ["models", "tables", "plots", "plots/time_series", "plots/examples"]
    for f in folders:
        os.makedirs(os.path.join(path, f), exist_ok=True)
    return path


# =====================================================
# LOAD FEATURES
# =====================================================
def load_features():
    df = pd.read_csv(FEATURE_FILE)
    print("Feature data:", df.shape)
    df = df.dropna(axis=1, how="all")
    df = df.dropna()
    ids = df["series_id"]
    X = df.drop(columns=["series_id"])
    return X, ids


# =====================================================
# TRAIN FINAL MODEL
# =====================================================
def train_model(X):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=PCA_VARIANCE, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    print("PCA dimensions:", X_pca.shape[1])
    model = SpectralClustering(
        n_clusters=N_CLUSTERS, random_state=SEED, affinity="nearest_neighbors"
    )
    labels = model.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    print("Silhouette:", score)
    return (scaler, pca, model, X_scaled, X_pca, labels)


# =====================================================
# SAVE MODEL
# =====================================================
def save_model(out, scaler, pca, model):
    with open(os.path.join(out, "models", "cluster_pipeline.pkl"), "wb") as f:
        pickle.dump(
            {"scaler": scaler, "pca": pca, "model": model, "clusters": N_CLUSTERS}, f
        )
    metadata = {
        "method": "SpectralClustering",
        "clusters": N_CLUSTERS,
        "scaler": "RobustScaler",
        "pca": PCA_VARIANCE,
        "created": str(datetime.now()),
    }
    with open(os.path.join(out, "models", "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


# =====================================================
# CLUSTER TABLES
# =====================================================
def save_cluster_tables(out, X, ids, labels):
    assignments = pd.DataFrame({"series_id": ids, "cluster": labels})
    assignments.to_csv(
        os.path.join(out, "tables", "cluster_assignments.csv"), index=False
    )
    profile = X.assign(cluster=labels).groupby("cluster").mean()
    profile.to_csv(os.path.join(out, "tables", "cluster_profiles.csv"))
    return assignments, profile


# =====================================================
# FEATURE IMPORTANCE
# =====================================================
def feature_importance(out, X, labels):
    global_mean = X.mean()
    global_std = X.std()
    rows = []
    for c in sorted(np.unique(labels)):
        cluster_mean = X[labels == c].mean()
        z = (cluster_mean - global_mean) / global_std
        for feature, value in z.items():
            rows.append({"cluster": c, "feature": feature, "z_score": value})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "tables", "feature_importance.csv"), index=False)
    return df


# =====================================================
# STATISTICAL TESTS
# =====================================================
def statistical_tests(out, X, labels):
    rows = []
    for feature in X.columns:
        groups = [X.loc[labels == c, feature] for c in np.unique(labels)]
        try:
            f, p = f_oneway(*groups)
        except:
            f, p = np.nan, np.nan
        try:
            h, pk = kruskal(*groups)
        except:
            h, pk = np.nan, np.nan
        rows.append({"feature": feature, "anova_p": p, "kruskal_p": pk})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "tables", "statistics.csv"), index=False)


# =====================================================
# PLOTS
# =====================================================
def make_plots(out, X_pca, labels, profile):
    # PCA plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=40)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Spectral clustering PCA view")
    plt.savefig(os.path.join(out, "plots", "pca_clusters.png"), dpi=300)
    plt.close()
    # sizes
    plt.figure(figsize=(7, 5))
    pd.Series(labels).value_counts().sort_index().plot(kind="bar")
    plt.title("Cluster sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Cells")
    plt.savefig(os.path.join(out, "plots", "cluster_sizes.png"), dpi=300)
    plt.close()
    # heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(profile.T, cmap="coolwarm", center=0)
    plt.title("Cluster feature profiles")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plots", "cluster_heatmap.png"), dpi=300)
    plt.close()


# =====================================================
# TIME SERIES VISUALIZATION
# =====================================================
def plot_time_series(out, assignments):
    if not os.path.exists(TIME_SERIES_FILE):
        print("No time series file found")
        return
    ts = pd.read_csv(TIME_SERIES_FILE)
    if "series_id" not in ts.columns:
        print("Time series needs series_id")
        return
    merged = ts.merge(assignments, on="series_id")
    for c in sorted(merged.cluster.unique()):
        cluster_data = merged[merged.cluster == c]
        values = cluster_data.drop(columns=["series_id", "cluster"])
        mean = values.mean()
        std = values.std()
        plt.figure(figsize=(12, 4))
        plt.plot(mean)
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
        plt.title(f"Cluster {c} average time series")
        plt.savefig(
            os.path.join(out, "plots/time_series", f"cluster_{c}_average.png"), dpi=300
        )
        plt.close()
        # examples
        examples = cluster_data.series_id.head(5)
        examples.to_csv(
            os.path.join(out, "tables", f"cluster_{c}_examples.csv"), index=False
        )


# =====================================================
# REPORT
# =====================================================
def create_report(out, profile, stats):
    path = os.path.join(out, "cluster_report.md")
    with open(path, "w") as f:
        f.write(f"""
# Forecastability Cluster Analysis
## Method
- RobustScaler
- PCA retaining {PCA_VARIANCE*100:.0f}% variance
- Spectral clustering
- Number of clusters: {N_CLUSTERS}
## Cluster profiles
""")
        f.write(profile.to_markdown())
        f.write("""
## Statistical significance
Features with low p-values are strongly different between clusters.
""")
        f.write(stats.to_markdown())


# =====================================================
# MAIN
# =====================================================
def main():
    out = create_output()
    X, ids = load_features()
    scaler, pca, model, X_scaled, X_pca, labels = train_model(X)
    save_model(out, scaler, pca, model)
    assignments, profile = save_cluster_tables(out, X, ids, labels)
    importance = feature_importance(out, X, labels)
    statistical_tests(out, X, labels)
    stats = pd.read_csv(os.path.join(out, "tables", "statistics.csv"))
    make_plots(out, X_pca, labels, profile)
    plot_time_series(out, assignments)
    create_report(out, profile, stats)
    print("\nDONE")
    print("Saved:", out)


if __name__ == "__main__":
    main()
