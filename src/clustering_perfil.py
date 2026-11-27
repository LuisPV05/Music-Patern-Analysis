"""Entrena K-Means y perfila cada cluster."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from preparacion import FEATURES, load_unique, prepare_features

FIG_DIR = Path("reports/figures")
OUT_PATH = Path("data/processed/tracks_clusters.csv")
K = 8
SEED = 42


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_unique()
    X, _ = prepare_features(df)

    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X)
    df["cluster"] = km.labels_
    df.to_csv(OUT_PATH, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    print("=== Tamaño de cada cluster ===")
    print(df["cluster"].value_counts().sort_index().to_string(), "\n")

    print("=== Perfil medio por cluster ===")
    cols = FEATURES[:-1] + ["popularity"]
    print(df.groupby("cluster")[cols].mean().round(2), "\n")

    print("=== Duración media (min) por cluster ===")
    print((df.groupby("cluster")["duration_ms"].mean() / 60000).round(2).to_string(), "\n")

    print("=== Géneros más frecuentes por cluster ===")
    for c in range(K):
        generos = (
            df[df["cluster"] == c]["track_genres"]
            .str.split("|").explode().value_counts().head(5)
        )
        print(f"Cluster {c}: " + ", ".join(f"{g} ({n})" for g, n in generos.items()))

    # Proyección 2D
    coords = PCA(n_components=2).fit_transform(X)
    idx = np.random.default_rng(SEED).choice(len(X), size=15_000, replace=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(coords[idx, 0], coords[idx, 1], c=km.labels_[idx], cmap="tab10", s=4, alpha=0.6)
    ax.legend(*sc.legend_elements(), title="Cluster", loc="best")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Clusters de canciones (proyección PCA, muestra de 15.000)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "clusters_pca.png", dpi=120)
    print("\nGuardado: reports/figures/clusters_pca.png")


if __name__ == "__main__":
    main()
