"""Heatmap de géneros vs características de audio y géneros más parecidos."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

PROCESSED_PATH = Path("data/processed/tracks_genres.csv")
FIG_DIR = Path("reports/figures")

FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "popularity",
]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_PATH)

    perfil = df.groupby("track_genre")[FEATURES].mean()
    # Normalización min-max por columna (0 = género más bajo, 1 = más alto)
    norm = (perfil - perfil.min()) / (perfil.max() - perfil.min())

    g = sns.clustermap(
        norm, cmap="viridis", figsize=(10, 26),
        method="ward", dendrogram_ratio=(0.12, 0.04),
        cbar_pos=(0.02, 0.97, 0.03, 0.015), yticklabels=True,
    )
    g.ax_heatmap.tick_params(axis="y", labelsize=7)
    g.ax_heatmap.set_ylabel("")
    g.figure.suptitle("Perfil de audio por género (normalizado)", y=1.0, fontsize=14)
    g.savefig(FIG_DIR / "generos_heatmap.png", dpi=110)
    print("Guardado: reports/figures/generos_heatmap.png\n")

    # Géneros más parecidos: distancia euclídea entre perfiles normalizados.
    # songwriter es idéntico a singer-songwriter (mismas canciones), se excluye.
    norm_pares = norm.drop(index="songwriter")
    dist = squareform(pdist(norm_pares.values))
    i, j = np.triu_indices(len(norm_pares), k=1)
    pares = pd.Series(
        dist[i, j],
        index=pd.MultiIndex.from_arrays(
            [norm_pares.index[i], norm_pares.index[j]]
        ),
    ).sort_values()

    print("=== 10 pares de géneros más parecidos ===")
    for (a, b), d in pares.head(10).items():
        print(f"{a} - {b}: {d:.3f}")

    print("\n=== 5 pares más distintos ===")
    for (a, b), d in pares.tail(5)[::-1].items():
        print(f"{a} - {b}: {d:.3f}")


if __name__ == "__main__":
    main()
