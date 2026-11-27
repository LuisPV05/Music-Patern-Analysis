"""Elección del número de clusters: método del codo y silhouette."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from preparacion import load_unique, prepare_features

FIG_DIR = Path("reports/figures")
K_RANGE = range(2, 13)
SEED = 42


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_unique()
    X, _ = prepare_features(df)

    rng = np.random.default_rng(SEED)
    muestra = X[rng.choice(len(X), size=10_000, replace=False)]

    inercias, siluetas = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        inercias.append(km.inertia_)
        sil = silhouette_score(muestra, km.predict(muestra))
        siluetas.append(sil)
        print(f"k={k:>2}  inercia={km.inertia_:>10,.0f}  silhouette={sil:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(list(K_RANGE), inercias, "o-", color="#1DB954")
    axes[0].set_title("Método del codo")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inercia")
    axes[1].plot(list(K_RANGE), siluetas, "o-", color="black")
    axes[1].set_title("Silhouette (muestra de 10.000)")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "clustering_k.png", dpi=120)
    print("\nGuardado: reports/figures/clustering_k.png")


if __name__ == "__main__":
    main()
