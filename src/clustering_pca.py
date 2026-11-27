"""PCA sobre las variables preparadas: varianza explicada y pesos."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from preparacion import FEATURES, load_unique, prepare_features

FIG_DIR = Path("reports/figures")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_unique()
    X, _ = prepare_features(df)

    pca = PCA().fit(X)
    var = pca.explained_variance_ratio_
    acum = np.cumsum(var)

    print("=== Varianza explicada por componente ===")
    for i, (v, a) in enumerate(zip(var, acum), start=1):
        print(f"PC{i}: {v:.1%}  (acumulada {a:.1%})")

    print("\n=== Pesos de las variables en los 3 primeros componentes ===")
    pesos = pd.DataFrame(
        pca.components_[:3].T, index=FEATURES, columns=["PC1", "PC2", "PC3"]
    ).round(2)
    print(pesos)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, len(var) + 1), var, color="#1DB954", label="Individual")
    ax.plot(range(1, len(var) + 1), acum, "o-", color="black", label="Acumulada")
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Varianza explicada")
    ax.set_title("PCA: varianza explicada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_varianza.png", dpi=120)
    print("\nGuardado: reports/figures/pca_varianza.png")


if __name__ == "__main__":
    main()
