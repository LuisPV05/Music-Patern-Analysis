"""Matriz de correlación de las características de audio."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RAW_PATH = Path("data/raw/dataset.csv")
FIG_DIR = Path("reports/figures")

FEATURES = [
    "popularity", "duration_ms", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo",
]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_PATH, index_col=0).drop_duplicates().dropna()

    # Spearman: robusto a las distribuciones tan sesgadas que hemos visto
    corr = df[FEATURES].corr(method="spearman")

    mask = None
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, center=0, square=True, ax=ax, mask=mask,
    )
    ax.set_title("Correlación de Spearman entre características de audio")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlacion.png", dpi=120)
    print("Guardado: reports/figures/correlacion.png\n")

    print("=== Pares más correlacionados (|rho|) ===")
    pares = corr.abs().where(~pd.DataFrame(
        [[i >= j for j in range(len(corr))] for i in range(len(corr))],
        index=corr.index, columns=corr.columns,
    )).stack().sort_values(ascending=False)
    for (a, b), _ in pares.head(8).items():
        print(f"{a} - {b}: {corr.loc[a, b]:+.2f}")


if __name__ == "__main__":
    main()
