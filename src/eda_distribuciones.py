"""Histogramas de las características de audio."""
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
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for ax, col in zip(axes.ravel(), FEATURES):
        data = df[col]
        if col == "duration_ms":
            data = data[data < data.quantile(0.99)] / 60000
            ax.set_xlabel("duration (min, hasta percentil 99)")
        sns.histplot(data, bins=40, ax=ax, color="#1DB954")
        ax.set_title(col)
    for ax in axes.ravel()[len(FEATURES):]:
        ax.axis("off")

    fig.suptitle("Distribución de las características de audio", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "distribuciones.png", dpi=120)
    print("Guardado: reports/figures/distribuciones.png")


if __name__ == "__main__":
    main()
