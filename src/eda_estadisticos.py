"""Estadísticos descriptivos de las características de audio."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/dataset.csv")

AUDIO_FEATURES = [
    "popularity", "duration_ms", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo",
]


def main() -> None:
    df = pd.read_csv(RAW_PATH, index_col=0).drop_duplicates().dropna()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("=== Estadísticos descriptivos ===")
    print(df[AUDIO_FEATURES].describe().round(3).T, "\n")

    print("=== Asimetría (skew) ===")
    print(df[AUDIO_FEATURES].skew().round(2).sort_values(), "\n")

    print("=== Variables categóricas ===")
    for col in ["key", "mode", "time_signature", "explicit"]:
        print(df[col].value_counts(normalize=True).round(3).to_string(), "\n")

    print(f"Géneros distintos: {df['track_genre'].nunique()}")
    print(f"Canciones con popularity = 0: {(df['popularity'] == 0).sum():,}")


if __name__ == "__main__":
    main()
