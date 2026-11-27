"""Perfil medio de audio por género musical."""
from pathlib import Path

import pandas as pd

PROCESSED_PATH = Path("data/processed/tracks_genres.csv")

FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]


def main() -> None:
    df = pd.read_csv(PROCESSED_PATH)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print(f"Géneros: {df['track_genre'].nunique()}")
    print("Canciones por género (min / mediana / max):")
    conteo = df["track_genre"].value_counts()
    print(f"{conteo.min()} / {conteo.median():.0f} / {conteo.max()}\n")

    perfil = df.groupby("track_genre")[FEATURES + ["popularity"]].mean().round(3)

    for col in ["energy", "danceability", "popularity", "acousticness", "valence"]:
        print(f"=== Top 5 géneros por {col} ===")
        print(perfil[col].sort_values(ascending=False).head(5).to_string(), "\n")

    print("=== Bottom 5 por energy ===")
    print(perfil["energy"].sort_values().head(5).to_string())


if __name__ == "__main__":
    main()
