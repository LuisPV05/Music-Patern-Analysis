"""Preparación de variables para clustering."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

UNIQUE_PATH = Path("data/processed/tracks_unique.csv")

FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]
LOG_FEATURES = ["speechiness", "liveness", "duration_ms", "instrumentalness"]


def load_unique(path: Path = UNIQUE_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Devuelve la matriz transformada y escalada, y el scaler ajustado."""
    X = df[FEATURES].copy()
    X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def main() -> None:
    df = load_unique()
    X, _ = prepare_features(df)

    print(f"Canciones: {X.shape[0]:,} | Variables: {X.shape[1]}\n")
    resumen = pd.DataFrame(X, columns=FEATURES).describe().round(2).T
    print(resumen[["mean", "std", "min", "50%", "max"]])


if __name__ == "__main__":
    main()
