"""Inspección de nulos y duplicados del Spotify Tracks Dataset."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/dataset.csv")


def main() -> None:
    df = pd.read_csv(RAW_PATH, index_col=0)

    print("=== Filas con nulos ===")
    print(df[df.isnull().any(axis=1)], "\n")

    print("=== Duplicados exactos (ejemplos) ===")
    exactos = df[df.duplicated(keep=False)].sort_values("track_id")
    print(exactos[["track_id", "track_name", "track_genre"]].head(6), "\n")

    print("=== Mismo track_id en varios géneros (ejemplos) ===")
    sin_exactos = df.drop_duplicates()
    repetidos = sin_exactos[sin_exactos["track_id"].duplicated(keep=False)]
    print(repetidos.sort_values("track_id")[["track_id", "track_name", "track_genre"]].head(6), "\n")

    print(f"track_id únicos: {df['track_id'].nunique():,}")
    print(f"Filas tras quitar duplicados exactos: {len(sin_exactos):,}")
    print(f"Canciones con más de un género: {sin_exactos.groupby('track_id')['track_genre'].nunique().gt(1).sum():,}")


if __name__ == "__main__":
    main()
