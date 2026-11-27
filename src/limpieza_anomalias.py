"""Cuenta valores anómalos del dataset antes de limpiarlo."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/dataset.csv")


def main() -> None:
    df = pd.read_csv(RAW_PATH, index_col=0).drop_duplicates().dropna()

    print(f"Filas de partida: {len(df):,}\n")

    reglas = {
        "tempo == 0": df["tempo"] == 0,
        "time_signature == 0": df["time_signature"] == 0,
        "danceability == 0": df["danceability"] == 0,
        "duration < 30 s": df["duration_ms"] < 30_000,
        "duration > 20 min": df["duration_ms"] > 20 * 60_000,
        "loudness < -35 dB": df["loudness"] < -35,
        "popularity == 0": df["popularity"] == 0,
    }
    for nombre, mascara in reglas.items():
        print(f"{nombre:<22} {mascara.sum():>7,}  ({mascara.mean():.2%})")

    print("\n=== ¿Se solapan tempo=0, time_signature=0 y danceability=0? ===")
    raros = df[(df["tempo"] == 0) | (df["time_signature"] == 0) | (df["danceability"] == 0)]
    print(f"Filas con alguno de los tres: {len(raros):,}")
    print(raros[["track_name", "tempo", "time_signature", "danceability", "duration_ms"]].head(8))


if __name__ == "__main__":
    main()
