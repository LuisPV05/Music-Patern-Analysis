"""Limpia el dataset y genera dos versiones en data/processed."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/dataset.csv")
OUT_DIR = Path("data/processed")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_PATH, index_col=0)
    print(f"Filas originales:            {len(df):>8,}")

    df = df.drop_duplicates()
    print(f"Tras quitar duplicados:      {len(df):>8,}")

    df = df.dropna()
    print(f"Tras quitar nulos:           {len(df):>8,}")

    df = df[(df["tempo"] > 0) & (df["time_signature"] > 0)]
    print(f"Tras quitar tempo/compás 0:  {len(df):>8,}")

    df = df.reset_index(drop=True)
    df["sin_popularidad"] = df["popularity"] == 0

    df.to_csv(OUT_DIR / "tracks_genres.csv", index=False)
    print(f"\nGuardado tracks_genres.csv: {len(df):,} filas")

    # Una fila por canción: se queda con la de mayor popularity y se
    # guardan todos sus géneros juntos en una columna.
    generos = df.groupby("track_id")["track_genre"].agg(lambda s: "|".join(sorted(s.unique())))
    unicas = (
        df.sort_values("popularity", ascending=False)
        .drop_duplicates(subset="track_id", keep="first")
        .drop(columns="track_genre")
        .merge(generos.rename("track_genres"), on="track_id")
        .reset_index(drop=True)
    )
    unicas.to_csv(OUT_DIR / "tracks_unique.csv", index=False)
    print(f"Guardado tracks_unique.csv: {len(unicas):,} filas")


if __name__ == "__main__":
    main()
