"""Vista general del Spotify Tracks Dataset: forma, tipos, nulos y duplicados."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/dataset.csv")


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def main() -> None:
    df = load_data()

    print(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}\n")
    print("Tipos de datos:")
    print(df.dtypes, "\n")

    nulos = df.isnull().sum()
    print("Nulos por columna:")
    print(nulos[nulos > 0] if nulos.any() else "Ninguno", "\n")

    print(f"Filas duplicadas: {df.duplicated().sum()}")
    print(f"track_id duplicados: {df['track_id'].duplicated().sum()}")


if __name__ == "__main__":
    main()
