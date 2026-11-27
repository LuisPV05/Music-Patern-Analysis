"""Modelo base de popularidad: media constante y regresión lineal."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preparacion import FEATURES, LOG_FEATURES, load_unique

EXTRA = ["explicit", "key", "mode", "time_signature"]
SEED = 42


def load_xy() -> tuple[pd.DataFrame, pd.Series]:
    df = load_unique()
    df = df[~df["sin_popularidad"]].reset_index(drop=True)
    X = df[FEATURES + EXTRA].copy()
    X["explicit"] = X["explicit"].astype(int)
    X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])
    return X, df["popularity"]


def main() -> None:
    X, y = load_xy()
    print(f"Canciones usadas: {len(X):,} | Variables: {X.shape[1]}")
    print(f"Popularidad: media {y.mean():.1f}, desviación {y.std():.1f}\n")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    modelos = {
        "Media constante": DummyRegressor(strategy="mean"),
        "Regresión lineal": make_pipeline(StandardScaler(), LinearRegression()),
    }
    for nombre, modelo in modelos.items():
        modelo.fit(X_tr, y_tr)
        pred = modelo.predict(X_te)
        cv = cross_val_score(modelo, X_tr, y_tr, cv=5, scoring="r2")
        print(f"{nombre:<18} R2 test={r2_score(y_te, pred):.3f} | "
              f"MAE={mean_absolute_error(y_te, pred):.2f} | "
              f"R2 CV={cv.mean():.3f} ± {cv.std():.3f}")

    lineal = modelos["Regresión lineal"]
    coefs = pd.Series(lineal[-1].coef_, index=X.columns).sort_values()
    print("\n=== Coeficientes (variables estandarizadas) ===")
    print(coefs.round(2).to_string())


if __name__ == "__main__":
    main()
