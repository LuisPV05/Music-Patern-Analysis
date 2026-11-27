"""Random Forest y Gradient Boosting para predecir popularidad."""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from popularidad_base import SEED, load_xy


def main() -> None:
    X, y = load_xy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    modelos = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=SEED),
    }
    entrenados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_tr, y_tr)
        pred = modelo.predict(X_te)
        cv = cross_val_score(modelo, X_tr, y_tr, cv=5, scoring="r2", n_jobs=1)
        entrenados[nombre] = modelo
        print(f"{nombre:<18} R2 test={r2_score(y_te, pred):.3f} | "
              f"MAE={mean_absolute_error(y_te, pred):.2f} | "
              f"R2 CV={cv.mean():.3f} ± {cv.std():.3f}")

    print("\n=== Importancia por permutación (Gradient Boosting, en test) ===")
    imp = permutation_importance(
        entrenados["Gradient Boosting"], X_te, y_te,
        n_repeats=5, random_state=SEED, scoring="r2",
    )
    print(pd.Series(imp.importances_mean, index=X.columns)
          .sort_values(ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
