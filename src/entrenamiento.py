"""
Entrenamiento y comparación de modelos sobre Wine Quality (Kaggle).

La calidad tiene varias clases desbalanceadas y fronteras borrosas (p. ej. 5 vs 6),
así que no esperamos métricas perfectas. La elección del modelo se hace con CV
solo en entrenamiento; el test queda para la evaluación final en prueba.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datos import cargar_vinos, raiz_proyecto
from metricas import metricas_multiclase

RANDOM_STATE = 42


def construir_modelos() -> dict[str, Pipeline | RandomForestClassifier]:
    """
    Dos modelos con supuestos distintos.

    - Regresión logística + escalado: baseline lineal; suele confundir clases
      vecinas de calidad.
    - Random forest: captura no linealidades; con max_depth limitado reducimos
      memorización frente a un bosque sin poda.
    """
    log_reg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )
    return {"reg_logistica": log_reg, "random_forest": rf}


def imprimir_metricas(titulo: str, scores: dict[str, float]) -> None:
    print(f"\n{titulo}")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"  {k}: {scores[k]:.4f}")


def main() -> None:
    X, y, meta = cargar_vinos()
    nombres_clase = meta.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Fuente: {meta.fuente}")
    print(
        f"Muestras: train={len(X_train)}, test={len(X_test)}, "
        f"features={X.shape[1]}, clases={y.nunique()}"
    )

    modelos = construir_modelos()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    resultados_cv: dict[str, float] = {}
    for nombre, modelo in modelos.items():
        scores = cross_val_score(
            modelo,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        media = float(np.mean(scores))
        resultados_cv[nombre] = media
        print(f"\nCV F1 (weighted) — {nombre}: media={media:.4f}, folds={scores.round(4)}")

    ganador = max(resultados_cv, key=resultados_cv.get)
    print(f"\nModelo elegido por CV (F1 weighted): {ganador}")

    modelo_final = modelos[ganador]
    modelo_final.fit(X_train, y_train)

    y_pred_train = modelo_final.predict(X_train)
    y_pred_test = modelo_final.predict(X_test)
    imprimir_metricas("Métricas en entrenamiento (referencia)", metricas_multiclase(y_train, y_pred_train))
    imprimir_metricas("Métricas en conjunto de prueba (modelo final)", metricas_multiclase(y_test, y_pred_test))

    artefacto = {
        "nombre_modelo": ganador,
        "modelo": modelo_final,
        "X_test": X_test,
        "y_test": y_test,
        "target_names": nombres_clase,
        "feature_names": list(X.columns),
        "fuente_datos": meta.fuente,
        "random_state_split": RANDOM_STATE,
    }
    ruta = raiz_proyecto() / "artifacts" / "wine_pipeline.joblib"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefacto, ruta)
    print(f"\nArtefacto guardado en: {ruta.resolve()}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
