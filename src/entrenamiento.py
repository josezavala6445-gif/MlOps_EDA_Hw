"""
Entrenamiento y comparación de modelos de clasificación para Wine.

Se usa validación cruzada estratificada solo sobre el conjunto de entrenamiento
para elegir el mejor modelo (evita ajustar la elección directamente al test).
Con 178 muestras las estimaciones tienen varianza alta; el comentario en código
deja explícita esa limitación.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datos import cargar_vinos, raiz_proyecto


# Semilla fija para reproducir el mismo train/test y el mismo CV.
RANDOM_STATE = 42


def construir_modelos() -> dict[str, Pipeline | RandomForestClassifier]:
    """
    Devuelve al menos dos clasificadores con supuestos distintos.

    - Regresión logística en pipeline con escalado: límites lineales entre clases,
      sensible a la escala de las features (por eso StandardScaler).
    - Random forest: fronteras no lineales, menos dependiente del escalado;
      suele ir bien en tablas pequeñas aunque puede sobreajustar si se relaja mucho.
    """
    log_reg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )
    return {"reg_logistica": log_reg, "random_forest": rf}


def metricas_multiclase(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calcula accuracy, precision, recall y F1 en problema multiclase.

    Se usa average='weighted' para ponderar cada clase según su soporte; con clases
    casi balanceadas (Wine) se acerca al comportamiento global que resume accuracy,
    pero sigue reflejando errores por clase de forma más fina que accuracy sola.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def imprimir_metricas(titulo: str, scores: dict[str, float]) -> None:
    print(f"\n{titulo}")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"  {k}: {scores[k]:.4f}")


def main() -> None:
    X, y, bunch = cargar_vinos()
    nombres_clase = list(bunch.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

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
    y_pred = modelo_final.predict(X_test)
    scores_test = metricas_multiclase(np.asarray(y_test), y_pred)
    imprimir_metricas("Métricas en conjunto de prueba (modelo final)", scores_test)

    artefacto = {
        "nombre_modelo": ganador,
        "modelo": modelo_final,
        "X_test": X_test,
        "y_test": y_test,
        "target_names": nombres_clase,
        "feature_names": list(X.columns),
        "random_state_split": RANDOM_STATE,
    }
    ruta = raiz_proyecto() / "artifacts" / "wine_pipeline.joblib"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefacto, ruta)
    print(f"\nArtefacto guardado en: {ruta.resolve()}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
