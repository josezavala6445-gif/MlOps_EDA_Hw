"""Métricas de clasificación multiclase usadas en entrenamiento y prueba."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def metricas_multiclase(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Accuracy, precision, recall y F1 con ``average='weighted'``.

    En calidad del vino las clases 5 y 6 dominan; el promedio ponderado refleja
    mejor el desempeño global que un macro puro cuando hay clases raras (3, 8).
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
