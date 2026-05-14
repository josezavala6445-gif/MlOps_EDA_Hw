"""
Utilidades compartidas para cargar el dataset Wine de scikit-learn.

Centralizar la carga evita duplicar lógica entre eda.py, entrenamiento.py y
prueba.py y garantiza que todos los módulos ven las mismas columnas y etiquetas.
"""

from __future__ import annotations

from pathlib import Path

from sklearn.datasets import load_wine


def raiz_proyecto() -> Path:
    """Devuelve la raíz del repositorio (carpeta padre de ``src``)."""
    return Path(__file__).resolve().parent.parent


def cargar_vinos():
    """
    Carga el dataset Wine como DataFrame de features y Series de etiquetas.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series, sklearn.utils.Bunch]
        Features, objetivo y el objeto bunch completo (incluye ``feature_names``
        y ``target_names``).
    """
    bunch = load_wine(as_frame=True)
    return bunch.data, bunch.target, bunch
