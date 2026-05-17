"""
Carga del dataset Wine Quality desde Kaggle (yasserh/wine-quality-dataset).

Centralizar la descarga y el preprocesado básico evita duplicar lógica entre
eda.py, entrenamiento.py y prueba.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter

KAGGLE_DATASET = "yasserh/wine-quality-dataset"
KAGGLE_FILE = "WineQT.csv"
COLUMNA_OBJETIVO = "quality"
COLUMNAS_IGNORAR = ("Id",)


@dataclass(frozen=True)
class MetadatosVino:
    """Información del dataset (sustituye al ``Bunch`` de sklearn)."""

    target_names: list[str]
    feature_names: list[str]
    fuente: str


def raiz_proyecto() -> Path:
    """Devuelve la raíz del repositorio (carpeta padre de ``src``)."""
    return Path(__file__).resolve().parent.parent


def _descargar_tabla() -> pd.DataFrame:
    """Descarga la versión más reciente del CSV en Kaggle vía kagglehub."""
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        KAGGLE_DATASET,
        KAGGLE_FILE,
    )


def cargar_vinos():
    """
    Carga features y etiqueta de calidad del vino.

    La columna ``quality`` es entera (escala ~3–8 en este CSV). Problema
    multiclase desbalanceado (muchas muestras en 5 y 6).

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series, MetadatosVino]
        Features numéricas, objetivo y metadatos con nombres de columnas/clases.
    """
    df = _descargar_tabla()
    df = df.drop(columns=[c for c in COLUMNAS_IGNORAR if c in df.columns])

    if COLUMNA_OBJETIVO not in df.columns:
        raise ValueError(
            f"No se encontró la columna objetivo '{COLUMNA_OBJETIVO}'. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    y = df[COLUMNA_OBJETIVO].astype(int)
    X = df.drop(columns=[COLUMNA_OBJETIVO])
    X = X.select_dtypes(include="number")

    # Misma limpieza en todos los módulos (el CSV trae filas repetidas)
    tabla = pd.concat([X, y.rename(COLUMNA_OBJETIVO)], axis=1)
    if tabla.duplicated().any():
        tabla = tabla.drop_duplicates()
        y = tabla[COLUMNA_OBJETIVO].astype(int)
        X = tabla.drop(columns=[COLUMNA_OBJETIVO])

    clases = sorted(y.unique())
    meta = MetadatosVino(
        target_names=[f"calidad_{c}" for c in clases],
        feature_names=list(X.columns),
        fuente=f"kaggle:{KAGGLE_DATASET}/{KAGGLE_FILE}",
    )
    return X, y, meta
