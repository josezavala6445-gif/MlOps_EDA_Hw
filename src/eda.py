"""
Análisis exploratorio del dataset Wine Quality (Kaggle).

Son ~1100 muestras con calidad en escala ordinal (varias clases, muchas en 5–6).
El EDA sirve a ver desbalance, escalas distintas entre químicos y correlaciones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datos import cargar_vinos, raiz_proyecto


def revisar_calidad(df: pd.DataFrame) -> None:
    """Reporta nulos; los duplicados ya se quitan al cargar en ``datos.cargar_vinos``."""
    print("\n--- Calidad de datos ---")
    nulos = int(df.isna().sum().sum())
    print(f"Total celdas NaN en features: {nulos}")
    if nulos > 0:
        print("Atención: hay valores faltantes; conviene imputar antes de modelar.")
    else:
        print("Sin nulos. Duplicados eliminados en la carga (ver datos.py).")


def estadisticas_descriptivas(df: pd.DataFrame, y: pd.Series) -> None:
    """Imprime resumen numérico y balance de clases."""
    print("\n--- Estadísticas descriptivas (features) ---")
    print(df.describe().round(3).to_string())
    print("\n--- Conteo por calidad (clase) ---")
    conteo = y.value_counts().sort_index()
    for calidad, n in conteo.items():
        print(f"  calidad {calidad}: {int(n)} muestras ({100 * n / len(y):.1f}%)")


def graficar_distribuciones(df: pd.DataFrame, salida: Path) -> None:
    """Histogramas de variables con escalas muy distintas (alcohol vs azúcar)."""
    salida.mkdir(parents=True, exist_ok=True)
    cols = ["alcohol", "volatile acidity", "residual sugar", "density"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(df[col], bins=25, color="#4c72b0", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("valor")
        ax.set_ylabel("frecuencia")
    fig.suptitle("Histogramas (Wine Quality — Kaggle)")
    fig.tight_layout()
    ruta = salida / "histogramas_clave.png"
    fig.savefig(ruta, dpi=120)
    print(f"Guardado: {ruta}")
    plt.close(fig)


def graficar_boxplots_por_clase(df: pd.DataFrame, y: pd.Series, salida: Path) -> None:
    """Boxplots por nivel de calidad: se ve solapamiento entre clases vecinas."""
    data = df.copy()
    data["calidad"] = y.values
    features = ["alcohol", "volatile acidity", "sulphates"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, features):
        sns.boxplot(data=data, x="calidad", y=col, ax=ax)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Distribución por nivel de calidad")
    fig.tight_layout()
    ruta = salida / "boxplots_por_clase.png"
    fig.savefig(ruta, dpi=120)
    print(f"Guardado: {ruta}")
    plt.close(fig)


def graficar_correlacion(df: pd.DataFrame, salida: Path) -> None:
    """Heatmap de correlación entre features químicas."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, ax=ax, cmap="vlag", center=0, square=True)
    ax.set_title("Correlación entre features")
    fig.tight_layout()
    ruta = salida / "matriz_correlacion.png"
    fig.savefig(ruta, dpi=120)
    print(f"Guardado: {ruta}")
    plt.close(fig)


def main() -> None:
    X, y, meta = cargar_vinos()

    print("Dataset Wine Quality (Kaggle)")
    print(f"Fuente: {meta.fuente}")
    print(f"Muestras: {len(X)}, features: {X.shape[1]}, clases: {y.nunique()}")

    revisar_calidad(X)
    estadisticas_descriptivas(X, y)

    print("\n--- Etiquetas de clase ---")
    for nombre in meta.target_names:
        print(f"  {nombre}")

    dir_fig = raiz_proyecto() / "artifacts" / "eda"
    graficar_distribuciones(X, dir_fig)
    graficar_boxplots_por_clase(X, y, dir_fig)
    graficar_correlacion(X, dir_fig)

    corr = X.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pares = corr.where(mask).stack().sort_values(ascending=False).head(3)
    print("\n--- Pares con mayor correlación positiva ---")
    for (a, b), val in pares.items():
        print(f"  {a} vs {b}: {val:.3f}")

    print("\nEDA terminado. Figuras en:", dir_fig.resolve())


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
