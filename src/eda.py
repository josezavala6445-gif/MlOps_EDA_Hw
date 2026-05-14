"""
Análisis exploratorio del dataset Wine (UCI) vía sklearn.

El dataset tiene 178 muestras y 3 clases de vino; con tan pocas filas el EDA
sirve sobre todo a entender escalas, correlaciones y si hay filas duplicadas
o valores faltantes antes de modelar.
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
    """
    Revisa valores faltantes, duplicados y tipos. Wine suele venir limpio;
    igual conviene demostrar el chequeo explícito.
    """
    print("\n--- Calidad de datos ---")
    nulos = df.isna().sum().sum()
    dups = df.duplicated().sum()
    print(f"Total celdas NaN: {int(nulos)}")
    print(f"Filas duplicadas (solo features): {int(dups)}")
    if nulos == 0 and dups == 0:
        print("No se aplicó limpieza adicional: no había nulos ni duplicados.")
    else:
        print("Atención: revisar filas problemáticas antes del modelado.")


def estadisticas_descriptivas(df: pd.DataFrame, y: pd.Series) -> None:
    """Imprime resumen numérico y balance de clases."""
    print("\n--- Estadísticas descriptivas (features) ---")
    print(df.describe().round(3).to_string())
    print("\n--- Conteo por clase ---")
    conteo = y.value_counts().sort_index()
    for clase, n in conteo.items():
        print(f"  clase {clase}: {int(n)} muestras")


def graficar_distribuciones(df: pd.DataFrame, y: pd.Series, salida: Path) -> None:
    """
    Histogramas de un subconjunto de variables (escalas muy distintas:
    ``proline`` llega a miles mientras ``non_flavanoid_phenols`` es pequeña).
    """
    salida.mkdir(parents=True, exist_ok=True)
    cols = ["alcohol", "flavanoids", "color_intensity", "proline"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.ravel()
    for ax, col in zip(axes, cols):
        ax.hist(df[col], bins=18, color="#4c72b0", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("valor")
        ax.set_ylabel("frecuencia")
    fig.suptitle("Histogramas (variables en rangos distintos)")
    fig.tight_layout()
    ruta = salida / "histogramas_clave.png"
    fig.savefig(ruta, dpi=120)
    print(f"Guardado: {ruta}")
    plt.close(fig)


def graficar_boxplots_por_clase(df: pd.DataFrame, y: pd.Series, salida: Path) -> None:
    """
    Boxplots por clase: ayuda a ver si alguna variable separa bien los grupos.
    """
    data = df.copy()
    data["clase"] = y.values
    features = ["alcohol", "od280/od315_of_diluted_wines", "flavanoids"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, col in zip(axes, features):
        sns.boxplot(data=data, x="clase", y=col, ax=ax)
        ax.set_title(col)
    fig.suptitle("Distribución por clase (0, 1, 2)")
    fig.tight_layout()
    ruta = salida / "boxplots_por_clase.png"
    fig.savefig(ruta, dpi=120)
    print(f"Guardado: {ruta}")
    plt.close(fig)


def graficar_correlacion(df: pd.DataFrame, salida: Path) -> None:
    """Matriz de correlación: detecta redundancia fuerte entre features."""
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
    X, y, bunch = cargar_vinos()

    print("Dataset Wine (sklearn)")
    print(f"Muestras: {len(X)}, features: {X.shape[1]}, clases: {y.nunique()}")

    revisar_calidad(X)
    estadisticas_descriptivas(X, y)
    # Imprimir nombres de clase legibles
    print("\n--- Nombres de clase (target) ---")
    for i, nombre in enumerate(bunch.target_names):
        print(f"  {i}: {nombre}")

    dir_fig = raiz_proyecto() / "artifacts" / "eda"
    graficar_distribuciones(X, y, dir_fig)
    graficar_boxplots_por_clase(X, y, dir_fig)
    graficar_correlacion(X, dir_fig)

    # Correlaciones muy altas (ejemplo de insight)
    corr = X.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pares = (
        corr.where(mask)
        .stack()
        .sort_values(ascending=False)
        .head(3)
    )
    print("\n--- Pares con mayor correlación positiva (triángulo superior) ---")
    for (a, b), val in pares.items():
        print(f"  {a} vs {b}: {val:.3f}")

    print("\nEDA terminado. Revisa las figuras en:", dir_fig.resolve())


if __name__ == "__main__":
    # Permite ejecutar como ``python src/eda.py`` añadiendo ``src`` al path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
