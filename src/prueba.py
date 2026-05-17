"""
Evaluación del modelo guardado en el conjunto de prueba serializado.

El artefacto incluye X_test e y_test generados en entrenamiento.py para que
esta etapa sea reproducible sin recomputar el split (útil para la demo y para
evitar desalineación si cambia el código de partición).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from datos import raiz_proyecto
from metricas import metricas_multiclase


def main() -> None:
    ruta = raiz_proyecto() / "artifacts" / "wine_pipeline.joblib"
    if not ruta.exists():
        print("No se encontró el modelo. Ejecuta primero: python src/entrenamiento.py")
        sys.exit(1)

    paquete = joblib.load(ruta)
    modelo = paquete["modelo"]
    X_test = paquete["X_test"]
    y_test = paquete["y_test"]
    etiquetas = paquete.get("target_names")

    y_pred = modelo.predict(X_test)
    scores = metricas_multiclase(np.asarray(y_test), y_pred)

    print("Modelo cargado:", paquete.get("nombre_modelo", "(sin nombre)"))
    if paquete.get("fuente_datos"):
        print("Datos:", paquete["fuente_datos"])
    print("\n--- Métricas finales (conjunto de prueba) ---")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Reporte por clase ---")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=etiquetas,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetas)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusión (test)")
    fig.tight_layout()
    salida = raiz_proyecto() / "artifacts" / "matriz_confusion_test.png"
    fig.savefig(salida, dpi=120)
    print(f"\nFigura guardada en: {salida.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
