# Guía de entrega — EDA y clasificación de vinos

Archivo PDF sugerido: **ApellidoNombre EDA Vinos.pdf**

## Contenido obligatorio del PDF

1. **Enlace al repositorio**  
   https://github.com/josezavala6445-gif/MlOps_EDA_Hw

2. **Captura de la estructura del repo**  
   Vista de carpetas (`src/`, `docs/`, `requirements.txt`, etc.).

3. **Captura ejecutando** `python src/eda.py`  
   Salida con estadísticas y rutas de figuras guardadas.

4. **Captura ejecutando** `python src/entrenamiento.py`  
   CV de ambos modelos, modelo elegido y métricas en test.

5. **Captura ejecutando** `python src/prueba.py`  
   Métricas finales y matriz de confusión (consola + opcional `artifacts/matriz_confusion_test.png`).

6. **Reporte de problemas (mínimo 2)**

| Problema | ¿Cómo lo resolviste? | ¿Se resolvió? |
|----------|----------------------|---------------|
| *(ejemplo)* Métricas 1.0 con `sklearn.datasets.load_wine` (dataset muy pequeño y separable) | Migrar a Kaggle Wine Quality (~1000 filas, 6 clases, más ruido) | Sí — métricas en test ~0.65 |
| *(ejemplo)* 125 filas duplicadas en el CSV | `drop_duplicates()` centralizado en `datos.cargar_vinos()` | Sí |
| *(ejemplo)* Gap train vs test (train ~0.91, test ~0.66) | Limitar `max_depth` del RF y reportar ambas métricas; interpretar como posible sobreajuste parcial | Parcial — mejora interpretación, no elimina toda la brecha |
| *(añade el tuyo)* | | |

Personaliza la tabla con problemas reales que hayas vivido (Kaggle sin credenciales, rutas de artefactos, warnings de sklearn, etc.).

## Comandos para reproducir capturas

```bash
cd MlOps_EDA_Hw
source .venv/bin/activate
python src/eda.py
python src/entrenamiento.py
python src/prueba.py
```

## Qué no va en Git (y por qué)

- `.venv/` — entorno local
- `artifacts/*.joblib`, `artifacts/**/*.png` — se regeneran; cada persona entrena de nuevo

Esto es intencional: el repo versiona **código**, no modelos ni figuras.
