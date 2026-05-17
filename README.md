# Clasificación de calidad del vino (EDA + MLOps modular)

Proyecto para la tarea de **EDA y clasificación de vinos**: código modular en `src/`, datos desde [Kaggle — Wine Quality](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) (`WineQT.csv`) vía `kagglehub`, y modelos comparados con validación cruzada.

**Repositorio:** https://github.com/josezavala6445-gif/MlOps_EDA_Hw

## Estructura del proyecto

```
MlOps_EDA_Hw/
├── README.md
├── requirements.txt
├── docs/
│   └── ENTREGA.md          # Checklist PDF y reporte de problemas
├── artifacts/              # Salidas generadas al ejecutar (no en Git)
│   ├── eda/                # Gráficos del EDA
│   └── wine_pipeline.joblib
└── src/
    ├── datos.py            # Descarga Kaggle + limpieza
    ├── metricas.py         # Accuracy, precision, recall, F1
    ├── eda.py
    ├── entrenamiento.py
    └── prueba.py
```

## Requisitos

- Python 3.10+
- Conexión a internet (primera ejecución descarga el CSV)
- Opcional: [API token de Kaggle](https://www.kaggle.com/settings) si `kagglehub` pide autenticación

## Instalación

```bash
git clone https://github.com/josezavala6445-gif/MlOps_EDA_Hw.git
cd MlOps_EDA_Hw

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución (orden obligatorio)

Desde la raíz del repositorio:

```bash
python src/eda.py
python src/entrenamiento.py
python src/prueba.py
```

| Script | Qué hace |
|--------|----------|
| `eda.py` | Estadísticas, histogramas, boxplots, correlación |
| `entrenamiento.py` | Split estratificado, 2 modelos, CV, guarda el mejor modelo |
| `prueba.py` | Carga el modelo, métricas finales y matriz de confusión |

**Artefactos locales** (se crean al correr; están en `.gitignore`):

- `artifacts/eda/*.png`
- `artifacts/wine_pipeline.joblib`
- `artifacts/matriz_confusion_test.png`

## Pipeline de datos

1. `datos.cargar_vinos()` descarga `WineQT.csv` con `kagglehub`.
2. Se elimina la columna `Id` y filas duplicadas.
3. Objetivo: `quality` (clases 3–8). Features: 11 variables químicas.
4. Entrenamiento: regresión logística (con escalado) vs random forest; el ganador se elige por **F1 weighted** en CV (5 folds).
5. Prueba: evalúa en el `X_test` / `y_test` guardados en el `.joblib`.

## Modelos

- **Regresión logística** + `StandardScaler`: baseline lineal, `class_weight='balanced'`.
- **Random forest**: no lineal, `max_depth=12` y `min_samples_leaf=4` para limitar sobreajuste.

## Entrega académica (PDF)

Ver [docs/ENTREGA.md](docs/ENTREGA.md): enlace a GitHub, capturas de estructura y de los tres scripts, y tabla de al menos dos problemas encontrados.

## Notas

- No hace falta ejecutar código aparte para descargar Kaggle: la descarga ocurre dentro de `datos.py` al correr cualquier script.
- Si `prueba.py` indica que falta el modelo, ejecuta antes `entrenamiento.py`.
