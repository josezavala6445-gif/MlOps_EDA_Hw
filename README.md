# Clasificación de vinos (EDA + modelos)

Proyecto modular para el dataset **Wine** de `sklearn.datasets`.

## Entorno

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución (desde la raíz del repo)

```bash
python src/eda.py
python src/entrenamiento.py
python src/prueba.py
```

Los gráficos del EDA se guardan en `artifacts/eda/`. El modelo y el conjunto de prueba serializado quedan en `artifacts/wine_pipeline.joblib` tras entrenar.
