# Issue #13: Implement Anomaly Scoring Thresholding & Evaluation Metrics

El objetivo de este issue es construir utilidades para calcular umbrales (thresholds) dinámicos para las puntuaciones de anomalía (ej. usando percentiles o la Teoría de Valores Extremos - EVT) y desarrollar métricas de evaluación estándar como ROC-AUC y Precision@k.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-13-metrics` a partir de `develop` y asegurar el entorno de trabajo:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-13-metrics
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Implementar Thresholding y Evaluación
Crear el archivo `src/models/evaluation.py` (o similar, como `src/models/metrics.py`) e implementar:
- **Cálculo de Umbrales**:
  - `compute_dynamic_threshold(scores, method="percentile", **kwargs)`: Retorna un valor de corte basado en el array de `scores` y el método especificado (percentile, o aproximación EVT empírica).
- **Métricas de Evaluación**:
  - `evaluate_anomalies(y_true, y_scores, k=None)`: Retorna un diccionario con las métricas calculadas. Debe incluir:
    - `roc_auc`: Área bajo la curva ROC (usando `sklearn.metrics.roc_auc_score`).
    - `precision_at_k`: Proporción de verdaderas anomalías entre las `k` muestras con mayor puntuación de anomalía.
    - Opcionalmente `average_precision` u otras métricas deseadas.

### 3. Escribir Tests Unitarios
Crear `tests/unit/test_evaluation.py` para verificar:
- Que `compute_dynamic_threshold` devuelva los valores esperados (ej. para percentiles exactos sobre datos sintéticos).
- Que `evaluate_anomalies` devuelva el valor correcto para `roc_auc` (ej. 1.0 para predicciones perfectas) y `precision@k` (ej. si hay 5 anomalías en el top 10, la precision es 0.5).

### 4. Verificar Calidad y Tests
Correr la suite de pruebas y los linters:
```bash
python -m pytest tests/
ruff check . --fix
mypy src/ tests/
```

### 5. Commit y Merge
```bash
git add src/models/ tests/
git commit -m "feat: implement dynamic thresholding and evaluation metrics (Issue #13)"
git checkout develop
git merge feature/issue-13-metrics
git push origin develop
```
