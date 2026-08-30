# Issue #10: Implement Baseline Unsupervised Anomaly Detector (Isolation Forest)

El objetivo de este issue es dar inicio al Milestone 3 (Machine Learning Anomaly Detection Models), estableciendo la arquitectura base para los modelos de detección de anomalías y desarrollando los detectores no supervisados de línea base (Isolation Forest y Local Outlier Factor) para calcular scores de anomalía en métricas de activos.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-10-baseline-models` a partir de `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-10-baseline-models
```

### 2. Definir la Interfaz Base para Modelos de Anomalías
Crear el archivo `src/models/base.py`:
- Definir la clase base abstracta `BaseAnomalyDetector(ABC)` con los métodos estándar:
  - `fit(X, y=None)`: Entrenar el modelo con el conjunto de características.
  - `predict(X)`: Predecir si una muestra es anómala (booleano o etiquetas 1/-1).
  - `score_samples(X)`: Obtener el score continuo normalizado de anomalía para cada observación.

### 3. Implementar Detectores Baseline (Isolation Forest y LOF)
Crear el archivo `src/models/baseline.py`:
- **`IsolationForestDetector`**: Encapsular `sklearn.ensemble.IsolationForest` adaptándolo a la interfaz `BaseAnomalyDetector`, con parámetros configurables (`contamination`, `n_estimators`, `random_state`).
- **`LOFDetector`**: Encapsular `sklearn.neighbors.LocalOutlierFactor` adaptándolo a la interfaz `BaseAnomalyDetector`, con soporte para scoring (`novelty=True`).

### 4. Escribir Tests Unitarios
Crear `tests/unit/test_baseline_models.py`:
- Generar datos sintéticos con clusters normales y outliers artificiales extremos.
- Probar que `fit`, `predict` y `score_samples` funcionen para `IsolationForestDetector` y `LOFDetector`.
- Validar que los scores continuos asignen mayores anomalías a los outliers que a los puntos normales.
- Manejar excepciones en caso de entradas vacías o dimensiones incompatibles.

### 5. Verificar Calidad y Tests
Correr la suite de pruebas y linters sobre los cambios nuevos:
```bash
python -m pytest -o addopts="" tests/unit/test_baseline_models.py
ruff check . --fix
mypy src/ tests/
```

### 6. Commit de los Cambios
Hacer commit de los nuevos archivos implementados:
```bash
git add src/models/ tests/unit/test_baseline_models.py
git commit -m "feat: implement baseline Isolation Forest and LOF anomaly detectors (Issue #10)"
```

### 7. Integrar con Develop
Hacer el merge de los cambios aprobados en `develop` y pushear:
```bash
git checkout develop
git merge feature/issue-10-baseline-models
git push origin develop
```
