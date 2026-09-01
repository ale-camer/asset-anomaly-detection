# Issue #12: Setup MLflow Experiment Tracking, Parameter & Metric Logging

El objetivo de este issue es integrar MLflow para el seguimiento de experimentos, registrando hiperparámetros, métricas de pérdida de reconstrucción (del Autoencoder) y métricas de anomalías (como el umbral/threshold).

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-12-mlflow` a partir de `develop` y asegurar el entorno:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-12-mlflow
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Integrar MLflow en los Modelos
Dado que `mlflow` ya es una dependencia en `pyproject.toml`, debemos actualizar los modelos en `src/models/`:

- **`src/models/base.py`**: 
  - Añadir soporte para registrar parámetros de inicialización de los modelos en MLflow (si existe un run activo).
- **`src/models/autoencoder.py`**:
  - En el método `fit`, verificar si hay un run de MLflow activo.
  - Registrar hiperparámetros (`hidden_dim`, `latent_dim`, `lr`, `epochs`, `batch_size`).
  - Durante el bucle de entrenamiento, registrar el error de reconstrucción por época (`mlflow.log_metric("train_loss", loss.item(), step=epoch)`).
  - Tras calcular el umbral de anomalía, registrarlo (`mlflow.log_metric("anomaly_threshold", self.threshold)`).
- **`src/models/baseline.py`**:
  - En los modelos `IsolationForestDetector` y `LOFDetector`, registrar sus hiperparámetros (`n_estimators`, `contamination`, `n_neighbors`) de estar MLflow activo.

### 3. Escribir Tests Unitarios
Crear o actualizar los tests (por ejemplo, en `tests/unit/test_models.py` o un archivo específico `test_mlflow.py`) para verificar:
- Que los modelos sigan entrenando sin problemas cuando MLflow **no** está activo.
- Que, mediante un mock de `mlflow`, se validen las llamadas a `log_metric` y `log_param` durante el entrenamiento.

### 4. Verificar Calidad y Tests
Correr la suite de pruebas y linters sobre los cambios nuevos:
```bash
python -m pytest -o addopts="" tests/
ruff check . --fix
mypy src/ tests/
```

### 5. Commit de los Cambios
Hacer commit de los nuevos archivos y modificaciones:
```bash
git add src/models/ tests/
git commit -m "feat: implement MLflow tracking for parameters and metrics (Issue #12)"
```

### 6. Integrar con Develop
Hacer el merge de los cambios aprobados en `develop` y pushear:
```bash
git checkout develop
git merge feature/issue-12-mlflow
git push origin develop
```
