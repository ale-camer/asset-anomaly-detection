# Issue #18: End-to-End Orchestration Testing with Mock Data & Healthchecks

El objetivo de este issue es asegurar la robustez de los pipelines completos (desde la ingesta de datos hasta el re-entrenamiento y registro del modelo) mediante pruebas de integración End-to-End (E2E). Se simulará el flujo de orquestación utilizando datos sintéticos y se validará el estado resultante en base de datos, almacenamiento de archivos y MLflow.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-18-e2e-testing` a partir de `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-18-e2e-testing
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Creación del Archivo de Pruebas E2E
Crear el archivo `tests/integration/test_e2e_orchestration.py`. Este archivo concentrará la ejecución secuencial de las funciones principales expuestas por nuestros DAGs.

**Fixtures necesarias (`conftest.py` o en el propio archivo)**:
- Directorios temporales (`tmp_path`) para simular el Data Lake (`data_raw_dir`), el Feature Store (`data_features_dir`) y el directorio de modelos (`models`).
- Mock de la API de mercado (ej. CoinGecko) para que retorne un DataFrame inicial controlado, evitando llamadas de red reales.
- Tracking URI local (basado en archivos locales o SQLite en memoria) para que MLflow no necesite el servidor de Postgres levantado para pasar los tests en CI.

### 3. Implementación del Flujo Completo
Programar un único test (o una clase de test secuencial) que simule la cadena de Airflow:
1. **Ingesta y Validación**: Ejecutar `ingest_market_data()` (mockeado) y `validate_raw_data()`. Comprobar que los archivos Parquet crudos se crearon.
2. **Generación de Features**: Ejecutar `compute_features()`. Comprobar que el Feature Store recibió los archivos Parquet procesados con las nuevas columnas (medias móviles, volatilidad, etc.).
3. **Consolidación de Entrenamiento**: Ejecutar `fetch_training_data()` apuntando al Feature Store local.
4. **Entrenamiento y Umbrales**: Ejecutar `train_anomaly_model()` y `calculate_thresholds()`. Comprobar que el `.pkl` y el arreglo `.scores.npy` fueron generados.
5. **Registro de MLflow**: Ejecutar `validate_and_register_model()`. Hacer aserciones (asserts) verificando que MLflow retornó un URI válido y que el modelo de producción se copió al directorio final.

### 4. Healthchecks y Validaciones del Estado
Incorporar aserciones que funcionen como "healthchecks" post-ejecución:
- Verificar existencia y permisos de los artefactos.
- Comprobar que el esquema del archivo Parquet final coincide con `FeatureSetRecord`.
- Inspeccionar si MLflow efectivamente tiene registrado un modelo con el nombre `asset-anomaly-detector`.

### 5. Validar Calidad de Código
Ejecutar la suite completa para asegurar que nada se rompió y que el test E2E pasa satisfactoriamente:
```bash
python -m pytest tests/
ruff check . --fix
mypy dags/ src/ tests/
```

### 6. Commit y Merge
```bash
git add tests/
git commit -m "test: implement E2E orchestration tests with mock data (Issue #18)"
git checkout develop
git merge feature/issue-18-e2e-testing
git push origin develop
```
