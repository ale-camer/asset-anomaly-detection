# Issue #16: Implement Airflow DAG for Scheduled Ingestion & Feature Transformation

El objetivo de este issue es crear un DAG (Directed Acyclic Graph) en Apache Airflow que automatice y orqueste diariamente (o por hora) la ingesta de datos del mercado, su validación y la consecuente computación de features.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-16-airflow-dag` a partir de `develop` y preparar el entorno de Python:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-16-airflow-dag
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Implementar el DAG de Ingesta y Features
Crear el archivo `dags/ingestion_feature_pipeline.py` y definir el DAG con la siguiente estructura de tareas (usando `PythonOperator` o la API de TaskFlow `@task`):
- **Tarea 1 (`ingest_market_data`)**: Instancia el cliente de CoinGecko (o el orquestador de ingesta) y extrae los datos crudos recientes, guardándolos en el Data Lake local (directorio `raw` o en MinIO si ya está conectado).
- **Tarea 2 (`validate_raw_data`)**: Valida el esquema y la completitud de los datos recién ingeridos, asegurando que no haya corrupciones o faltantes masivos.
- **Tarea 3 (`compute_features`)**: Ejecuta el pipeline de features (`src.features.transformers` / `src.features.pipeline`), calcula indicadores técnicos, retornos, etc., y persiste el resultado en el Feature Store (directorio `features`).

**Configuración del DAG**:
- **Schedule**: `@daily` (o un cron específico, ej. `0 2 * * *`).
- **Retries**: 3 reintentos con un delay de 5 minutos (para mitigar intermitencias en las APIs de mercado).
- **Dependencias**: `ingest_market_data >> validate_raw_data >> compute_features`.

### 3. Tests Unitarios del DAG (Opcional pero Recomendado)
Crear `tests/unit/test_dags.py` para asegurar que el DAG fue interpretado correctamente por Airflow sin errores de sintaxis o dependencias cíclicas:
- Cargar la `DagBag` de Airflow y validar que `ingestion_feature_pipeline` exista y no tenga errores de importación.

### 4. Validar Calidad de Código
Correr las verificaciones habituales para asegurar que los nuevos archivos Python sigan los estándares:
```bash
python -m pytest tests/
ruff check . --fix
mypy dags/ src/ tests/
```

### 5. Commit y Merge
```bash
git add dags/ tests/
git commit -m "feat: implement Airflow DAG for daily ingestion and feature computation (Issue #16)"
git checkout develop
git merge feature/issue-16-airflow-dag
git push origin develop
```
