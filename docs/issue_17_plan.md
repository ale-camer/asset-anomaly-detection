# Issue #17: Implement Airflow DAG for Automated Model Retraining & Validation

El objetivo de este issue es crear un DAG de Apache Airflow que automatice el ciclo de vida del modelo de detección de anomalías. Este flujo se encargará de re-entrenar el modelo con datos frescos del Feature Store, calcular sus umbrales óptimos, evaluar su rendimiento y, en caso de superar los criterios de aceptación, promoverlo en el MLflow Model Registry.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-17-retraining-dag` a partir de `develop` y preparar el entorno virtual:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-17-retraining-dag
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Implementar el DAG de Re-entrenamiento
Crear el archivo `dags/model_retraining_pipeline.py` y definir el DAG con la siguiente estructura de tareas:
- **Tarea 1 (`fetch_training_data`)**: Extrae el histórico reciente desde el Feature Store (`src.features.store`) preparándolo como un dataset tabular.
- **Tarea 2 (`train_anomaly_model`)**: Inicializa y entrena el modelo de detección (p. ej. `IsolationForestBaseline` o el Autoencoder) utilizando las features extraídas.
- **Tarea 3 (`calculate_thresholds`)**: Utiliza `src.models.evaluation` (por ejemplo, umbrales EVT o por percentil) para determinar el punto de corte óptimo de anomalías sobre un set de validación.
- **Tarea 4 (`validate_and_register_model`)**: Compara las métricas del modelo entrenado contra la versión productiva actual. Si las métricas son aceptables, utiliza `log_model_to_mlflow` y `save_model_artifact` (en `src.models.registry`) para guardar el modelo, loguear las métricas y exportar la serialización final (incluyendo ONNX).

**Configuración del DAG**:
- **Schedule**: `@weekly` (o un cron de mantenimiento los domingos).
- **Control de errores**: Manejo seguro del import de `airflow` (`AIRFLOW_AVAILABLE`) igual que se hizo en el Issue 16.
- **Dependencias**: `fetch_training_data >> train_anomaly_model >> calculate_thresholds >> validate_and_register_model`.

### 3. Agregar Tests Unitarios al DAG
Actualizar (o crear un nuevo archivo si es necesario, p. ej. `tests/unit/test_retraining_dag.py`) para validar:
- La correcta estructura del DAG y sus dependencias (ignorándolo si `airflow` no está localmente instalado).
- Mocks para asegurar que las tareas invocan correctamente a MLflow, a los modelos y al Feature Store.

### 4. Validar Calidad de Código
Ejecutar la validación completa sobre los nuevos archivos:
```bash
python -m pytest tests/
ruff check . --fix
mypy dags/ src/ tests/
```

### 5. Commit y Merge
```bash
git add dags/ tests/
git commit -m "feat: implement Airflow DAG for model retraining and registry promotion (Issue #17)"
git checkout develop
git merge feature/issue-17-retraining-dag
git push origin develop
```
