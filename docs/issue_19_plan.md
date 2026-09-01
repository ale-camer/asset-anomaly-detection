# Issue #19: Implement FastAPI Anomaly Scoring Inference Endpoint

El objetivo de este issue es construir el servicio de inferencia en tiempo real utilizando FastAPI. Esta API cargará el modelo de detección de anomalías directamente desde el Model Registry de MLflow y expondrá endpoints para verificar la salud del servicio y realizar predicciones online.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-19-fastapi-service` a partir de `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-19-fastapi-service
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Definir los Esquemas Pydantic
Crear el archivo `src/api/schemas.py` para validar la entrada y salida de la API:
- **`PredictionRequest`**: Define la estructura del payload que contiene las features requeridas por el modelo (e.g. `close`, `volume`, medias móviles, etc.). Puede aceptar un único registro o una lista (batch).
- **`PredictionResponse`**: Estructura de la respuesta que devolverá el score continuo de anomalía (`anomaly_score`) y el booleano resultante (`is_anomaly`).

### 3. Construir la API con FastAPI
Crear `src/api/main.py` implementando:
- **Gestión del Ciclo de Vida (Lifespan)**: Al iniciar la aplicación, cargar dinámicamente el último modelo en producción desde MLflow usando `load_model_from_mlflow(registered_model_name="asset-anomaly-detector")`. Mantenerlo en memoria.
- **Endpoint `GET /health`**: Retorna el estado del servicio y si el modelo está correctamente inicializado. Útil para Docker/Kubernetes probes.
- **Endpoint `POST /predict`**:
  1. Valida el payload de entrada (`PredictionRequest`).
  2. Transforma los datos a un DataFrame de pandas.
  3. Ejecuta el modelo en memoria para predecir.
  4. Retorna el resultado estructurado (`PredictionResponse`).

### 4. Escribir Tests Unitarios
Crear `tests/unit/test_api.py` utilizando el `TestClient` de FastAPI:
- **`test_health_check`**: Verifica que `GET /health` responda 200 OK.
- **`test_predict_endpoint`**:
  - Mockear el evento de inicio (startup) o la carga del modelo para inyectar un modelo falso (dummy detector) y no depender de MLflow local.
  - Enviar un payload JSON válido a `POST /predict` y asegurar que retorna el formato de predicción esperado con status 200.
  - Probar enviar un payload inválido para asegurar que FastAPI devuelve un error de validación 422.

### 5. Validar Calidad de Código
Ejecutar la suite completa antes de realizar los commits:
```bash
python -m pytest tests/
ruff check . --fix
mypy src/ tests/
```

### 6. Commit y Merge
```bash
git add src/api/ tests/unit/
git commit -m "feat: implement FastAPI inference service and endpoints (Issue #19)"
git checkout develop
git merge feature/issue-19-fastapi-service
git push origin develop
```
