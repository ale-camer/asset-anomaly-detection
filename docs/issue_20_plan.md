# Issue #20: Implement Data Drift & Concept Drift Monitoring (Evidently / Prometheus)

El objetivo de este issue es agregar observabilidad y monitoreo al sistema. Por un lado, se instrumentará el servicio de inferencia con métricas de Prometheus en tiempo real (para tasa de anomalías y latencia). Por otro lado, se integrará Evidently AI para calcular el corrimiento de datos (Data Drift) comparando el tráfico de inferencia con los datos originales de entrenamiento.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-20-drift-monitoring` a partir de `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-20-drift-monitoring
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Instrumentar Métricas con Prometheus
Crear el archivo `src/api/metrics.py` (o similar) para definir las métricas usando `prometheus_client`:
- **`PREDICTION_REQUESTS`** (`Counter`): Contador de peticiones totales.
- **`ANOMALIES_DETECTED`** (`Counter`): Contador de anomalías confirmadas.
- **`ANOMALY_SCORE_GAUGE`** (`Gauge` o `Histogram`): Para registrar la distribución de los scores de anomalía en tiempo real.

**Modificar `src/api/main.py`**:
- Exponer el endpoint `GET /metrics` importando `make_asgi_app` de `prometheus_client`.
- En el endpoint `POST /predict`, incrementar estas métricas según los resultados devueltos por el modelo.

### 3. Implementar Reportes de Drift con Evidently AI
Crear el módulo `src/monitoring/drift.py`:
- Crear una función `generate_drift_report(reference_df, current_df)` que utilice `Report(metrics=[DataDriftPreset()])` de la librería `evidently`.
- La función debe comparar la distribución de las features en ambos datasets y guardar el reporte en disco (formato JSON o HTML).
- Esto permitirá en un futuro orquestarlo vía Airflow para emitir alertas si la distribución del mercado cambia drásticamente respecto al último entrenamiento.

### 4. Escribir Tests Unitarios
Crear `tests/unit/test_monitoring.py`:
- **Test de Prometheus**: Usar el `TestClient` para asegurar que tras llamar a `/predict`, el endpoint `/metrics` devuelve los contadores incrementados con la sintaxis de Prometheus.
- **Test de Evidently**: Pasar dos DataFrames simulados (uno normal y otro alterado) a `generate_drift_report` y verificar que genera el archivo JSON o HTML del reporte de drift sin errores.

### 5. Validar Calidad de Código
Ejecutar la suite de tests y linters:
```bash
python -m pytest tests/
ruff check . --fix
mypy src/ tests/
```

### 6. Commit y Merge
```bash
git add src/ tests/
git commit -m "feat: implement Prometheus metrics and Evidently AI drift monitoring (Issue #20)"
git checkout develop
git merge feature/issue-20-drift-monitoring
git push origin develop
```
