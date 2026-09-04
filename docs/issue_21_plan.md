# Issue #21: Implement Streamlit Real-Time Anomaly Dashboard & Alerting UI

El objetivo de este issue es construir una interfaz gráfica e interactiva con Streamlit para la visualización en tiempo real de anomalías en activos financieros y criptomonedas. La aplicación permitirá monitorear series temporales con marcas visuales de anomalías, interactuar directamente con el servicio de inferencia de FastAPI para realizar predicciones bajo demanda, gestionar alertas según la severidad del puntaje de anomalía, y visualizar reportes de Data Drift generados por Evidently AI.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-21-streamlit-dashboard` a partir de `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-21-streamlit-dashboard
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Implementar el Cliente de API y Componentes de Alerta
Crear el paquete `src/ui/`:
- **`src/ui/__init__.py`**: Inicialización del módulo de interfaz gráfica.
- **`src/ui/api_client.py`**:
  - Implementar un cliente HTTP robusto utilizando `httpx` para comunicarse con la API de inferencia (FastAPI).
  - Funciones principales:
    - `check_api_health(base_url: str) -> dict`: Consulta el endpoint `GET /health` para verificar la disponibilidad del servicio y la carga del modelo.
    - `predict_anomalies(base_url: str, features: list[dict]) -> dict`: Envía el payload a `POST /predict` y retorna las predicciones y scores.
    - Manejo seguro de timeouts, errores de conexión y respuestas no exitosas para garantizar que la UI permanezca funcional ante caídas del backend.
- **`src/ui/components.py`**:
  - Funciones auxiliares para determinar el nivel de alerta según el score continuo (`Low`, `Medium`, `High`, `Critical`).
  - Renderizado de tarjetas de métricas (`st.metric`) con conteo de anomalías y ratios.
  - Componente para renderizar de forma segura el reporte HTML de Data Drift generado por Evidently AI (`data/processed/reports/data_drift_report.html`).

### 3. Construir la Aplicación con Streamlit
Crear `src/ui/app.py` estructurando la vista mediante pestañas interactivas o navegación lateral:
- **Vista de Series Temporales y Anomalías**:
  - Carga y visualización de datos históricos de precios e indicadores técnicos (medias móviles, volatilidad, RSI).
  - Marcado visual de eventos anómalos detectados a lo largo del tiempo.
- **Simulador de Inferencia en Tiempo Real**:
  - Formulario interactivo con controles deslizantes (sliders) e inputs numéricos para definir valores de mercado simulados.
  - Botón para ejecutar la predicción contra la API FastAPI (`POST /predict`).
  - Despliegue inmediato del resultado (`is_anomaly`), el score numérico y banners de alerta contextuales.
- **Observabilidad y Monitoreo de Drift**:
  - Panel con el estado en vivo del servicio FastAPI y versión del modelo en producción.
  - Pestaña para inspeccionar visualmente el último reporte de Data Drift de Evidently AI.

### 4. Soporte en Docker e Infraestructura
- **`docker/streamlit/Dockerfile`**: Definir la imagen para empaquetar el dashboard de Streamlit basado en `python:3.11-slim`, exponiendo el puerto 8501.
- **`docker-compose.yml`**:
  - Incorporar el servicio `streamlit` conectado a `anomaly-network`.
  - Configurar mapeo de puertos `8501:8501`.
  - Inyectar la variable de entorno `API_BASE_URL` apuntando al servicio de inferencia.

### 5. Escribir Tests Unitarios
Crear `tests/unit/test_ui.py` para asegurar la confiabilidad del cliente y la lógica de la UI:
- **`test_check_api_health`**: Validar respuestas exitosas (200 OK) y degradadas (503 / errores de conexión) mediante mock de `httpx`.
- **`test_predict_anomalies`**: Verificar serialización de features hacia `POST /predict` y deserialización de la respuesta.
- **`test_alert_severity_logic`**: Validar la clasificación de severidad de alertas para diferentes rangos de scores y umbrales.
- **`test_data_formatting`**: Comprobar la transformación de datos para gráficos y visualizaciones.

### 6. Validar Calidad de Código
Ejecutar la suite completa de pruebas y linters:
```bash
python -m pytest tests/
ruff check . --fix
mypy src/ tests/
```

### 7. Commit y Merge
```bash
git add src/ui/ docker/ docker-compose.yml tests/unit/test_ui.py docs/issue_21_plan.md
git commit -m "feat: implement Streamlit real-time anomaly dashboard and alerting UI (Issue #21)"
git checkout develop
git merge feature/issue-21-streamlit-dashboard
git push origin develop
```
