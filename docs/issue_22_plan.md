# Issue #22: Setup GitHub Actions CI/CD Pipeline & Documentation Finalization

El objetivo de este issue es establecer la infraestructura de Integración Continua (CI) mediante GitHub Actions para automatizar las pruebas, validaciones de estilo y chequeos de tipos estáticos ante cada Pull Request o push a las ramas principales. Asimismo, se finalizará la documentación del proyecto (README y checklist maestro) consolidando la finalización del Milestone 5 y el cierre exitoso del sistema MLOps de detección de anomalías.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-22-ci-cd-and-docs` a partir de `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-22-ci-cd-and-docs
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Implementar el Pipeline de CI con GitHub Actions
Crear el directorio `.github/workflows/` y el archivo de workflow `.github/workflows/ci.yml`:
- **Disparadores (Triggers)**:
  - Eventos `push` en las ramas `main` y `develop`.
  - Eventos `pull_request` dirigidos a `main` y `develop`.
- **Estructura del Job (`test-and-lint`)**:
  - Matriz de entorno con `ubuntu-latest` y Python `3.11` (y opcionalmente `3.12`).
  - Pasos (Steps):
    1. `actions/checkout@v4`: Clonado del repositorio.
    2. `actions/setup-python@v5`: Configuración del runtime de Python con caché de pip activada.
    3. Instalación de dependencias: `pip install --upgrade pip` e instalación de `.[dev]`.
    4. Chequeo de formateo y estilo: `ruff check .`
    5. Análisis estático de tipos: `mypy src/ tests/`
    6. Ejecución de pruebas con cobertura: `pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing`
    7. Carga de reporte de cobertura (opcional artifact o badge).
- **Validación de Construcción de Contenedores (`docker-build-check`)**:
  - Validar sintaxis y armado de la infraestructura:
    - Validación de archivo `docker compose config`.
    - Prueba de construcción (build) para las imágenes de `docker/mlflow/Dockerfile` y `docker/streamlit/Dockerfile`.

### 3. Finalización y Consolidación de la Documentación
- **Actualizar `README.md`**:
  - Incorporar el badge de estado de CI de GitHub Actions (`Build & Tests`).
  - Marcar como completados todos los hitos y los 22 issues en la sección de Milestones & Roadmap.
  - Documentar las instrucciones de uso para:
    - Servicio de inferencia FastAPI (`GET /health`, `POST /predict`, `GET /metrics`).
    - Dashboard interactivo de Streamlit (`http://localhost:8501`).
    - Monitoreo de Data Drift con Evidently AI.
    - Orquestación con Apache Airflow y Docker Compose.
- **Actualizar `docs/issue_0_setup.md`**:
  - Marcar con `[x]` todos los issues del 1 al 22 como completados.
  - Dejar el registro maestro de Day 0 a Day Final actualizado.

### 4. Validar Calidad de Código Localmente
Verificar que la suite completa pase sin advertencias ni errores antes de publicar el flujo:
```bash
python -m pytest tests/
ruff check . --fix
mypy src/ tests/
```

### 5. Commit y Merge
```bash
git add .github/ README.md docs/
git commit -m "ci: setup GitHub Actions pipeline and finalize documentation (Issue #22)"
git checkout develop
git merge feature/issue-22-ci-cd-and-docs
git push origin develop
```

### 6. Cierre del Proyecto (Release v1.0.0)
Una vez validada la integración en `develop`:
```bash
git checkout main
git merge develop
git tag -a v1.0.0 -m "Release v1.0.0: Full MLOps platform for asset anomaly detection"
git push origin main --tags
```
