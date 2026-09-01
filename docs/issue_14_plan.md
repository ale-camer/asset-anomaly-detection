# Issue #14: Implement Model Registry Packaging, Artifact Serialization & Unit Tests

El objetivo de este issue es implementar el empaquetado de modelos, la serialización de artefactos (por ejemplo, vía MLflow y ONNX) y completar la suite de tests unitarios para los componentes de modelado. Al finalizar, los cambios se deben integrar directamente a la rama `main`, tal como fue solicitado.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-14-registry` a partir de `main` (rama de integración objetivo) y asegurar que el entorno esté preparado:
```bash
git checkout main
git pull origin main
git checkout -b feature/issue-14-registry
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Implementar Empaquetado y Serialización de Modelos
Crear o actualizar el módulo de registro (ej. `src/models/registry.py`) para manejar la serialización de los modelos:
- **Integración con MLflow Model Registry**:
  - Implementar funciones genéricas para guardar y cargar los modelos entrenados (`IsolationForestDetector`, `LOFDetector`, `AutoencoderAnomalyDetector`) utilizando los artefactos nativos de MLflow (como `mlflow.sklearn.log_model` o `mlflow.pytorch.log_model`).
- **Soporte de ONNX (Opcional/Avanzado)**:
  - Implementar la exportación del autoencoder de PyTorch a formato ONNX (usando `torch.onnx.export`) para inferencia optimizada y ligera.
- **Gestión de Versiones**:
  - Asegurar que al empaquetar un modelo, este quede versionado en el Model Registry de MLflow, de manera que la API de inferencia pueda luego solicitar "la última versión en Producción".

### 3. Escribir Tests Unitarios
Crear `tests/unit/test_registry.py` (o similar) para verificar:
- Que los modelos pueden guardarse y recargarse exitosamente sin perder su estado interno ni hiperparámetros.
- Que un modelo recargado arroje exactamente las mismas predicciones y *scores* que el modelo en memoria justo antes de serializarse.
- Validar el proceso de exportación a ONNX (verificando la estructura del grafo si la librería ONNX está instalada).

### 4. Verificar Calidad y Tests
Correr la suite de pruebas y linters sobre los componentes de modelado para garantizar su cobertura:
```bash
python -m pytest tests/
ruff check . --fix
mypy src/ tests/
```

### 5. Commit y Merge con Main
Una vez que las pruebas sean exitosas, integrar directamente a la rama `main`:
```bash
git add src/models/ tests/
git commit -m "feat: implement model packaging, ONNX/MLflow serialization and tests (Issue #14)"
git checkout main
git merge feature/issue-14-registry
git push origin main
```
