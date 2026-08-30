# Issue #9: Implement Unit Tests & Data Quality Verification for Feature Pipeline

El objetivo de este issue es asegurar que toda la lógica de transformación de características (feature engineering) funcione correctamente de principio a fin, validando el pipeline completo, verificando cómo se manejan los valores nulos (NaNs) generados por los desfases de las ventanas de tiempo, y probando casos borde.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-9-feature-pipeline-tests` a partir de `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-9-feature-pipeline-tests
```

### 2. Implementar Tests de Integración del Pipeline
Crear el archivo `tests/integration/test_feature_pipeline.py`:
- Configurar un `sklearn.pipeline.Pipeline` completo que encadene todos los transformadores (`TimeSeriesRollingFeatures`, `VolatilityFeatures`, `MomentumFeatures`, `PriceVelocityFeatures`).
- Proveer un dataset sintético realista con escenarios de casos borde (ej. varianza nula, missing values puntuales).
- Validar que el pipeline final produzca el esquema exacto esperado por el Feature Store sin perder datos por recortes accidentales, comprobando la correcta imputación o manejo de NaNs resultantes de las ventanas móviles.

### 3. Verificar Calidad y Tests
Correr la suite de pruebas y linters sobre los cambios nuevos:
```bash
python -m pytest -o addopts="" tests/
ruff check . --fix
mypy src/ tests/
```

### 4. Commit de los Cambios
Hacer commit de los nuevos archivos implementados:
```bash
git add tests/integration/test_feature_pipeline.py
git commit -m "test: implement data quality and pipeline integration tests (Issue #9)"
```

### 5. Integrar con Develop y Main
Al ser un hito (fin de pipeline), hacer el merge de los cambios aprobados en `develop`, luego en `main` y pushear:
```bash
git checkout develop
git merge feature/issue-9-feature-pipeline-tests
git push origin develop

git checkout main
git merge develop
git push origin main
```
