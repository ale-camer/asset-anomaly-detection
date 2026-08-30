# Issue #7: Implement Volatility, Momentum & Technical Anomaly Feature Extractors

El objetivo de este issue es enriquecer la fase de Feature Engineering agregando transformadores que calculen indicadores técnicos avanzados utilizados habitualmente para identificar anomalías en series de tiempo financieras: Volatilidad de Parkinson, RSI (Relative Strength Index), MACD y Velocidad de Precio.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-7-technical-features` a partir de `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-7-technical-features
```

### 2. Implementar Transformadores Técnicos
Añadir las nuevas clases en `src/features/transformers.py` (o en un nuevo archivo `src/features/technical.py` si se prefiere separar):
- **`VolatilityFeatures`**: Calcular la Volatilidad de Parkinson (requiere columnas `high` y `low`).
- **`MomentumFeatures`**: Calcular RSI y MACD.
- **`PriceVelocityFeatures`**: Calcular la velocidad de cambio del precio (retornos logarítmicos o porcentuales, y su ratio de cambio).
- Todas estas clases deben seguir el estándar de scikit-learn (`BaseEstimator`, `TransformerMixin`) e incluir un soporte adecuado de agrupación (`group_by="symbol"`).

### 3. Escribir Tests Unitarios
Agregar pruebas en `tests/unit/test_transformers.py` (o en `test_technical.py`):
- Crear DataFrames simulados con `high`, `low` y `close`.
- Verificar que el cálculo de la Volatilidad de Parkinson sea matemáticamente correcto.
- Verificar que RSI fluctúe entre 0 y 100.
- Comprobar que MACD devuelva las series de línea MACD y Señal.
- Validar el aislamiento entre distintos activos usando múltiples símbolos.

### 4. Verificar Calidad y Tests
Correr la suite de pruebas y linters sobre los cambios nuevos:
```bash
python -m pytest -o addopts="" tests/unit/
ruff check . --fix
mypy src/ tests/
```

### 5. Commit de los Cambios
Hacer commit de los nuevos archivos implementados:
```bash
git add src/features/ tests/unit/
git commit -m "feat: implement volatility, momentum and velocity feature extractors (Issue #7)"
```

### 6. Integrar con Develop y Main
Hacer el merge de los cambios aprobados en `develop`, luego en `main` y pushear:
```bash
git checkout develop
git merge feature/issue-7-technical-features
git push origin develop

git checkout main
git merge develop
git push origin main
```
