# Issue #5: Implement Unit & Integration Tests for Data Ingestion Module

El objetivo de este issue es asegurar la robustez del módulo de ingestión escribiendo pruebas de integración que comprueben el flujo completo: desde la obtención de datos (con respuestas de API mockeadas para no depender de la red real), hasta su validación con Pydantic y su almacenamiento en formato Parquet utilizando el Sink de almacenamiento.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-5-integration-tests` a partir de `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-5-integration-tests
```

### 2. Escribir Tests de Integración
Crear el archivo `tests/integration/test_ingestion_integration.py`:
- Configurar un mock para la API usando `unittest.mock.patch` sobre `httpx.Client.get`.
- Instanciar `CoinGeckoConnector` y recuperar datos simulados.
- Instanciar `ParquetStorageSink` y persistir el DataFrame resultante en un directorio temporal provisto por el fixture `tmp_path`.
- Leer el dataset guardado en formato Parquet y verificar que los datos hagan un round-trip correcto sin pérdida de integridad.

### 3. Verificar Calidad y Tests
Correr toda la suite de pruebas (unitarias y de integración) junto con linters sobre los cambios nuevos:
```bash
python -m pytest -o addopts="" tests/
ruff check . --fix
mypy src/ tests/
```

### 4. Commit de los Cambios
Hacer commit de los nuevos archivos implementados:
```bash
git add tests/integration/test_ingestion_integration.py
git commit -m "test: implement integration tests for data ingestion to storage (Issue #5)"
```

### 5. Integrar con Develop y Main
Hacer el merge de los cambios aprobados en `develop`, luego hacer el merge de `develop` a `main` y pushear:
```bash
git checkout develop
git merge feature/issue-5-integration-tests
git push origin develop

git checkout main
git merge develop
git push origin main
```
