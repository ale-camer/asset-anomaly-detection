# Issue #8: Implement Feature Persistence Layer & Schema Validation

El objetivo de este issue es establecer una capa de persistencia para guardar los datasets de características (features) calculados, garantizando la consistencia de los datos antes de guardarlos mediante esquemas de validación estrictos (Pydantic). 

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-8-feature-persistence` a partir de `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-8-feature-persistence
```

### 2. Definir los Esquemas de Validación
Crear el archivo `src/features/models.py`:
- Definir un modelo Pydantic v2 estricto `FeatureSetRecord` (o similar).
- Configurar la validación para las nuevas métricas (medias móviles, volatilidad, momentum).

### 3. Implementar el Feature Store / Persistence Layer
Crear el archivo `src/features/store.py`:
- Implementar la clase `FeatureStoreSink` (pudiendo heredar o componer `ParquetStorageSink`).
- Validar el DataFrame contra el esquema `FeatureSetRecord` antes de la persistencia.
- Guardar los datos en el directorio correspondiente (`settings.data_features_dir`), particionado por activo y fecha.

### 4. Escribir Tests Unitarios
Crear `tests/unit/test_feature_store.py`:
- Verificar que la validación de esquema rechace características faltantes o con tipos incorrectos (NaNs imprevistos).
- Comprobar que la persistencia en formato Parquet se realice correctamente en los directorios esperados.

### 5. Verificar Calidad y Tests
Correr la suite de pruebas y linters sobre los cambios nuevos:
```bash
python -m pytest -o addopts="" tests/unit/
ruff check . --fix
mypy src/ tests/
```

### 6. Commit de los Cambios
Hacer commit de los nuevos archivos implementados:
```bash
git add src/features/ tests/unit/
git commit -m "feat: implement feature persistence layer with schema validation (Issue #8)"
```

### 7. Integrar con Develop y Main
Hacer el merge de los cambios aprobados en `develop`, luego en `main` y pushear:
```bash
git checkout develop
git merge feature/issue-8-feature-persistence
git push origin develop

git checkout main
git merge develop
git push origin main
```
