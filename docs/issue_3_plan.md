# Issue #3: Implement Pydantic Validation Schemas for Raw Ingested Market Data

El objetivo de este issue es definir modelos estrictos utilizando Pydantic v2 para validar los datos sin procesar ingestados, específicamente enfocados en instantáneas de precios de mercado, volumen y libros de órdenes (order-book snapshots).

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-3-pydantic-schemas` a partir de `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-3-pydantic-schemas
```

### 2. Implementar los Modelos de Validación (Pydantic v2)
Crear el archivo `src/ingestion/models.py` para definir los esquemas estrictos de Pydantic.
- Definir un modelo `PriceSnapshot` con validadores estrictos (ej. precio > 0).
- Definir un modelo `VolumeSnapshot` con validadores (ej. volumen >= 0).
- Definir un modelo `OrderBookSnapshot` que contenga listas de "bids" y "asks" (donde cada entrada es una tupla o modelo de `[precio, cantidad]`).
- Todos los modelos deben tener `model_config = ConfigDict(strict=True)` para asegurar una validación estricta de tipos de Pydantic v2.

### 3. Escribir Tests Unitarios
Garantizar que las validaciones de los modelos funcionen como se espera y rechacen datos inválidos.
- Crear `tests/unit/test_models.py`.
- Usar `pytest` para probar la instanciación exitosa con datos correctos.
- Probar que se levanten excepciones de tipo `ValidationError` al pasar tipos incorrectos (debido a `strict=True`) o valores inválidos (como precios negativos).

### 4. Verificar Calidad y Tests
Correr la suite de pruebas y linters sobre los cambios nuevos:
```bash
python -m pytest -o addopts="" tests/unit/test_models.py
ruff check . --fix
mypy src/ tests/
```

### 5. Commit de los Cambios
Hacer commit de los nuevos archivos implementados:
```bash
git add src/ingestion/models.py tests/unit/test_models.py
git commit -m "feat: implement strict pydantic v2 schemas for market data (Issue #3)"
```

### 6. Integrar con Develop
Hacer el merge de los cambios aprobados en `develop` y pushear:
```bash
git checkout develop
git merge feature/issue-3-pydantic-schemas
git push origin develop
```
