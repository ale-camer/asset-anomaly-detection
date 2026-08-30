# Issue #4: Implement Local & Lake Storage Layer (Parquet Sink)

El objetivo de este issue es construir la capa de persistencia (storage layer) encargada de escribir los datos de mercado validados en formato Parquet. Los datos deben ser particionados lógicamente por activo (symbol) y fecha para optimizar consultas futuras.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-4-storage-layer` a partir de `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-4-storage-layer
```

### 2. Actualizar Dependencias
Agregar `pyarrow` al archivo `pyproject.toml` (necesario como motor de Pandas para leer/escribir Parquet eficientemente) y actualizar el entorno:
```bash
# Nota: primero se agregará pyarrow a las dependencias en pyproject.toml
pip install -e ".[dev]"
```

### 3. Implementar el Parquet Sink
Crear el archivo `src/storage/parquet_sink.py`:
- Implementar una clase `ParquetStorageSink` (o similar).
- Implementar un método para persistir un `pandas.DataFrame`.
- Configurar el motor de pandas (`pyarrow`) y especificar `partition_cols=['symbol', 'date']` para generar la estructura de carpetas `symbol=BTC/date=2024-01-01`.

### 4. Escribir Tests Unitarios
Crear `tests/unit/test_storage.py`:
- Utilizar un DataFrame ficticio basado en el esquema de datos esperado.
- Usar el fixture `tmp_path` de pytest para escribir el archivo localmente.
- Verificar que el archivo Parquet se escriba exitosamente y que se cree la estructura de particiones correcta.

### 5. Verificar Calidad y Tests
Correr la suite de pruebas y linters sobre los cambios nuevos:
```bash
python -m pytest -o addopts="" tests/unit/test_storage.py
ruff check . --fix
mypy src/ tests/
```

### 6. Commit de los Cambios
Hacer commit de los nuevos archivos y de las dependencias actualizadas:
```bash
git add pyproject.toml src/storage/ tests/unit/test_storage.py
git commit -m "feat: implement parquet storage sink partitioned by asset and date (Issue #4)"
```

### 7. Integrar con Develop
Hacer el merge de los cambios aprobados en `develop` y pushear:
```bash
git checkout develop
git merge feature/issue-4-storage-layer
git push origin develop
```
