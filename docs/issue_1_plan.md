# Issue #1: Setup Data Ingestion Configuration & Base Connector Interface

The goal of this issue is to define the configuration management system and abstract base classes required for market data ingestion sources. This provides the foundational architecture for fetching, validating, and saving financial data before implementing specific API clients (like CoinGecko).

## Pasos a Seguir

### 1. Preparar el Entorno y la Rama de Trabajo
Instalar las dependencias de desarrollo y crear la nueva rama `feature/issue-1-ingestion-base` a partir de `develop`:
```bash
pip install -e ".[dev]"
git checkout develop
git pull origin develop
git checkout -b feature/issue-1-ingestion-base
```

### 2. Implementar Configuration Management
Utilizar `pydantic-settings` para el manejo de variables de entorno de `.env`.
- Crear el archivo `src/utils/config.py`.
- Definir la clase `Settings` heredando de `pydantic_settings.BaseSettings`.
- Definir el modelo de settings, base de datos, URLs y API keys.
- Crear una instancia global `settings` disponible para toda la app.

### 3. Implementar Base Connector Interface
Definir la clase abstracta para todos los conectores de ingesta usando `abc` y `pydantic`.
- Crear el archivo `src/ingestion/base.py`.
- Definir la clase abstracta `BaseConnector` usando `abc.ABC`.
- Agregar los métodos abstractos `fetch_data(symbol, start_date, end_date)` y `save_data(data, path)`.
- Definir un esquema (modelo Pydantic) estándar para los datos devueltos por cualquier conector.

### 4. Escribir Tests Unitarios
Garantizar la correcta instanciación de configuración y clases abstractas.
- Crear `tests/unit/test_config.py` para verificar que `Settings` levante bien el `.env`.
- Crear `tests/unit/test_ingestion_base.py` para asegurar que `BaseConnector` no puede instanciarse directamente sin implementar sus métodos abstractos.

### 5. Verificar Calidad y Tests
Correr la suite de pruebas y linters sobre los cambios nuevos:
```bash
pytest tests/unit/test_config.py tests/unit/test_ingestion_base.py
ruff check .
mypy src/ tests/
```

### 6. Commit de los Cambios
Hacer commit de los nuevos archivos implementados:
```bash
git add src/utils/config.py src/ingestion/base.py tests/unit/test_config.py tests/unit/test_ingestion_base.py
git commit -m "feat: implement config and base connector for issue 1"
```

### 7. Integrar con Develop
Hacer el merge de los cambios aprobados en `develop` y pushear:
```bash
git checkout develop
git merge feature/issue-1-ingestion-base
git push origin develop
```
