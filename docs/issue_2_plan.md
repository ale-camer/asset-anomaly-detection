# Issue #2: Implement Financial & Crypto Market Data Ingestion Client

The goal of this issue is to create a concrete API client (using CoinGecko as the source) that inherits from the `BaseConnector` interface, implementing retry logic (`tenacity`), error handling, and rate limiting.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-2-coingecko-client` a partir de `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-2-coingecko-client
```

### 2. Actualizar Dependencias
Agregar `tenacity>=8.3.0` al archivo `pyproject.toml` en la lista de `dependencies` para manejar la lógica de reintentos e instalar las dependencias actualizadas:
```bash
pip install -e ".[dev]"
```

### 3. Implementar CoinGecko Connector
Crear la clase concreta `CoinGeckoConnector` implementando los métodos de `BaseConnector`.
- Crear el archivo `src/ingestion/coingecko.py`.
- Definir la clase `CoinGeckoConnector(BaseConnector)`.
- Utilizar `httpx` o `requests` para las llamadas HTTP a la API de CoinGecko.
- Añadir el decorador `@retry` de `tenacity` a las llamadas para manejar fallos temporales (como errores 5xx).
- Implementar rate limiting básico, teniendo en cuenta los límites de la API pública de CoinGecko (e.g. errores 429).

### 4. Escribir Tests Unitarios
Garantizar que el cliente reaccione correctamente ante distintas respuestas de la API.
- Crear `tests/unit/test_coingecko.py`.
- Usar `pytest` y mocks (`unittest.mock` o equivalentes) para simular respuestas exitosas, respuestas 429 (rate limit) y 500 (server error).
- Verificar que la librería de reintentos ejecute las llamadas nuevamente en caso de error.
- Validar que los datos retornados cumplan con el esquema `MarketDataRecord`.

### 5. Verificar Calidad y Tests
Correr la suite de pruebas y linters sobre los cambios nuevos:
```bash
python -m pytest -o addopts="" tests/unit/test_coingecko.py
ruff check . --fix
mypy src/ tests/
```

### 6. Commit de los Cambios
Hacer commit de los nuevos archivos implementados y las dependencias actualizadas:
```bash
git add pyproject.toml src/ingestion/coingecko.py tests/unit/test_coingecko.py
git commit -m "feat: implement CoinGecko API client with retry and rate limiting"
```

### 7. Integrar con Develop
Hacer el merge de los cambios aprobados en `develop` y pushear:
```bash
git checkout develop
git merge feature/issue-2-coingecko-client
git push origin develop
```
