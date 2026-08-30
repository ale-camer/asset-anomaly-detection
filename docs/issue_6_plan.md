# Issue #6: Implement Time-Series Windowing & Rolling Statistics Transformers

El objetivo de este issue es construir los transformadores encargados de calcular características basadas en el tiempo (feature engineering) para los datos de mercado. Se deben implementar cálculos de media móvil (rolling mean), desviación estándar móvil (rolling std) y medias móviles exponenciales (EMA) sobre ventanas de tiempo definidas.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-6-transformers` a partir de `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-6-transformers
```

### 2. Implementar los Transformadores de Time-Series
Crear el archivo `src/features/transformers.py`:
- Implementar una clase (ej. `TimeSeriesRollingFeatures`) compatible con transformaciones de Pandas o Scikit-Learn.
- Añadir parámetros para las ventanas de tiempo (ej. 7, 14 o 30 periodos).
- Implementar el cálculo de _Rolling Mean_, _Rolling Std_ y _EMA_ para columnas clave como `close` y `volume`.

### 3. Escribir Tests Unitarios
Crear `tests/unit/test_transformers.py` para validar los cálculos:
- Proveer un DataFrame mock con datos secuenciales conocidos.
- Aplicar los transformadores.
- Verificar matemáticamente que la media móvil, EMA y desviación estándar tengan los valores esperados.

### 4. Verificar Calidad y Tests
Correr la suite de pruebas y linters sobre los cambios nuevos:
```bash
python -m pytest -o addopts="" tests/unit/test_transformers.py
ruff check . --fix
mypy src/ tests/
```

### 5. Commit de los Cambios
Hacer commit de los nuevos archivos implementados:
```bash
git add src/features/ tests/unit/test_transformers.py
git commit -m "feat: implement time-series rolling statistics transformers (Issue #6)"
```

### 6. Integrar con Develop y Main
Hacer el merge de los cambios aprobados en `develop`, luego en `main` y pushear:
```bash
git checkout develop
git merge feature/issue-6-transformers
git push origin develop

git checkout main
git merge develop
git push origin main
```
