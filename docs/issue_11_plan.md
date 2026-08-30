# Issue #11: Implement Deep Learning / Autoencoder Anomaly Detector

El objetivo de este issue es construir un detector de anomalías basado en Deep Learning utilizando una arquitectura de Autoencoder con PyTorch. El modelo aprenderá la representación latente y reconstrucción de patrones normales de mercado, utilizando el error de reconstrucción (Reconstruction MSE Error) como métrica continua de anomalía.

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-11-autoencoder` a partir de `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-11-autoencoder
```

### 2. Actualizar Dependencias
Agregar `torch>=2.2.0` en `pyproject.toml` dentro de `dependencies` (limpiando duplicados si existieran) e instalar el entorno:
```bash
pip install -e ".[dev]"
```

### 3. Implementar el Autoencoder con PyTorch
Crear el archivo `src/models/autoencoder.py`:
- Definir la red neuronal `AutoencoderNetwork(nn.Module)` con capas Encoder (compresión) y Decoder (reconstrucción).
- Implementar la clase adaptadora `AutoencoderAnomalyDetector(BaseAnomalyDetector)`:
  - `fit(X)`: Entrenar el autoencoder minimizando el error cuadrático medio (MSE Loss) con optimizador Adam.
  - `score_samples(X)`: Calcular el error de reconstrucción por muestra (MSE continuo).
  - `predict(X)`: Clasificar como anomalía (1) si el error de reconstrucción supera el umbral configurado (ej. percentil o $\mu + k \cdot \sigma$).

### 4. Escribir Tests Unitarios
Crear `tests/unit/test_autoencoder.py`:
- Validar convergencia básica del entrenamiento con datos sintéticos.
- Comprobar que muestras fuera de distribución (outliers) obtengan errores de reconstrucción significativamente superiores a las normales.
- Validar soporte con DataFrames (`feature_cols`), arrays de NumPy, y detección de modelos no ajustados.

### 5. Verificar Calidad y Tests
Correr la suite de pruebas y linters sobre los cambios nuevos:
```bash
python -m pytest -o addopts="" tests/unit/test_autoencoder.py
ruff check . --fix
mypy src/ tests/
```

### 6. Commit de los Cambios
Hacer commit de los nuevos archivos implementados:
```bash
git add pyproject.toml src/models/ tests/unit/test_autoencoder.py
git commit -m "feat: implement deep learning autoencoder anomaly detector (Issue #11)"
```

### 7. Integrar con Develop
Hacer el merge de los cambios aprobados en `develop` y pushear:
```bash
git checkout develop
git merge feature/issue-11-autoencoder
git push origin develop
```
