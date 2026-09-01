# Issue #15: Configure Docker Compose Infrastructure (Postgres, MinIO, MLflow, Airflow)

El objetivo de este issue es definir y configurar la orquestación de infraestructura mediante **Docker Compose**. Se desplegará una base de datos PostgreSQL (backend de metadatos), MinIO (almacenamiento S3 compatible para data lake y artefactos), un servidor de MLflow y el clúster de Apache Airflow (orquestación).

## Pasos a Seguir

### 1. Preparar la Rama de Trabajo
Crear la nueva rama `feature/issue-15-docker-compose` a partir de `develop` (rama principal de desarrollo) y asegurar el entorno:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/issue-15-docker-compose
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Crear Estructura y Archivos de Soporte
Crear el directorio `docker/` para almacenar configuraciones adicionales (ej. scripts de inicialización):
- Crear `docker/postgres/init.sql` para crear automáticamente las bases de datos de Airflow y MLflow (ya que Postgres por defecto solo crea una).
- Crear el archivo `.env.example` en la raíz con las variables de entorno base (credenciales de Postgres, acceso a MinIO, etc.).

### 3. Definir el `docker-compose.yml`
Crear el archivo `docker-compose.yml` en la raíz del proyecto definiendo los siguientes servicios:
- **`postgres`**: Imagen de `postgres:15`, montando un volumen local para persistencia y cargando el `init.sql`.
- **`minio`**: Imagen de `minio/minio`, configurado con sus credenciales de root y un volumen local.
- **`minio-create-buckets`** (opcional/init): Contenedor temporal (usando `minio/mc`) que cree los buckets necesarios (ej. `mlflow-artifacts`, `data-lake`) al iniciar la primera vez.
- **`mlflow`**: Imagen base de Python o un Dockerfile personalizado, ejecutando `mlflow server` conectado a la base de datos de Postgres y usando MinIO como `default-artifact-root`.
- **`airflow-init`**, **`airflow-webserver`**, **`airflow-scheduler`**: Servicios de Apache Airflow conectados a Postgres, con volúmenes montados hacia los dags locales (si aplica en un futuro) y las configuraciones de Celery o LocalExecutor.

### 4. Documentación
Crear el archivo `docs/infrastructure.md` o actualizar el `README.md` detallando las instrucciones para levantar el entorno:
```bash
cp .env.example .env
docker compose up -d
```
E incluir los puertos de acceso (ej. MLflow en 5000, MinIO Console en 9001, Airflow en 8080).

### 5. Validar Calidad de Código
Ejecutar validaciones sobre los nuevos archivos (aunque son YAML, es buena práctica comprobar si se rompieron scripts de Python y mantener la limpieza general):
```bash
ruff check . --fix
mypy src/ tests/
```

### 6. Commit y Merge
Integrar los cambios a `develop`:
```bash
git add docker-compose.yml .env.example docker/ docs/
git commit -m "feat: configure docker-compose stack for MLflow, MinIO, Postgres and Airflow (Issue #15)"
git checkout develop
git merge feature/issue-15-docker-compose
git push origin develop
```
