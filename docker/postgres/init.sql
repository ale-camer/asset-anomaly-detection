-- Initialize multiple databases for MLflow and Airflow metadata

SELECT 'CREATE DATABASE mlflow_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db')\gexec

SELECT 'CREATE DATABASE airflow_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow_db')\gexec

GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlops_user;
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO mlops_user;
