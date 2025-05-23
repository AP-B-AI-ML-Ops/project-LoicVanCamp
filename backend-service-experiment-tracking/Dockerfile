FROM ghcr.io/mlflow/mlflow:latest

RUN pip install -U pip
RUN pip install psycopg2-binary mlflow

CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "postgresql+psycopg2://postgres:postgres@backend-database:5432/mlflow_db"]
