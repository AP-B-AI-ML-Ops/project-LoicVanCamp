FROM python:3.13-slim

RUN pip install -U pip

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "postgresql+psycopg2://postgres:postgres@backend-database:5432/mlflow_db"]
