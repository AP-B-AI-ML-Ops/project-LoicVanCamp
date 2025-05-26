FROM python:3.13-slim

WORKDIR /app

RUN pip install --upgrade pip prefect

COPY trigger_deployments.py /app/trigger_deployments.py

CMD ["python", "/app/trigger_deployments.py"]
