FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY trigger_deployments.py /app/trigger_deployments.py

CMD ["python", "/app/trigger_deployments.py"]
