FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl

RUN pip install -U pip setuptools wheel

WORKDIR /prefect

RUN pwd

COPY ["./.prefectignore", "./prefect.yaml", "./"]

COPY requirements.txt .
RUN pip install -r ./requirements.txt

CMD ["prefect", "server", "start", "--host", "0.0.0.0", "--port", "4200"]
