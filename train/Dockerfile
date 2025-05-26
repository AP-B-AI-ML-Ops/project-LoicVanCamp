FROM python:3.13-slim

WORKDIR /app

RUN pip install -U pip

COPY . /app

RUN pip install -r requirements.txt --upgrade

CMD ["python", "scripts/main.py"]
