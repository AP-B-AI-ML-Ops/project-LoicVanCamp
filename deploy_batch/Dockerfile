FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl

RUN pip install --upgrade pip

COPY . /app

RUN pip install --upgrade -r requirements.txt

# Add startup script
COPY startPoolWorkers.sh /app/startPoolWorkers.sh
RUN chmod +x /app/startPoolWorkers.sh

CMD ["/app/startPoolWorkers.sh"]
