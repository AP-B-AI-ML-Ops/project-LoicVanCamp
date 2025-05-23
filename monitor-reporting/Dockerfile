FROM python:3.13-alpine

# Update system packages to address vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean

RUN pip install -U pip

COPY requirements.txt ./
RUN pip install -r requirements.txt --upgrade

COPY . /

RUN prefect config set PREFECT_API_URL=http://orchestration:4200/api

RUN chmod +x ./startPoolWorkers.sh

CMD [ "bash", "/startPoolWorkers.sh" ]
