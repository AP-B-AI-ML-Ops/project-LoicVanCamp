#!/bin/bash

prefect work-pool create --type process monitoring --overwrite
prefect worker start -p monitoring &

python /monitoring.py

prefect deployment run 'run-monitoring/monitoring-flow'
