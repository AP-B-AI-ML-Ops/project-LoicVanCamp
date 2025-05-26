#!/bin/bash

# Wait for Prefect API to be ready
until curl -sf http://orchestration:4200/docs; do
  echo "Waiting for Prefect Orchestration API..."
  sleep 5
done

# Start Prefect worker and flow
prefect work-pool create --type process training --overwrite
prefect worker start --pool training &

python /app/scripts/train_flow.py
