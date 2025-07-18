#!/bin/bash

# Wait for Prefect API to be ready
until curl -sf http://orchestration:4200/docs; do
  echo "Waiting for Prefect Orchestration API..."
  sleep 5
done

# Start Prefect work pool and worker
prefect work-pool create --type process batching --overwrite
prefect worker start --pool batching &

python /app/batch.py
