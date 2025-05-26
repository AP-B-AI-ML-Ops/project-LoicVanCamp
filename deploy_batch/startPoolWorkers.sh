#!/bin/bash
# filepath: c:\Users\Loïc Van Camp\Documents\MLOps\project-LoicVanCamp\deploy_batch\startPoolWorkers.sh

# Wait for Prefect API to be ready
until curl -sf http://orchestration:4200/docs; do
  echo "Waiting for Prefect Orchestration API..."
  sleep 5
done

# Start Prefect work pool and worker
prefect work-pool create --type process batch --overwrite
prefect worker start --pool batch &

python /app/batch.py
