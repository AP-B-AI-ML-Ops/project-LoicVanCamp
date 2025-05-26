"""Trigger and monitor Prefect deployments for batch and monitoring pipelines."""

import re
import subprocess
import time


def wait_for_deployment(deployment_name):
    """Wait for a Prefect deployment to become available."""
    for _ in range(30):  # Try for up to 5 minutes
        result = subprocess.run(
            ["prefect", "deployment", "inspect", deployment_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return True
        time.sleep(5)
    return False


def extract_flow_run_id(output):
    """Extract the flow run ID from Prefect CLI output."""
    match = re.search(r"Flow run ID:\s*([a-f0-9\-]+)", output)
    if match:
        return match.group(1)
    match = re.search(
        r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})", output
    )
    if match:
        return match.group(1)
    return None


def extract_state_type(flow_run_inspect_output):
    """Extract the state type from Prefect flow run inspect output."""
    match = re.search(r"type=StateType\.([A-Z]+)", flow_run_inspect_output)
    if match:
        return match.group(1)
    return None


def trigger_and_wait_for_run(deployment_name):
    """Trigger a Prefect deployment and wait for its completion."""
    if not wait_for_deployment(deployment_name):
        print(f"Deployment {deployment_name} not found.")
        return False

    print(f"Triggering {deployment_name}")
    result = subprocess.run(
        ["prefect", "deployment", "run", deployment_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"Failed to trigger deployment {deployment_name}: {result.stderr}")
        return False

    flow_run_id = extract_flow_run_id(result.stdout)
    if not flow_run_id:
        print(f"Could not parse flow run ID from output:\n{result.stdout}")
        return False
    print(f"Triggered {deployment_name}, flow_run_id: {flow_run_id}")

    # Wait for the flow run to complete
    for _ in range(60):  # Wait up to 10 minutes
        status_result = subprocess.run(
            ["prefect", "flow-run", "inspect", flow_run_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if status_result.returncode == 0 and status_result.stdout.strip():
            state = extract_state_type(status_result.stdout)
            print(f"{deployment_name} run state: {state}")
            if state in ("COMPLETED", "FAILED", "CANCELLED", "CRASHED"):
                return state == "COMPLETED"
        else:
            print(
                f"Flow run not ready or no output yet for {deployment_name}. Retrying..."
            )
        time.sleep(10)
    print(f"Timeout waiting for {deployment_name} run to complete.")
    return False


if __name__ == "__main__":
    print("Starting deployment trigger...")
    # Use the exact deployment names as registered in Prefect
    BATCH_DEPLOYMENT = "run-batch/batch-predictor"
    MONITORING_DEPLOYMENT = "run-monitoring/monitoring-pipeline"

    if trigger_and_wait_for_run(BATCH_DEPLOYMENT):
        trigger_and_wait_for_run(MONITORING_DEPLOYMENT)
    else:
        print(
            "Batch run did not complete successfully. Monitoring will not be triggered."
        )
