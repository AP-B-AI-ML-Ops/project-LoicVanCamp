"""Batch prediction using a pre-trained model.

This module loads a trained model and DictVectorizer, processes batch data,
makes predictions, saves results and metrics, and supports Prefect orchestration.
"""

# pylint: disable=invalid-name

import json
import os
import pickle
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from prefect import flow, task
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://experiment-tracking:5000")
client = MlflowClient("http://experiment-tracking:5000")
model_name = "rf-math-pass-predictor"


def read_dataframe(filename: str):
    """Read a CSV file into a pandas DataFrame and add pass_math column if needed.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame with pass_math column.
    """
    df = pd.read_csv(filename)
    # Create pass_math column if not present
    if "pass_math" not in df.columns and "math score" in df.columns:
        df["pass_math"] = (df["math score"] >= 50).astype(int)
    return df


def load_dv(dv_path):
    """Load a DictVectorizer from a pickle file.

    Args:
        dv_path (str): Path to the DictVectorizer pickle file.

    Returns:
        DictVectorizer: Loaded DictVectorizer object.
    """
    with open(dv_path, "rb") as f_in:
        dv = pickle.load(f_in)
    return dv


def prep_features(df, dv):
    """Transform categorical features using the DictVectorizer.

    Args:
        df (pd.DataFrame): DataFrame with input data.
        dv (DictVectorizer): Fitted DictVectorizer.

    Returns:
        np.ndarray: Transformed feature matrix.
    """
    categorical = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
    ]
    dicts = df[categorical].to_dict(orient="records")
    X = dv.transform(dicts)
    return X


def get_latest_version(registered_model_name):
    """Get the latest version object of a registered model.

    Args:
        registered_model_name (str): Name of the registered model.

    Returns:
        ModelVersion: Latest version object.
    """
    versions = client.get_latest_versions(registered_model_name)
    if versions:
        return versions[-1]
    raise RuntimeError("No model versions found.")


def load_model_and_dv():
    """Load the latest model and DictVectorizer from MLflow."""
    print("...loading model and DictVectorizer from MLflow")
    latest_version = get_latest_version(model_name)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version.version}")
    # Download dv.pkl from the run's artifacts
    run_id = latest_version.run_id
    dv_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="model/dv.pkl"
    )
    dv = load_dv(Path(dv_path))  # Use Path() here
    return model, dv, run_id, latest_version.name


def wait_for_model(model_path: str, timeout: int = 300, interval: int = 5):
    """Wait for the model file to exist, with a timeout.

    Args:
        model_path (str): Path to the model file.
        timeout (int, optional): Maximum wait time in seconds. Defaults to 300.
        interval (int, optional): Interval between checks in seconds. Defaults to 5.

    Raises:
        TimeoutError: If the model file is not found within the timeout.
    """
    waited = 0
    while not os.path.exists(model_path):
        if waited >= timeout:
            raise TimeoutError(
                f"Model file {model_path} not found after {timeout} seconds."
            )
        print(f"⏳ Waiting for model file: {model_path} ({waited}/{timeout} sec)")
        time.sleep(interval)
        waited += interval
    print(f"✅ Model file found: {model_path}")


@task
def wait_for_model_task(model_path, timeout=300, interval=5):
    """Prefect task to wait for the model file to exist.

    Args:
        model_path (str): Path to the model file.
        timeout (int, optional): Maximum wait time in seconds. Defaults to 300.
        interval (int, optional): Interval between checks in seconds. Defaults to 5.
    """
    wait_for_model(model_path, timeout, interval)


@task
def load_dv_task(dv_path):
    """Prefect task to load the DictVectorizer.

    Args:
        dv_path (str): Path to the DictVectorizer pickle file.

    Returns:
        DictVectorizer: Loaded DictVectorizer object.
    """
    return load_dv(dv_path)


@task
def read_dataframe_task(filepath):
    """Prefect task to read a CSV file into a DataFrame.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return read_dataframe(filepath)


@task
def prep_features_task(df, dv):
    """Prefect task to transform features using the DictVectorizer.

    Args:
        df (pd.DataFrame): DataFrame with input data.
        dv (DictVectorizer): Fitted DictVectorizer.

    Returns:
        np.ndarray: Transformed feature matrix.
    """
    return prep_features(df, dv)


@task
def save_result_task(df_result, run_id, update_reference=False):
    """Save prediction results to a CSV file and current.csv. Optionally update reference.csv."""
    path = os.path.join("batch_data", "report", "students")
    os.makedirs(path, exist_ok=True)
    result_path = os.path.join(path, f"{run_id}.csv")
    reference_path = os.path.join(path, "reference.csv")
    current_path = os.path.join(path, "current.csv")
    df_result.to_csv(result_path, index=False)
    df_result.to_csv(current_path, index=False)
    # Only update reference.csv if requested or if it doesn't exist
    if update_reference or not os.path.exists(reference_path):
        df_result.to_csv(reference_path, index=False)
        print(f"Saved to {result_path}, {current_path}, and updated {reference_path}")
    else:
        print(f"Saved to {result_path}, {current_path} (reference.csv unchanged)")


@task
def save_metrics_task(run_id, run_name, rmse):
    """Save run metrics as JSON for monitoring.

    Args:
        run_id (str): Run ID.
        run_name (str): Run name.
        rmse (float): Root mean squared error.
    """
    metrics = {"run_id": run_id, "run_name": run_name, "rmse": rmse}
    path = os.path.join("batch_data", "report", "students")
    os.makedirs(path, exist_ok=True)
    metrics_path = os.path.join(path, f"{run_id}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    print(f"Saved metrics to {metrics_path}")


@task
def load_model_and_dv_task():
    """Prefect task to load the latest model and DictVectorizer from MLflow.

    Returns:
        tuple: (model, dv, run_id, model_name)
    """
    return load_model_and_dv()


@flow(name="run-batch")
def run_batch(
    filepath: str,
):
    """Run batch prediction on the provided CSV file.

    Args:
        filepath (str): Path to the input CSV file for batch prediction.

    This flow loads the latest model and DictVectorizer from MLflow,
    reads the input data, performs predictions, saves results and metrics.
    """
    model, dv, run_id, run_name = load_model_and_dv_task()
    df = read_dataframe_task(filepath)
    X = prep_features_task(df, dv)
    y_pred = model.predict(X)
    df_result = df.copy()
    df_result["pass_math_pred"] = y_pred
    if "pass_math" in df_result.columns:
        df_result["pass_math_delta"] = (
            df_result["pass_math"] - df_result["pass_math_pred"]
        )
        rmse = np.sqrt(
            mean_squared_error(df_result["pass_math"], df_result["pass_math_pred"])
        )
    else:
        rmse = None
    df_result["model_id"] = run_id
    save_result_task(df_result, run_id, update_reference=False)
    save_metrics_task(run_id, run_name, rmse)


if __name__ == "__main__":
    run_batch.serve(
        name="batch-pipeline",
        parameters={
            "filepath": "/app/batch_data/Students.csv",
        },
    )
