"""Batch prediction using a pre-trained model.

This module loads a trained model and DictVectorizer, processes batch data,
makes predictions, saves results and metrics, and supports Prefect orchestration.
"""

# pylint: disable=invalid-name

import json
import os
import pickle
import time

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
    """Get the latest version number of a registered model.

    Args:
        registered_model_name (str): Name of the registered model.

    Returns:
        int: Latest version number.
    """
    versions = client.get_latest_versions(registered_model_name)
    if versions:
        return versions[-1].version
    raise RuntimeError("No model versions found.")


def load_model():
    """Load the latest version of the registered model from MLflow.

    Returns:
        mlflow.pyfunc.PyFuncModel: Loaded model.
    """
    print("...loading model")
    latest_version = get_latest_version(model_name)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version}")
    return model


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
        print(f"⏳ Wachten op modelbestand: {model_path} ({waited}/{timeout} sec)")
        time.sleep(interval)
        waited += interval
    print(f"✅ Modelbestand gevonden: {model_path}")


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
def load_model_task():
    """Prefect task to load the latest model from MLflow.

    Returns:
        mlflow.pyfunc.PyFuncModel: Loaded model.
    """
    return load_model()


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
def save_result_task(df_result, run_id):
    """Save prediction results to a CSV file.

    Args:
        df_result (pd.DataFrame): DataFrame with prediction results.
        run_id (str): Run ID for naming the output file.
    """
    path = os.path.join("batch_data", "report", "students")
    os.makedirs(path, exist_ok=True)
    df_result.to_csv(os.path.join(path, f"{run_id}.csv"), index=False)
    print(f"Saved to {os.path.join(path, f'{run_id}.csv')}")


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


@flow
def run_batch(filepath, dv_path, model_path="models/model.pkl"):
    """Main Prefect flow for running batch prediction.

    Args:
        filepath (str): Path to the input CSV file.
        dv_path (str): Path to the DictVectorizer pickle file.
        model_path (str, optional): Path to the model file. Defaults to "models/model.pkl".
    """
    wait_for_model(model_path)
    model = load_model_task()
    dv = load_dv_task(dv_path)
    df = read_dataframe_task(filepath)
    X = prep_features_task(df, dv)
    y_pred = model.predict(X)
    df_result = df.copy()
    run_id = getattr(getattr(model, "metadata", None), "run_id", "unknown")
    run_name = getattr(getattr(model, "metadata", None), "run_name", "unknown")
    df_result["pass_math_pred"] = y_pred
    if "pass_math" in df_result.columns:
        df_result["pass_math_delta"] = (
            df_result["pass_math"] - df_result["pass_math_pred"]
        )
        # Bereken RMSE
        rmse = np.sqrt(
            mean_squared_error(df_result["pass_math"], df_result["pass_math_pred"])
        )
    else:
        rmse = None
    df_result["model_id"] = run_id
    save_result_task(df_result, run_id)
    save_metrics_task(run_id, run_name, rmse)


if __name__ == "__main__":
    run_batch.serve(
        name="batch-flow",
        parameters={
            "filepath": "data/StudentsPerformance.csv",
            "dv_path": "models/dv.pkl",
            "model_path": "models/model.pkl",
        },
    )
