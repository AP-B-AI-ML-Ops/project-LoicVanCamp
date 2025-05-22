# pylint: disable=<C0103, C0301, W0621>
"""batch prediction using a pre-trained model."""
import mlflow
from mlflow import MlflowClient
import pandas as pd
from prefect import flow, task
import os
import requests
import pickle

mlflow.set_tracking_uri("http://experiment-tracking:5000")
client = MlflowClient("http://experiment-tracking:5000")
model_name = "rf-math-pass-predictor"


def read_dataframe(filename: str):
    """Read a CSV file into a pandas DataFrame."""
    df = pd.read_csv(filename)
    # Create pass_math column if not present
    if "pass_math" not in df.columns and "math score" in df.columns:
        df["pass_math"] = (df["math score"] >= 50).astype(int)
    return df


def load_dv(dv_path):
    """Load a dictvectorizer from a pickle file."""
    with open(dv_path, "rb") as f_in:
        dv = pickle.load(f_in)
    return dv


def prep_features(df, dv):
    categorical = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course"
    ]
    dicts = df[categorical].to_dict(orient="records")
    X = dv.transform(dicts)
    return X


# Get latest version of the registered model on the client
def get_latest_version(model_name):
    versions = client.get_latest_versions(model_name)
    # choose highest version (meestal de laatste)
    if versions:
        return versions[-1].version
    raise RuntimeError("No model versions found.")


def load_model():
    print("...loading model")
    latest_version = get_latest_version(model_name)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version}")

    return model


@task
def load_model_task():
    return load_model()


@task
def load_dv_task(dv_path):
    return load_dv(dv_path)


@task
def read_dataframe_task(filepath):
    return read_dataframe(filepath)


@task
def prep_features_task(df, dv):
    return prep_features(df, dv)


@task
def save_result_task(df_result, run_id):
    path = os.path.join("batch-data", "report", "students")
    os.makedirs(path, exist_ok=True)
    df_result.to_csv(os.path.join(path, f"{run_id}.csv"), index=False)
    print(f"Saved to {os.path.join(path, f'{run_id}.csv')}")


@flow
def run_batch(filepath, dv_path):
    model = load_model_task()
    dv = load_dv_task(dv_path)
    df = read_dataframe_task(filepath)
    X = prep_features_task(df, dv)
    y_pred = model.predict(X)
    df_result = df.copy()
    run_id = getattr(getattr(model, "metadata", None), "run_id", "unknown")
    df_result["pass_math_pred"] = y_pred
    if "pass_math" in df_result.columns:
        df_result["pass_math_delta"] = df_result["pass_math"] - df_result["pass_math_pred"]
    df_result["model_id"] = run_id
    save_result_task(df_result, run_id)


if __name__ == "__main__":
    run_batch.serve(
        name="batch-flow",
        parameters={
            "filepath": "data/StudentsPerformance.csv",
            "dv_path": "models/dv.pkl"
        },
    )
