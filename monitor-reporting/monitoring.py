import os
from datetime import datetime, timezone
from dotenv import load_dotenv
import shutil
from glob import glob
import json

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sqlalchemy import create_engine, types
from sqlalchemy_utils import database_exists, create_database

from evidently import Report
from evidently.presets import DataDriftPreset
from prefect import flow, task


@task
def load_env():
    load_dotenv()


@task
def load_model(model_name, model_version):
    mlflow_uri = "http://experiment-tracking:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient(mlflow_uri)
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}")
    return model


@task
def load_data():
    reference_data = pd.read_csv("batch-data/report/students/reference.csv")
    current_data = pd.read_csv("batch-data/report/students/current.csv")
    return reference_data, current_data


@task
def load_run_metrics():
    # Zoek het nieuwste metrics-bestand
    candidates = glob("batch-data/report/students/*_metrics.json")
    if not candidates:
        return {}
    latest = max(candidates, key=os.path.getctime)
    with open(latest, "r") as f:
        return json.load(f)


@task
def generate_report(reference_data, current_data):
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_data,
                          current_data=current_data)
    return snapshot


@task
def extract_metrics(snapshot, run_metrics):
    json_data = snapshot.dict()
    result_data = []
    report_time = datetime.now(timezone.utc)
    for metric in json_data['metrics']:
        metric_id = metric['metric_id']
        value = metric['value']
        # Zorg dat value altijd een dict is
        if isinstance(value, dict):
            value_with_meta = dict(value)
        else:
            value_with_meta = {"value": value}
        value_with_meta.update({
            "run_id": run_metrics.get("run_id"),
            "run_name": run_metrics.get("run_name"),
            "rmse": run_metrics.get("rmse")
        })
        result_data.append({
            "run_time": report_time,
            "metric_name": metric_id,
            "value": value_with_meta
        })
    metrics_df = pd.DataFrame(result_data)
    metrics_df.to_csv("batch-data/report/evidently_metrics.csv", index=False)
    print("✅ Evidently metrics saved to batch-data/report/evidently_metrics.csv")
    return metrics_df


@task
def save_to_db(metrics_df):
    DB_USER = os.getenv("POSTGRES_USER")
    DB_PWD = os.getenv("POSTGRES_PASSWORD")
    DB_NAME = "mlflow_db"
    DB_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PWD}@backend-database/{DB_NAME}"

    if not database_exists(DB_URI):
        create_database(DB_URI)

    engine = create_engine(DB_URI)
    metrics_df.to_sql("evidently_metrics", engine, if_exists="append",
                      index=False, dtype={"value": types.JSON})
    print("✅ Evidently metrics saved to database")


@task
def ensure_reference_csv():
    ref_path = "batch-data/report/students/reference.csv"
    if not os.path.exists(ref_path):
        # Zoek het nieuwste batch-resultaat
        candidates = glob("batch-data/report/students/*.csv")
        candidates = [f for f in candidates if not f.endswith("reference.csv")]
        if not candidates:
            raise FileNotFoundError(
                "Geen batch-resultaten gevonden om reference.csv aan te maken!")
        latest = max(candidates, key=os.path.getctime)
        shutil.copy(latest, ref_path)
        print(f"✅ reference.csv aangemaakt op basis van: {latest}")
    else:
        print("ℹ️ reference.csv bestaat al.")


@task
def ensure_current_csv():
    curr_path = "batch-data/report/students/current.csv"
    if not os.path.exists(curr_path):
        # Zoek het nieuwste batch-resultaat
        candidates = glob("batch-data/report/students/*.csv")
        candidates = [f for f in candidates if not f.endswith(
            "reference.csv") and not f.endswith("current.csv")]
        if not candidates:
            raise FileNotFoundError(
                "Geen batch-resultaten gevonden om current.csv aan te maken!")
        latest = max(candidates, key=os.path.getctime)
        shutil.copy(latest, curr_path)
        print(f"✅ current.csv aangemaakt op basis van: {latest}")
    else:
        print("ℹ️ current.csv bestaat al.")


@flow
def run_monitoring(model_name: str = "rf-math-pass-predictor", model_version: int = 1):
    load_env()
    load_model(model_name, model_version)
    ensure_reference_csv()
    ensure_current_csv()
    reference_data, current_data = load_data()
    run_metrics = load_run_metrics()
    snapshot = generate_report(reference_data, current_data)
    metrics_df = extract_metrics(snapshot, run_metrics)
    save_to_db(metrics_df)


if __name__ == "__main__":
    run_monitoring.serve(
        name="monitoring-flow",
        parameters={
            "model_name": "rf-math-pass-predictor",
            "model_version": 1
        }
    )
