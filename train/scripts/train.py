"""Train a Logistic Regression model on student performance data and log results to MLflow.

This script loads preprocessed training and validation data, trains a Logistic Regression model,
evaluates it, logs metrics and the model to MLflow, and supports experiment tracking.
"""

# pylint: disable=invalid-name,line-too-long

import os
import pickle
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set up MLflow tracking URI and experiment
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://experiment-tracking:5000")
)
mlflow.set_experiment("student-performance-train")


def load_pickle(filename):
    """Load a Python object from a pickle file.

    Args:
        filename (str): Path to the pickle file.

    Returns:
        Any: The loaded Python object.
    """
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run_train():
    """Train a Logistic Regression model and log results to MLflow using data from MLflow artifacts."""
    mlflow.sklearn.autolog()

    # Get latest preprocess run
    client = MlflowClient()
    preprocess_exp = client.get_experiment_by_name("preprocess")
    runs = client.search_runs(
        experiment_ids=preprocess_exp.experiment_id,
        run_view_type=1,
        max_results=1,
        order_by=["start_time DESC"],
    )
    if not runs:
        raise RuntimeError("No runs found in 'preprocess' experiment.")
    preprocess_run_id = runs[0].info.run_id

    # Download data from MLflow artifacts
    train_path = mlflow.artifacts.download_artifacts(
        run_id=preprocess_run_id, artifact_path="data/train.pkl"
    )
    val_path = mlflow.artifacts.download_artifacts(
        run_id=preprocess_run_id, artifact_path="data/val.pkl"
    )
    dv_path = mlflow.artifacts.download_artifacts(
        run_id=preprocess_run_id, artifact_path="data/dv.pkl"
    )

    X_train, y_train = load_pickle(Path(train_path))
    X_val, y_val = load_pickle(Path(val_path))
    dv = load_pickle(Path(dv_path))

    with mlflow.start_run():
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        mlflow.log_metric("accuracy", acc)
        print("Validation accuracy:", acc)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Save DictVectorizer to disk and log as artifact
        with open("dv.pkl", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("dv.pkl", artifact_path="model")


if __name__ == "__main__":
    print("...training model")
    run_train()
