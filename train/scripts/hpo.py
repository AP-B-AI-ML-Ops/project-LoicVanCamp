"""Hyperparameter optimization for RandomForest using Optuna and MLflow.

This script runs hyperparameter optimization for a RandomForestClassifier using Optuna,
logs results to MLflow, and supports configuration via command-line arguments.
"""

# pylint: disable=import-error,invalid-name,line-too-long,no-value-for-parameter

import os
from pathlib import Path

import click
import mlflow
import optuna
from mlflow.tracking import MlflowClient
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import load_pickle

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://experiment-tracking:5000")
)
mlflow.set_experiment("student-performance-hpo")


@click.command()
@click.option("--num_trials", default=10, help="Number of experiments to try")
def run_optimization(num_trials: int):
    """Run hyperparameter optimization for RandomForestClassifier.

    Args:
        num_trials (int): Number of Optuna trials to run.
    """
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

    X_train, y_train = load_pickle(Path(train_path))
    X_val, y_val = load_pickle(Path(val_path))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "random_state": 42,
            "n_jobs": -1,
        }

        with mlflow.start_run():
            mlflow.log_params(params)

            rf = RandomForestClassifier(**params)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            mlflow.log_metric("accuracy", acc)

            return -acc  # Optuna minimizes

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=num_trials)

    print("✅ Best score:", -study.best_value)
    print("✅ Best params:", study.best_params)


if __name__ == "__main__":
    run_optimization()  # Click will parse command-line arguments
