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
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .utils import load_pickle

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://experiment-tracking:5000")
)
mlflow.set_experiment("student-performance-hpo")


@click.command()
@click.option(
    "--data_dir",
    default="models",
    help="Directory where train.pkl and val.pkl are stored.",
)
@click.option("--num_trials", default=10, help="Number of experiments to try")
def run_optimization(data_dir: str, num_trials: int):
    """Run hyperparameter optimization for RandomForestClassifier.

    Args:
        data_dir (str): Directory containing train.pkl and val.pkl.
        num_trials (int): Number of Optuna trials to run.
    """
    data_path = Path(data_dir)
    X_train, y_train = load_pickle(data_path / "train.pkl")
    X_val, y_val = load_pickle(data_path / "val.pkl")

    def objective(trial):
        """Objective function for Optuna to optimize RandomForest hyperparameters.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.

        Returns:
            float: Negative accuracy score (since Optuna minimizes).
        """
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
