# pylint: disable=<C0103>
"""Register the best Random Forest model from Hyperopt runs.

This script loads the best hyperparameter optimization (HPO) runs from MLflow,
re-trains and logs models, and registers the best model in the MLflow Model Registry.
"""

# pylint: disable=no-value-for-parameter,import-error,invalid-name

import os
import pickle
from pathlib import Path

import click
import mlflow
from dotenv import load_dotenv
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

load_dotenv()

# Constants
HPO_EXPERIMENT_NAME = "student-performance-hpo"
EXPERIMENT_NAME = "best-model"
RF_PARAMS = [
    "max_depth",
    "n_estimators",
    "min_samples_split",
    "min_samples_leaf",
    "random_state",
    "n_jobs",
]

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://experiment-tracking:5000")
)
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(path: Path):
    """Load a Python object from a pickle file.

    Args:
        path (Path): Path to the pickle file.

    Returns:
        Any: The loaded Python object.
    """
    with path.open("rb") as f_in:
        return pickle.load(f_in)


def get_latest_preprocess_run_id():
    """Get the latest run_id from the 'preprocess' experiment."""
    client = MlflowClient()
    preprocess_experiment = client.get_experiment_by_name("preprocess")
    if preprocess_experiment is None:
        raise RuntimeError("No 'preprocess' experiment found in MLflow.")
    runs = client.search_runs(
        experiment_ids=preprocess_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["start_time DESC"],
    )
    if not runs:
        raise RuntimeError("No runs found in 'preprocess' experiment.")
    return runs[0].info.run_id


def train_and_log_model(params: dict):
    """Train a RandomForestClassifier with given parameters, log to MLflow, and save the model.

    Args:
        params (dict): Hyperparameters for the RandomForestClassifier.

    Returns:
        str: The MLflow run ID.
    """

    # Get latest preprocess run_id
    PREPROCESS_RUN_ID = get_latest_preprocess_run_id()

    # Download data from MLflow artifacts
    train_path = mlflow.artifacts.download_artifacts(
        run_id=PREPROCESS_RUN_ID, artifact_path="data/train.pkl"
    )
    val_path = mlflow.artifacts.download_artifacts(
        run_id=PREPROCESS_RUN_ID, artifact_path="data/val.pkl"
    )
    test_path = mlflow.artifacts.download_artifacts(
        run_id=PREPROCESS_RUN_ID, artifact_path="data/test.pkl"
    )
    X_train, y_train = load_pickle(Path(train_path))
    X_val, y_val = load_pickle(Path(val_path))
    X_test, y_test = load_pickle(Path(test_path))
    # Ensure experiment is set before logging
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        # Cast parameters to int where needed
        for param in RF_PARAMS:
            if param in params:
                try:
                    params[param] = int(params[param])
                except (ValueError, TypeError):
                    pass

        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)

        # Validate
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        mlflow.log_metric("val_accuracy", val_acc)

        test_acc = accuracy_score(y_test, clf.predict(X_test))
        mlflow.log_metric("test_accuracy", test_acc)

        # Log model
        mlflow.sklearn.log_model(clf, artifact_path="model")
        mlflow.log_artifact("dv.pkl", artifact_path="model")

        return run.info.run_id


@click.command()
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote",
)
def run_register_model(top_n: int):
    """Find, retrain, and register the best model from HPO runs.

    Args:
        top_n (int): Number of top HPO runs to consider.
    """
    client = MlflowClient()

    # Search top N HPO-runs (by accuracy)
    hpo_experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    hpo_runs = client.search_runs(
        experiment_ids=hpo_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.accuracy DESC"],
    )

    print(f"🔍 HPO-runs found: {len(hpo_runs)}")

    # Retrain and log with all top runs
    for run in hpo_runs:
        train_and_log_model(run.data.params)

    # Search best model by test_accuracy
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_accuracy DESC"],
    )[0]

    best_run_id = best_run.info.run_id
    model_uri = f"runs:/{best_run_id}/model"

    # Register the model
    mlflow.register_model(model_uri=model_uri, name="rf-math-pass-predictor")

    print(f"✅ Best model registered with run ID: {best_run_id}")


if __name__ == "__main__":
    run_register_model()
