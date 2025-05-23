# pylint: disable=<C0103>
"""Register the best Random Forest model from Hyperopt runs."""
import pickle
from pathlib import Path

import click
import joblib
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Constants
HPO_EXPERIMENT_NAME = "student-performance-hpo"
EXPERIMENT_NAME = "best-model"
RF_PARAMS = [
    "max_depth",
    "n_estimators",
    "min_samples_split",
    "min_samples_leaf",
    "random_state",
    "n_jobs"
]

mlflow.set_tracking_uri("http://experiment-tracking:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(path: Path):
    with path.open("rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path: Path, params: dict):
    X_train, y_train = load_pickle(data_path / "train.pkl")
    X_val, y_val = load_pickle(data_path / "val.pkl")
    X_test, y_test = load_pickle(data_path / "test.pkl")

    with mlflow.start_run() as run:
        # Cast parameters to int where needed
        for param in RF_PARAMS:
            if param in params:
                try:
                    params[param] = int(params[param])
                except Exception:
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
        # Save the model in ./models/model.pkl
        joblib.dump(clf, "models/model.pkl")

        return run.info.run_id


@click.command()
@click.option(
    "--data_path",
    default="models",
    help="Location where the processed students performance data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):
    data_path = Path(data_path)
    client = MlflowClient()

    # Zoek top N HPO-runs (op basis van accuracy)
    hpo_experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    hpo_runs = client.search_runs(
        experiment_ids=hpo_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.accuracy DESC"]
    )

    print(f"🔍 HPO-runs found: {len(hpo_runs)}")

    # Hertrain en log met alle top runs
    for run in hpo_runs:
        train_and_log_model(data_path, run.data.params)

    # Zoek beste model op test_accuracy
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_accuracy DESC"]
    )[0]

    best_run_id = best_run.info.run_id
    model_uri = f"runs:/{best_run_id}/model"

    # Registreer het model
    mlflow.register_model(
        model_uri=model_uri,
        name="rf-math-pass-predictor"
    )

    print(f"✅ Best model registered with run ID: {best_run_id}")


if __name__ == "__main__":
    run_register_model()
