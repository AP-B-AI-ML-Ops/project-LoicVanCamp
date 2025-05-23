"""Train a Logistic Regression model on student performance data and log results to MLflow.

This script loads preprocessed training and validation data, trains a Logistic Regression model,
evaluates it, logs metrics and the model to MLflow, and supports experiment tracking.
"""

# pylint: disable=invalid-name

import os
import pickle

import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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


def run_train(data_path: str):
    """Train a Logistic Regression model and log results to MLflow.

    Args:
        data_path (str): Directory containing train.pkl and val.pkl.
    """
    mlflow.sklearn.autolog()
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        mlflow.log_metric("accuracy", acc)
        print("Validation accuracy:", acc)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    print("...training model")
    run_train("models")
