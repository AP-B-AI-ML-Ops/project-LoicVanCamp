"""Prefect training pipeline for student math pass prediction.

This module defines a Prefect flow and tasks for preprocessing data,
training a model, running hyperparameter optimization, and registering
the best model in the MLflow Model Registry.
"""

import subprocess

from prefect import flow, task


@task
def preprocess_data():
    """
    Run the data preprocessing script to prepare training data.

    This task executes 'preprocess_data.py' with the raw data path as input.
    """
    subprocess.run(
        [
            "python",
            "scripts/preprocess_data.py",
            "--raw_data_path",
            "data/StudentsPerformance.csv",
        ],
        check=True,
    )


@task
def train_model():
    """
    Train the model using the preprocessed data.

    This task executes 'train.py' to train and log the model to MLflow.
    """
    subprocess.run(
        ["python", "scripts/train.py"],
        check=True,
    )


@task
def run_hpo():
    """
    Run hyperparameter optimization for the model.

    This task executes 'hpo.py' to perform HPO and log results to MLflow.
    """
    subprocess.run(
        [
            "python",
            "scripts/hpo.py",
            "--num_trials",
            "20",
        ],
        check=True,
    )


@task
def register_best_model():
    """
    Register the best model from HPO runs in the MLflow Model Registry.

    This task executes 'register_model.py' to retrain and register the top N models.
    """
    subprocess.run(
        [
            "python",
            "scripts/register_model.py",
            "--top_n",
            "5",
        ],
        check=True,
    )


@flow(name="run-train")
def train_pipeline():
    """
    Prefect flow to run the full training pipeline:
    - Preprocess data
    - Train model
    - Run hyperparameter optimization
    - Register the best model
    """
    preprocess_data()
    train_model()
    run_hpo()
    register_best_model()


if __name__ == "__main__":
    train_pipeline.serve(
        name="train-pipeline",
        parameters={},
    )
