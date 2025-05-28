"""Preprocess student performance data for ML training and evaluation.

This script loads raw student performance data, creates a target column,
splits the data into train/validation/test sets, vectorizes categorical features,
and saves the processed datasets and vectorizer as pickle files.
"""

# pylint: disable=no-value-for-parameter,invalid-name

import pickle
import tempfile
from pathlib import Path

import click
import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

load_dotenv()


def dump_pickle(obj, path: Path):
    """Save a Python object to a file using pickle.

    Args:
        obj: The Python object to serialize.
        path (Path): The file path where the object will be saved.
    """
    with path.open("wb") as f_out:
        pickle.dump(obj, f_out)


def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    """Transform categorical features using a DictVectorizer.

    Args:
        df (pd.DataFrame): DataFrame containing the data to transform.
        dv (DictVectorizer): The DictVectorizer instance.
        fit_dv (bool, optional): Whether to fit the vectorizer. Defaults to False.

    Returns:
        tuple: (X, dv) where X is the transformed feature matrix and dv is the vectorizer.
    """
    categorical = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
    ]
    dicts = df[categorical].to_dict(orient="records")
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


@click.command()
@click.option(
    "--raw_data_path",
    default="data/StudentsPerformance.csv",
    help="Location where the students performance data was saved",
)
def run_data_prep(raw_data_path: str):
    """Main function to preprocess data and log train/val/test splits and vectorizer to MLflow.

    Args:
        raw_data_path (str): Path to the raw CSV data file.
    """
    # Load dataset
    raw_data_path = Path(raw_data_path)
    df = pd.read_csv(raw_data_path)

    # Create target column
    df["pass_math"] = (df["math score"] >= 50).astype(int)

    # Split in train/val/test
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

    y_train = df_train["pass_math"].values
    y_val = df_val["pass_math"].values
    y_test = df_test["pass_math"].values

    # Vectorizer and features
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # Save and log to MLflow
    mlflow.set_experiment("preprocess")
    with mlflow.start_run(run_name="preprocess") as run:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            train_pkl = tmpdir / "train.pkl"
            val_pkl = tmpdir / "val.pkl"
            test_pkl = tmpdir / "test.pkl"
            dv_pkl = tmpdir / "dv.pkl"
            with train_pkl.open("wb") as f:
                pickle.dump((X_train, y_train), f)
            with val_pkl.open("wb") as f:
                pickle.dump((X_val, y_val), f)
            with test_pkl.open("wb") as f:
                pickle.dump((X_test, y_test), f)
            with dv_pkl.open("wb") as f:
                pickle.dump(dv, f)
            mlflow.log_artifact(str(train_pkl), artifact_path="data")
            mlflow.log_artifact(str(val_pkl), artifact_path="data")
            mlflow.log_artifact(str(test_pkl), artifact_path="data")
            mlflow.log_artifact(str(dv_pkl), artifact_path="data")
    print(f"✅ Preprocessing completed. Data logged to MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    run_data_prep()
