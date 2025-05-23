"""Preprocess student performance data for ML training and evaluation.

This script loads raw student performance data, creates a target column,
splits the data into train/validation/test sets, vectorizes categorical features,
and saves the processed datasets and vectorizer as pickle files.
"""

# pylint: disable=no-value-for-parameter,invalid-name

import pickle
from pathlib import Path

import click
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


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
@click.option(
    "--dest_path",
    default="models",
    help="Location where the resulting files will be saved",
)
def run_data_prep(raw_data_path: str, dest_path: str):
    """Main function to preprocess data and save train/val/test splits and vectorizer.

    Args:
        raw_data_path (str): Path to the raw CSV data file.
        dest_path (str): Directory where processed files will be saved.
    """
    # Load dataset
    raw_data_path = Path(raw_data_path)
    dest_path = Path(dest_path)
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

    # Save
    dest_path.mkdir(parents=True, exist_ok=True)
    dump_pickle(dv, dest_path / "dv.pkl")
    dump_pickle((X_train, y_train), dest_path / "train.pkl")
    dump_pickle((X_val, y_val), dest_path / "val.pkl")
    dump_pickle((X_test, y_test), dest_path / "test.pkl")

    print(f"✅ Preprocessing completed. Data saved in: {dest_path}")


if __name__ == "__main__":
    run_data_prep()
