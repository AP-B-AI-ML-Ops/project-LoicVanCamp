import pickle
from pathlib import Path

import click
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


def dump_pickle(obj, path: Path):
    with path.open("wb") as f_out:
        pickle.dump(obj, f_out)


def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    categorical = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course"
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
    default="../../train/data/StudentsPerformance.csv",
    help="Location where the students performance data was saved"
)
@click.option(
    "--dest_path",
    default="../../models",
    help="Location where the resulting files will be saved"
)
def run_data_prep(raw_data_path: str, dest_path: str):
    # Load dataset
    raw_data_path = Path(raw_data_path).resolve()
    dest_path = Path(dest_path).resolve()
    df = pd.read_csv(raw_data_path)

    # Extract the Target : passed for math?
    df["pass_math"] = (df["math score"] >= 50).astype(int)
    y = df["pass_math"].values

    # Split into train/val/test
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=42)

    y_train = df_train["pass_math"].values
    y_val = df_val["pass_math"].values
    y_test = df_test["pass_math"].values

    # Fit the DictVectorizer and preprocess data
    dv = DictVectorizer()

    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # Create dest_path folder unless it already exists
    dest_path.mkdir(parents=True, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle(dv, dest_path / "dv.pkl")
    dump_pickle((X_train, y_train), dest_path / "train.pkl")
    dump_pickle((X_val, y_val), dest_path / "val.pkl")
    dump_pickle((X_test, y_test), dest_path / "test.pkl")

    print("✅ Preprocessing completed. Data saved in:", dest_path)


if __name__ == "__main__":
    run_data_prep()
