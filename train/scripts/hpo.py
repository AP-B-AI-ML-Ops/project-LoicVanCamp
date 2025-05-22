import pickle
from pathlib import Path
import pandas as pd

import click
import mlflow
import optuna

from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# MLflow config
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("student-performance-hpo")


def load_pickle(path: Path):
    with path.open("rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_dir",
    default="../../models",  # relatieve path t.o.v. scripts/
    help="Directory where model.pkl and dv.pkl are."
)
@click.option(
    "--num_trials",
    default=10,
    help="Number of experiments to try"
)
def run_optimization(data_dir: str, num_trials: int):
    data_path = Path(data_dir).resolve()
    model_path = data_path / "model.pkl"
    dv_path = data_path / "dv.pkl"

    # Load features & vectorizer
    df = pd.read_csv(Path(__file__).resolve(
    ).parents[1] / "data" / "StudentsPerformance.csv")
    df["pass_math"] = (df["math score"] >= 50).astype(int)

    categorical = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course"
    ]

    dv = load_pickle(dv_path)
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

    train_dicts = df_train[categorical].to_dict(orient="records")
    val_dicts = df_val[categorical].to_dict(orient="records")

    X_train = dv.transform(train_dicts)
    y_train = df_train["pass_math"]
    X_val = dv.transform(val_dicts)
    y_val = df_val["pass_math"]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "random_state": 42,
            "n_jobs": -1
        }

        with mlflow.start_run():
            mlflow.log_params(params)

            rf = RandomForestClassifier(**params)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            mlflow.log_metric("accuracy", acc)

            return -acc  # Optuna minimizes

    study = optuna.create_study(
        direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=num_trials)

    print("✅ Best score:", -study.best_value)
    print("✅ Best params:", study.best_params)


if __name__ == "__main__":
    run_optimization()
