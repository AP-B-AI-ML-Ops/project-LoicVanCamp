import os
import pickle
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://experiment-tracking:5000"))
mlflow.set_experiment("student-performance-train")

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def run_train(data_path: str):
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