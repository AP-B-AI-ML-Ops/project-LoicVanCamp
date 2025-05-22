import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

# Load the dataset
df = pd.read_csv("data/StudentsPerformance.csv")

# Create binary target: 1 if math score >= 50, else 0
df["pass_math"] = (df["math score"] >= 50).astype(int)

# Define categorical features
categorical = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

# Split into training and test sets
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Vectorize the training data
dv = DictVectorizer(sparse=False)
train_dicts = df_train[categorical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)
y_train = df_train["pass_math"]

# Start MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("student-performance")

with mlflow.start_run():
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Log model hyperparameters (none in this case, but good habit)
    mlflow.log_param("model_type", "LogisticRegression")

    # Evaluate on test data
    test_dicts = df_test[categorical].to_dict(orient='records')
    X_test = dv.transform(test_dicts)
    y_test = df_test["pass_math"]
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # Save and log the model and vectorizer
    with open("models/model.pkl", "wb") as f_out:
        pickle.dump(model, f_out)
    with open("models/dv.pkl", "wb") as f_out:
        pickle.dump(dv, f_out)

    mlflow.sklearn.log_model(model, artifact_path="model")
    print("✅ Model and vectorizer saved and logged")

    # Save and log plot
    sns.histplot(y_pred, label="Prediction", kde=False, stat="density")
    sns.histplot(y_test, label="Actual", kde=False, stat="density", color="orange")
    plt.legend()
    plt.title("Prediction vs Actual Pass/Fail")
    plot_path = "models/pred_vs_actual.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
