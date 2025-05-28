"""Web service for student exam prediction using FastAPI.

This module provides a web interface for users to input student data and receive
a math exam pass/fail prediction using a trained model.
"""

# pylint: disable=invalid-name,redefined-outer-name,too-many-positional-arguments

import os
import pickle

import mlflow
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from mlflow.tracking import MlflowClient

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- Load model and DictVectorizer from MLflow at startup ---

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://experiment-tracking:5000"
)
MODEL_NAME = "rf-math-pass-predictor"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(MLFLOW_TRACKING_URI)


def load_latest_model_and_dv():
    """Load the latest model and DictVectorizer from MLflow Model Registry."""
    versions = client.get_latest_versions(MODEL_NAME)
    if not versions:
        raise RuntimeError(f"No versions found for model '{MODEL_NAME}' in MLflow.")
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{latest.version}")

    # Download DictVectorizer artifact from the same run
    dv_path = mlflow.artifacts.download_artifacts(
        run_id=latest.run_id, artifact_path="model/dv.pkl"
    )
    with open(dv_path, "rb") as f:
        dv = pickle.load(f)
    return model, dv


model, dv = load_latest_model_and_dv()


@app.get("/", response_class=HTMLResponse)
def form_get(request: Request, prediction: str = None):
    """Render the prediction form.

    Args:
        request (Request): FastAPI request object.
        prediction (str, optional): Prediction result to display.

    Returns:
        HTMLResponse: Rendered HTML form.
    """
    return templates.TemplateResponse(
        "form.html", {"request": request, "prediction": prediction}
    )


@app.post("/", response_class=HTMLResponse)
def form_post(
    request: Request,
    gender: str = Form(...),
    race_ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    math_score: int = Form(...),
    reading_score: int = Form(...),
    writing_score: int = Form(...),
):
    """Handle form submission and return prediction.

    Args:
        request (Request): FastAPI request object.
        gender (str): Student gender.
        race_ethnicity (str): Student race/ethnicity.
        parental_level_of_education (str): Parental education level.
        lunch (str): Lunch type.
        test_preparation_course (str): Test preparation course status.
        math_score (int): Math score.
        reading_score (int): Reading score.
        writing_score (int): Writing score.

    Returns:
        HTMLResponse: Rendered HTML form with prediction.
    """
    features = {
        "gender": gender,
        "race/ethnicity": race_ethnicity,
        "parental level of education": parental_level_of_education,
        "lunch": lunch,
        "test preparation course": test_preparation_course,
        "math score": math_score,
        "reading score": reading_score,
        "writing score": writing_score,
    }
    try:
        X = dv.transform([features])
        pred = model.predict(X)[0]
        prediction = "Pass" if pred == 1 else "Fail"
    except (ValueError, KeyError, AttributeError) as error:
        prediction = f"Error: {error}"
    return templates.TemplateResponse(
        "form.html", {"request": request, "prediction": prediction}
    )
