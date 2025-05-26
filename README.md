### 📝 `README.md` Template

# Student Exam Performance Prediction

## Dataset(s)
The dataset used in this project is the **Students Performance in Exams** dataset, available on Kaggle. It contains information about students' gender, race/ethnicity, parental education level, lunch type, test preparation course completion, and their scores in math, reading, and writing exams.

- **Source:** [Students Performance in Exams - Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Features:** Gender, Race/Ethnicity, Parental Level of Education, Lunch Type, Test Preparation Course, Math Score, Reading Score, Writing Score.
- **Target variable:** `pass_math` (binary classification: pass = 1, fail = 0 based on a score threshold of 50).
- **Disclaimer**: race/Ethnicity grouping categories could not be made out. I Found out too late to delete them from the project.

**Data splits:**
- **Training data:** 70% of the dataset
- **Validation data:** 10% of the dataset
- **Test data:** 20% of the dataset

New data can be submitted via the web service API, where users can input student details for prediction.

## Project Explanation

This project builds an end-to-end machine learning pipeline to predict whether a student will pass their math exam based on demographic and educational background features. The goal is to provide a web service where users can input student data and receive a real-time prediction (pass/fail) for the math exam.

The pipeline includes:
- Data preprocessing and feature engineering
- Model training and experiment tracking
- Model deployment as a web API (FastAPI)
- Automated batch prediction and monitoring
- Model performance monitoring and reporting

The application is fully containerized using Docker and orchestrated with Prefect, ensuring reproducibility and easy deployment.

## Flows & Actions

### Flows

1. **Model Training Flow (Prefect):**
   - Manages the entire training pipeline: data loading, preprocessing, model training, and logging metrics with MLflow.
   - Ensures reproducibility and tracks all experiments.

2. **Model Deployment Flow (FastAPI):**
   - Exposes the trained model through a web API.
   - Allows users to submit student data and receive predictions in real-time.

3. **Batch Prediction Flow:**
   - Periodically runs batch predictions on new data.
   - Saves results and metrics for monitoring.

4. **Monitoring Flow (Evidently + Prefect):**
   - Monitors the deployed model’s performance and data drift.
   - Generates reports and saves metrics to a database and Grafana dashboard.
   - Can trigger retraining or alerts if performance drops below a threshold.

### Actions

1. **Data Preprocessing:**
   - Cleans and encodes categorical features.
   - Splits the data into training, validation, and test sets.
   - Saves processed data and vectorizer as MLflow artifacts.

2. **Model Training:**
   - Trains a machine learning model (RandomForestClassifier).
   - Tracks experiments and hyperparameter optimization with MLflow.
   - Registers the best model in the MLflow Model Registry.

3. **Model Deployment:**
   - Deploys the trained model as a FastAPI application for prediction requests.
   - Containerized for reproducibility and easy deployment.

4. **Batch Prediction:**
   - Runs batch predictions on new data files.
   - Logs results and metrics for monitoring.

5. **Model Monitoring:**
   - Tracks performance and data drift using Evidently.
   - Saves metrics to a PostgreSQL database and visualizes them in Grafana.
   - Supports conditional workflows (e.g., retraining if performance degrades).

## Technologies Used

- **MLflow:** Experiment tracking and model registry
- **Prefect:** Workflow orchestration
- **Evidently:** Model monitoring and data drift detection
- **Grafana:** Visualization of model metrics
- **FastAPI:** Web service for real-time predictions
- **Docker & Docker Compose:** Containerization and orchestration
- **PostgreSQL:** Backend database for MLflow and monitoring metrics

## Reproducibility & Instructions

- All code, data splits, and dependencies are versioned and tracked.
- To run the project:
  1. Clone the repository.
  2. Ensure Docker and Docker Compose are installed and running.
  3. Run `docker compose up --build` from the project root.
  4. Access the API at [http://localhost:8000](http://localhost:8000).
  5. Access MLflow UI at [http://localhost:5000](http://localhost:5000).
  6. Access Grafana at [http://localhost:3400](http://localhost:3400) (login: admin/admin).
  7. Access Prefect at [http://localhost:4200/](http://localhost:4200/).

- All dependencies are specified in `requirements.txt` and `pyproject.toml`.
- Pre-commit hooks and unit tests are included for code quality and reproducibility.

---

**This project demonstrates a full MLOps workflow: from data and experiment tracking, to automated training, deployment, and monitoring, all orchestrated and containerized for production readiness.**
