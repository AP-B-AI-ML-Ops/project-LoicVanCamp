### 📝 `README.md` Template

# Student Exam Performance Prediction

## Dataset(s)
The dataset used in this project is the **Students Performance in Exams** dataset, available on Kaggle. It contains information about students' gender, race/ethnicity, parental education level, lunch type, test preparation course completion, and their scores in math, reading, and writing exams.

- **Source:** [Students Performance in Exams - Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Dataset details:**
  - **Features:** Gender, Race/Ethnicity, Parental Level of Education, Lunch Type, Test Preparation Course, Math Score, Reading Score, Writing Score.
  - **Target variable:** `pass_math` (binary classification: pass = 1, fail = 0 based on a score threshold of 50).

We will use this dataset to train a model to predict whether a student will pass their math exam based on the input features. The dataset will be split into training (80%) and test (20%) sets.

For training, we will use:
- **Training data:** 70% of the dataset
- **Validation data:** 10% of the dataset
- **Test data:** 20% of the dataset
- New data will be input via a web service API where users can submit new student details for prediction.

## Project Explanation
This project aims to build a machine learning model that predicts whether a student will pass their math exam based on various factors such as their background, parental education level, test preparation, and lunch type. The goal is to create a web service that allows anyone to input their data (e.g., gender, lunch type) and get a prediction about whether they will pass the math exam.

**Steps involved:**
1. **Data Preprocessing:** Clean the dataset and prepare it for model training.
2. **Model Training:** Train a machine learning model (e.g., Random Forest Classifier) to predict if a student will pass or fail their math exam based on the provided features.
3. **Model Deployment:** Deploy the trained model as a web service (FastAPI) for real-time predictions.
4. **Monitoring:** Monitor the model’s performance and retrain it when necessary.

The overall goal is to build an end-to-end machine learning pipeline that integrates model development, deployment, and monitoring with a web service application.

## Flows & Actions

### Flows:
1. **Model Training Flow (Prefect):**
   - This flow will manage the entire training pipeline, from data loading and preprocessing to model training and logging metrics with MLflow.
   - It will track the model training process and ensure reproducibility of results.

2. **Model Deployment Flow (FastAPI):**
   - This flow will expose the trained model through a web API, allowing users to submit student data and receive predictions.
   - The API will serve predictions in real-time based on the trained model.

3. **Monitoring Flow (Evidently):**
   - This flow will monitor the deployed model’s performance and generate reports (e.g., accuracy, confusion matrix) to check if the model’s performance is degrading over time.
   - If necessary, the flow will trigger a retraining action if a performance threshold is violated.

### Actions:
1. **Data Preprocessing:**
   - Prepare the dataset by encoding categorical features and splitting the data into training and test sets.

2. **Model Training:**
   - Use a machine learning algorithm (e.g., RandomForestClassifier) to train the model.
   - Log experiments and models using **MLflow** for versioning and tracking.

3. **Model Deployment:**
   - Deploy the model as a FastAPI application for prediction requests.
   - Set up the infrastructure for API hosting (e.g., Docker, Docker Compose).

4. **Model Monitoring:**
   - Track performance over time and retrain the model when necessary.
