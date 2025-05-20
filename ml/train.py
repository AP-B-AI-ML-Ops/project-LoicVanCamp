import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

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

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate on test data
test_dicts = df_test[categorical].to_dict(orient='records')
X_test = dv.transform(test_dicts)
y_test = df_test["pass_math"]
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot prediction vs actual
sns.histplot(y_pred, label="Prediction", kde=False, stat="density")
sns.histplot(y_test, label="Actual", kde=False, stat="density", color="orange")
plt.legend()
plt.title("Prediction vs Actual Pass/Fail")
plt.show()

# Save model and vectorizer
with open("models/model.pkl", "wb") as f_out:
    pickle.dump(model, f_out)

with open("models/dv.pkl", "wb") as f_out:
    pickle.dump(dv, f_out)

print("✅ Model and vectorizer saved to 'models/'")
