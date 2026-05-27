# src/evaluate.py

import pandas as pd
import json
import yaml
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Load processed data
df = pd.read_csv("data/processed/processed.csv")

# Features and target
X = df.drop(columns=["RainTomorrow"])
y = df["RainTomorrow"]

# Split (same seed as training to get the same test set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["train"]["test_size"],
    random_state=params["train"]["random_state"]
)

# Load trained model
model = joblib.load("models/model.pkl")

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save metrics to reports/
os.makedirs("reports", exist_ok=True)
metrics = {
    "accuracy": round(accuracy, 4),
    "f1_score": round(f1, 4)
}

with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved to reports/metrics.json")
