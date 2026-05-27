import pandas as pd
import json
import yaml
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Load processed data
df = pd.read_csv(params["data"]["processed_path"])

# Features and target
target = params["target"]["name"]
categorical_cols = params["features"]["categorical"]
numerical_cols = params["features"]["numeric"]

X = df[categorical_cols + numerical_cols]
y = df[target]

# Chronological split (same as train.py)
test_size = params["model"]["test_size"]
split_index = int(len(df) * (1 - test_size))
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

# Load trained model
model_path = params["artifacts"]["model_path"]
model = joblib.load(model_path)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall    = recall_score(y_test, y_pred, average="weighted")
roc_auc   = roc_auc_score(y_test, y_prob)

# Print results
print(f"Accuracy  : {accuracy:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save metrics
metrics_path = params["evaluation"]["metrics_path"]
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

metrics = {
    "accuracy":  round(accuracy, 4),
    "f1_score":  round(f1, 4),
    "precision": round(precision, 4),
    "recall":    round(recall, 4),
    "roc_auc":   round(roc_auc, 4),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\nMetrics saved to {metrics_path}")
