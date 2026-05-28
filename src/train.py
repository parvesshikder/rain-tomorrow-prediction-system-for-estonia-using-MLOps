import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

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

# Preprocessing
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy=params["preprocessing"]["missing_numeric_strategy"])),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy=params["preprocessing"]["missing_categorical_strategy"])),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Full pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=params["model"]["max_iter"],
        C=1.0,
        random_state=params["model"]["random_state"],
        class_weight=params["model"]["class_weight"]
    ))
])

# Chronological train/test split
test_size = params["model"]["test_size"]
split_index = int(len(df) * (1 - test_size))
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# MLflow tracking
mlflow.set_tracking_uri(params["tracking"]["mlflow_tracking_uri"])
mlflow.set_experiment(params["tracking"]["experiment_name"])

with mlflow.start_run():
    mlflow.log_param("max_iter", params["model"]["max_iter"])
    mlflow.log_param("test_size", params["model"]["test_size"])
    mlflow.log_param("random_state", params["model"]["random_state"])
    mlflow.log_param("class_weight", params["model"]["class_weight"])

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    mlflow.log_metrics(metrics)

    model_path = params["artifacts"]["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_pipeline, model_path)

    mlflow.sklearn.log_model(model_pipeline, "model", registered_model_name="EstoniaRainModel")
    print(f"Model trained and saved to {model_path}")

    # Automated Model Promotion
    client = mlflow.tracking.MlflowClient()
    latest_versions = client.get_latest_versions("EstoniaRainModel", stages=["None"])
    if latest_versions:
        latest_version = latest_versions[0].version
        prod_versions = client.get_latest_versions("EstoniaRainModel", stages=["Production"])
        
        current_accuracy = metrics["accuracy"]
        
        if not prod_versions:
            client.transition_model_version_stage(
                name="EstoniaRainModel", version=latest_version, stage="Production"
            )
            print("First model registered and promoted to Production.")
        else:
            prod_run = client.get_run(prod_versions[0].run_id)
            prod_accuracy = prod_run.data.metrics.get("accuracy", 0.0)
            
            if current_accuracy > prod_accuracy:
                client.transition_model_version_stage(
                    name="EstoniaRainModel", version=latest_version, stage="Production"
                )
                print(f"Model Promoted! New Accuracy ({current_accuracy:.4f}) > Old ({prod_accuracy:.4f})")
            else:
                client.transition_model_version_stage(
                    name="EstoniaRainModel", version=latest_version, stage="Archived"
                )
                print(f"Model Rejected. New Accuracy ({current_accuracy:.4f}) <= Old ({prod_accuracy:.4f})")
