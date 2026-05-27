# src/train.py

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
from sklearn.model_selection import train_test_split

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Load processed data
df = pd.read_csv("data/processed/processed.csv")

# Features and target
X = df.drop(columns=["RainTomorrow"])
y = df["RainTomorrow"]

# Identify column types
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

# Preprocessing
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
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
        max_iter=params["train"]["max_iter"],
        C=params["train"]["C"],
        random_state=params["train"]["random_state"]
    ))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["train"]["test_size"],
    random_state=params["train"]["random_state"]
)

# MLflow tracking
mlflow.set_experiment("rain-prediction")

with mlflow.start_run():
    # Log params
    mlflow.log_param("max_iter", params["train"]["max_iter"])
    mlflow.log_param("C", params["train"]["C"])
    mlflow.log_param("test_size", params["train"]["test_size"])

    # Train
    model_pipeline.fit(X_train, y_train)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_pipeline, "models/model.pkl")

    # Log model to MLflow
    mlflow.sklearn.log_model(model_pipeline, "model")

    print("Model trained and saved to models/model.pkl")
