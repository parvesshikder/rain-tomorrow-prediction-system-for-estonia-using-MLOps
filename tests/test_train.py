# tests/test_train.py

import pytest
import pandas as pd
import numpy as np
import joblib
import json
import yaml
import os

# Load params once for all tests
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

PROCESSED_PATH = params["data"]["processed_path"]
MODEL_PATH     = params["artifacts"]["model_path"]
METRICS_PATH   = params["evaluation"]["metrics_path"]
TARGET         = params["target"]["name"]
CAT_COLS       = params["features"]["categorical"]
NUM_COLS       = params["features"]["numeric"]


# ── Data Tests ──────────────────────────────────────────────────────────────

def test_processed_file_exists():
    """Processed dataset must exist."""
    assert os.path.exists(PROCESSED_PATH), f"Missing: {PROCESSED_PATH}"

def test_processed_data_not_empty():
    """Processed dataset must have rows."""
    df = pd.read_csv(PROCESSED_PATH)
    assert len(df) > 0, "Processed data is empty"

def test_required_columns_exist():
    """All feature columns and target must be present."""
    df = pd.read_csv(PROCESSED_PATH)
    required = CAT_COLS + NUM_COLS + [TARGET]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"

def test_target_is_binary():
    """Target column must contain only 0 and 1."""
    df = pd.read_csv(PROCESSED_PATH)
    unique_values = set(df[TARGET].dropna().unique())
    assert unique_values.issubset({0, 1}), \
        f"Target has unexpected values: {unique_values}"

def test_no_all_null_columns():
    """No column should be entirely null."""
    df = pd.read_csv(PROCESSED_PATH)
    for col in CAT_COLS + NUM_COLS:
        assert df[col].notna().any(), f"Column entirely null: {col}"


# ── Model Tests ──────────────────────────────────────────────────────────────

def test_model_file_exists():
    """Trained model file must exist."""
    assert os.path.exists(MODEL_PATH), f"Missing: {MODEL_PATH}"

def test_model_loads_successfully():
    """Model must load without errors."""
    model = joblib.load(MODEL_PATH)
    assert model is not None

def test_model_has_predict_method():
    """Model must have a predict method."""
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")

def test_model_has_predict_proba_method():
    """Model must have predict_proba for ROC AUC."""
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict_proba")

def test_model_predicts_correct_shape():
    """Model predictions must match number of input rows."""
    df = pd.read_csv(PROCESSED_PATH)
    X = df[CAT_COLS + NUM_COLS].head(10)
    model = joblib.load(MODEL_PATH)
    preds = model.predict(X)
    assert len(preds) == 10, "Prediction count mismatch"

def test_model_predicts_binary_values():
    """Model must predict only 0 or 1."""
    df = pd.read_csv(PROCESSED_PATH)
    X = df[CAT_COLS + NUM_COLS].head(50)
    model = joblib.load(MODEL_PATH)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1}), \
        f"Unexpected prediction values: {set(preds)}"


# ── Metrics Tests ─────────────────────────────────────────────────────────────

def test_metrics_file_exists():
    """Metrics report must exist."""
    assert os.path.exists(METRICS_PATH), f"Missing: {METRICS_PATH}"

def test_metrics_has_required_keys():
    """Metrics must contain all required keys."""
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    for key in ["accuracy", "f1_score", "precision", "recall", "roc_auc"]:
        assert key in metrics, f"Missing metric: {key}"

def test_accuracy_is_reasonable():
    """Accuracy must be above 50% (better than random)."""
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    assert metrics["accuracy"] > 0.5, \
        f"Accuracy too low: {metrics['accuracy']}"

def test_roc_auc_is_reasonable():
    """ROC AUC must be above 0.5 (better than random)."""
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    assert metrics["roc_auc"] > 0.5, \
        f"ROC AUC too low: {metrics['roc_auc']}"
