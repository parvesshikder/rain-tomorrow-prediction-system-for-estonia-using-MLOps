import json
import os
from datetime import datetime, timezone
from typing import Any

from src.config import PARAMS


def _prediction_log_path() -> str:
    return PARAMS["monitoring"]["prediction_log_path"]


def log_prediction(
    city: str,
    input_features: dict[str, Any],
    prediction: str,
    rain_probability: float,
) -> dict[str, Any]:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "city": city,
        "input_features": input_features,
        "prediction": prediction,
        "rain_probability": round(float(rain_probability), 4),
    }

    path = _prediction_log_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def recent_predictions(limit: int = 20) -> list[dict[str, Any]]:
    path = _prediction_log_path()
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()[-limit:]

    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return entries
