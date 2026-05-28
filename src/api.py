from datetime import date as Date
from datetime import timedelta
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, field_validator

from src.config import PARAMS
from src.monitor import log_prediction, recent_predictions


app = FastAPI(
    title=PARAMS["project"]["title"],
    description="FastAPI deployment for Estonia rain-tomorrow prediction.",
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app)


class PredictionRequest(BaseModel):
    city: str = Field(default="Tallinn")
    latitude: float | None = None
    longitude: float | None = None
    date: Date = Field(default_factory=Date.today)
    temperature_2m_max: float = 10.0
    temperature_2m_min: float = 3.0
    temperature_2m_mean: float = 6.5
    relative_humidity_2m_mean: float = 80.0
    precipitation_sum: float = 0.0
    rain_sum: float = 0.0
    wind_speed_10m_max: float = 15.0
    wind_gusts_10m_max: float = 25.0
    pressure_msl_mean: float = 1013.0
    RainToday: int | None = None
    month: int | None = None
    day_of_year: int | None = None

    @field_validator("RainToday")
    @classmethod
    def validate_binary_rain_today(cls, value: int | None) -> int | None:
        if value is not None and value not in {0, 1}:
            raise ValueError("RainToday must be 0 or 1")
        return value


class PredictionResponse(BaseModel):
    city: str
    prediction: str
    rain_probability: float
    model_path: str
    logged: bool


def _locations_by_city() -> dict[str, dict[str, Any]]:
    return {loc["city"].lower(): loc for loc in PARAMS["data"]["locations"]}


def _resolve_location(city: str) -> dict[str, Any]:
    location = _locations_by_city().get(city.lower())
    if not location:
        valid = ", ".join(loc["city"] for loc in PARAMS["data"]["locations"])
        raise HTTPException(status_code=400, detail=f"Unknown city. Choose one of: {valid}")
    return location


def _prepared_features(payload: PredictionRequest) -> dict[str, Any]:
    values = payload.model_dump()
    location = _resolve_location(payload.city)

    values["city"] = location["city"]
    values["latitude"] = payload.latitude if payload.latitude is not None else location["latitude"]
    values["longitude"] = payload.longitude if payload.longitude is not None else location["longitude"]
    values["RainToday"] = (
        payload.RainToday
        if payload.RainToday is not None
        else int(payload.precipitation_sum >= PARAMS["target"]["precipitation_threshold_mm"])
    )
    values["month"] = payload.month if payload.month is not None else payload.date.month
    values["day_of_year"] = payload.day_of_year if payload.day_of_year is not None else payload.date.timetuple().tm_yday
    values.pop("date", None)

    feature_columns = PARAMS["features"]["categorical"] + PARAMS["features"]["numeric"]
    return {column: values[column] for column in feature_columns}


@lru_cache(maxsize=1)
def load_model():
    model_path = PARAMS["artifacts"]["model_path"]
    try:
        return joblib.load(model_path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {model_path}. Run the training pipeline first.",
        ) from exc


def _predict_from_features(features: dict[str, Any]) -> tuple[int, float]:
    model = load_model()
    frame = pd.DataFrame([features])
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0][1])
    return prediction, probability


@app.get("/", response_class=HTMLResponse)
def browser_ui() -> str:
    cities = "\n".join(
        f'<option value="{loc["city"]}">{loc["city"]}</option>'
        for loc in PARAMS["data"]["locations"]
    )
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Estonia Rain Predictor</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Arial, sans-serif;
      background: #f4f7fa;
      color: #172033;
    }}
    body {{
      margin: 0;
      padding: 24px;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    h2 {{
      margin: 0;
      font-size: 18px;
    }}
    p {{
      line-height: 1.5;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(320px, .9fr);
      gap: 18px;
      margin-top: 18px;
    }}
    form, .panel {{
      background: #ffffff;
      border: 1px solid #d8dee9;
      border-radius: 8px;
      padding: 16px;
    }}
    .form-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .status {{
      display: inline-flex;
      margin-top: 8px;
      border-radius: 999px;
      padding: 5px 9px;
      background: #eef2f7;
      color: #475569;
      font-size: 12px;
      font-weight: 700;
    }}
    .status[data-tone="ok"] {{
      background: #e7f5ee;
      color: #12613f;
    }}
    .status[data-tone="error"] {{
      background: #fff0f0;
      color: #a22121;
    }}
    label {{
      display: grid;
      gap: 5px;
      font-size: 13px;
      font-weight: 700;
      color: #334155;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    input, select, button {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #c9d2df;
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      margin-top: 14px;
    }}
    button {{
      border: 0;
      background: #0f6b5f;
      color: white;
      font-weight: 700;
      cursor: pointer;
    }}
    button.secondary {{
      max-width: 190px;
      background: #174ea6;
    }}
    button:disabled {{
      cursor: not-allowed;
      opacity: .65;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-size: 13px;
    }}
    .result-card {{
      border: 1px solid #d8dee9;
      border-radius: 8px;
      padding: 18px;
      background: #f8fafc;
      margin-bottom: 14px;
    }}
    .result-card.rainy {{
      border-color: #7aa7d9;
      background: #eef6ff;
    }}
    .result-card.dry {{
      border-color: #87c7a3;
      background: #effaf3;
    }}
    .eyebrow {{
      margin: 0 0 6px;
      color: #64748b;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .04em;
      text-transform: uppercase;
    }}
    .result {{
      font-size: 22px;
      font-weight: 800;
      margin-bottom: 4px;
    }}
    .probability-display {{
      display: grid;
      place-items: center;
      min-height: 150px;
      border-radius: 8px;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      margin: 14px 0;
    }}
    .probability-number {{
      font-size: 72px;
      line-height: 1;
      font-weight: 900;
      color: #0f6b5f;
    }}
    .probability-caption {{
      margin-top: 6px;
      color: #64748b;
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .result-card.rainy .probability-number {{
      color: #174ea6;
    }}
    .result-card.dry .probability-number {{
      color: #0f6b5f;
    }}
    .result-note {{
      margin-top: 8px;
      color: #475569;
      font-size: 14px;
    }}
    .result-note strong {{
      color: #172033;
    }}
    .muted {{
      color: #64748b;
      font-size: 13px;
    }}
    .weather-card {{
      border-top: 1px solid #e2e8f0;
      padding-top: 14px;
    }}
    .weather-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin: 12px 0;
    }}
    .metric {{
      border-radius: 8px;
      background: #f8fafc;
      padding: 10px;
    }}
    .metric span {{
      display: block;
      color: #64748b;
      font-size: 12px;
      font-weight: 700;
    }}
    .metric strong {{
      display: block;
      margin-top: 4px;
      font-size: 18px;
    }}
    details {{
      margin-top: 12px;
    }}
    summary {{
      cursor: pointer;
      color: #334155;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    @media (max-width: 780px) {{
      .layout, .grid, .weather-grid {{
        grid-template-columns: 1fr;
      }}
      .form-head, .actions {{
        display: grid;
      }}
      button.secondary {{
        max-width: none;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Estonia Rain Tomorrow Predictor</h1>
    <p>Load recent weather and tomorrow forecast data for an Estonian city, then estimate the chance of rain.</p>
    <div class="layout">
      <form id="predict-form">
        <div class="form-head">
          <div>
            <h2>Weather Inputs</h2>
            <span id="weather-status" class="status">No weather data loaded</span>
          </div>
          <button type="button" id="load-weather" class="secondary">Load Forecast Data</button>
        </div>
        <input name="RainToday" type="hidden">
        <div class="grid">
          <label>City<select name="city">{cities}</select></label>
          <label>Date<input name="date" type="date"></label>
          <label>Max temperature<input name="temperature_2m_max" type="number" step="0.1" placeholder="Load weather data"></label>
          <label>Min temperature<input name="temperature_2m_min" type="number" step="0.1" placeholder="Load weather data"></label>
          <label>Mean temperature<input name="temperature_2m_mean" type="number" step="0.1" placeholder="Load weather data"></label>
          <label>Humidity<input name="relative_humidity_2m_mean" type="number" step="0.1" placeholder="Load weather data"></label>
          <label>Precipitation<input name="precipitation_sum" type="number" step="0.1" placeholder="Load weather data"></label>
          <label>Rain amount<input name="rain_sum" type="number" step="0.1" placeholder="Load weather data"></label>
          <label>Wind speed<input name="wind_speed_10m_max" type="number" step="0.1" placeholder="Load weather data"></label>
          <label>Wind gusts<input name="wind_gusts_10m_max" type="number" step="0.1" placeholder="Load weather data"></label>
          <label>Pressure<input name="pressure_msl_mean" type="number" step="0.1" placeholder="Load weather data"></label>
        </div>
        <div class="actions">
          <button type="submit" id="predict-button" disabled>Estimate Rain Chance</button>
        </div>
      </form>
      <section class="panel">
        <div id="result-card" class="result-card">
          <p class="eyebrow">Prediction</p>
          <div id="result" class="result">No prediction yet</div>
          <div class="probability-display">
            <div>
              <div id="probability-number" class="probability-number">--%</div>
              <div class="probability-caption">Estimated rain chance</div>
            </div>
          </div>
          <div id="probability-label" class="result-note">Load forecast data to enable prediction.</div>
        </div>
        <div class="weather-card">
          <p class="eyebrow">Forecast Inputs</p>
          <div class="weather-grid">
            <div class="metric"><span>City</span><strong id="summary-city">-</strong></div>
            <div class="metric"><span>Humidity</span><strong id="summary-humidity">-</strong></div>
            <div class="metric"><span>Temperature</span><strong id="summary-temp">-</strong></div>
            <div class="metric"><span>Precipitation</span><strong id="summary-precip">-</strong></div>
          </div>
        </div>
        <details>
          <summary>Response details</summary>
          <pre id="details">Load forecast data or run a prediction to see details.</pre>
        </details>
      </section>
    </div>
  </main>
  <script>
    const form = document.querySelector("#predict-form");
    const loadButton = document.querySelector("#load-weather");
    const predictButton = document.querySelector("#predict-button");
    const dateInput = form.elements.date;
    const weatherStatus = document.querySelector("#weather-status");
    const resultCard = document.querySelector("#result-card");
    const result = document.querySelector("#result");
    const details = document.querySelector("#details");
    const probabilityNumber = document.querySelector("#probability-number");
    const probabilityLabel = document.querySelector("#probability-label");
    const requiredWeatherFields = [
      "temperature_2m_max",
      "temperature_2m_min",
      "temperature_2m_mean",
      "relative_humidity_2m_mean",
      "precipitation_sum",
      "rain_sum",
      "wind_speed_10m_max",
      "wind_gusts_10m_max",
      "pressure_msl_mean"
    ];

    function setStatus(message, tone = "neutral") {{
      weatherStatus.textContent = message;
      weatherStatus.dataset.tone = tone;
    }}

    function setInputValue(name, value) {{
      const input = form.elements[name];
      if (!input || value === undefined || value === null) return;
      if (input.type === "number") {{
        input.value = Number(value).toFixed(1).replace(/\\.0$/, "");
      }} else {{
        input.value = value;
      }}
    }}

    function hasCompleteInputs() {{
      return requiredWeatherFields.every((name) => form.elements[name].value !== "");
    }}

    function updatePredictButton() {{
      predictButton.disabled = !hasCompleteInputs();
    }}

    function resetWeatherInputs() {{
      dateInput.value = "";
      for (const name of requiredWeatherFields) {{
        form.elements[name].value = "";
      }}
      form.elements.RainToday.value = "";
      document.querySelector("#summary-city").textContent = "-";
      document.querySelector("#summary-humidity").textContent = "-";
      document.querySelector("#summary-temp").textContent = "-";
      document.querySelector("#summary-precip").textContent = "-";
      resultCard.className = "result-card";
      result.textContent = "No prediction yet";
      probabilityNumber.textContent = "--%";
      probabilityLabel.textContent = "Load forecast data to enable prediction.";
      details.textContent = "Load forecast data or run a prediction to see details.";
      setStatus("No weather data loaded");
      updatePredictButton();
    }}

    function payloadFromForm() {{
      const data = Object.fromEntries(new FormData(form).entries());
      for (const key of Object.keys(data)) {{
        if (data[key] === "") {{
          delete data[key];
        }} else if (key !== "city" && key !== "date") {{
          data[key] = Number(data[key]);
        }}
      }}
      return data;
    }}

    function updateWeatherSummary(input) {{
      document.querySelector("#summary-city").textContent = input.city || form.elements.city.value;
      document.querySelector("#summary-humidity").textContent =
        input.relative_humidity_2m_mean !== undefined ? `${{input.relative_humidity_2m_mean}}%` : "-";
      document.querySelector("#summary-temp").textContent =
        input.temperature_2m_mean !== undefined ? `${{input.temperature_2m_mean}} C` : "-";
      document.querySelector("#summary-precip").textContent =
        input.precipitation_sum !== undefined ? `${{input.precipitation_sum}} mm` : "-";
    }}

    function fillFormFromWeather(input) {{
      for (const [key, value] of Object.entries(input)) {{
        setInputValue(key, value);
      }}
      if (input.date) setInputValue("date", input.date);
      updateWeatherSummary(input);
      updatePredictButton();
    }}

    form.elements.city.addEventListener("change", resetWeatherInputs);
    form.addEventListener("input", updatePredictButton);

    loadButton.addEventListener("click", async () => {{
      const city = form.elements.city.value;
      loadButton.disabled = true;
      setStatus(`Loading forecast data for ${{city}}...`);
      details.textContent = "Requesting recent weather and tomorrow forecast from Open-Meteo...";
      try {{
        const response = await fetch(`/weather/forecast-input?city=${{encodeURIComponent(city)}}`);
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || "Weather request failed");
        fillFormFromWeather(body.prediction_input || {{}});
        setStatus(`Loaded forecast inputs for ${{body.city}}`, "ok");
        details.textContent = JSON.stringify({{
          prediction_input: body.prediction_input,
          recent_history: body.recent_history,
          tomorrow_forecast: body.tomorrow_forecast
        }}, null, 2);
      }} catch (error) {{
        setStatus("Could not load weather", "error");
        details.textContent = error.message;
      }} finally {{
        loadButton.disabled = false;
      }}
    }});

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      const data = payloadFromForm();
      predictButton.disabled = true;
      resultCard.className = "result-card";
      result.textContent = "Predicting...";
      probabilityNumber.textContent = "--%";
      probabilityLabel.textContent = "Running the trained model.";
      const response = await fetch("/predict", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(data)
      }});
      const body = await response.json();
      if (!response.ok) {{
        result.textContent = "Prediction failed";
        probabilityLabel.textContent = "Check the response details below.";
        details.textContent = JSON.stringify(body, null, 2);
        predictButton.disabled = false;
        return;
      }}
      const probability = Math.round(body.rain_probability * 100);
      const isRainy = body.prediction === "Yes";
      resultCard.className = `result-card ${{isRainy ? "rainy" : "dry"}}`;
      result.textContent = isRainy
        ? "Rain is likely tomorrow."
        : "Rain is unlikely tomorrow.";
      probabilityNumber.textContent = `${{probability}}%`;
      probabilityLabel.innerHTML = `Estimated for <strong>${{body.city}}</strong> using recent weather and tomorrow forecast.`;
      details.textContent = JSON.stringify(body, null, 2);
      predictButton.disabled = false;
    }});
  </script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict[str, Any]:
    model_path = PARAMS["artifacts"]["model_path"]
    try:
        load_model()
        model_loaded = True
    except HTTPException:
        model_loaded = False

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_path": model_path,
        "prediction_log_path": PARAMS["monitoring"]["prediction_log_path"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    features = _prepared_features(payload)
    prediction, probability = _predict_from_features(features)
    prediction_label = (
        PARAMS["project"]["target_output"]["positive_class"]
        if prediction == 1
        else PARAMS["project"]["target_output"]["negative_class"]
    )

    log_prediction(
        city=features["city"],
        input_features=features,
        prediction=prediction_label,
        rain_probability=probability,
    )

    return PredictionResponse(
        city=features["city"],
        prediction=prediction_label,
        rain_probability=round(probability, 4),
        model_path=PARAMS["artifacts"]["model_path"],
        logged=True,
    )


@app.get("/weather/forecast-input")
@app.get("/weather/current")
def forecast_weather_input(city: str = Query(default="Tallinn")) -> dict[str, Any]:
    location = _resolve_location(city)
    today = Date.today()
    tomorrow = today + timedelta(days=1)
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "wind_speed_10m",
            "wind_gusts_10m",
            "pressure_msl",
        ]),
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "pressure_msl_mean",
        ]),
        "timezone": PARAMS["data"]["timezone"],
        "past_days": 3,
        "forecast_days": 2,
    }

    try:
        response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
        response.raise_for_status()
        weather = response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Weather API request failed: {exc}") from exc

    current = weather.get("current", {})
    daily = weather.get("daily", {})
    daily_records = _daily_records(daily)
    if not daily_records:
        raise HTTPException(status_code=502, detail="Weather API response did not include daily forecast data.")

    tomorrow_record = _record_for_date(daily_records, tomorrow) or daily_records[-1]
    recent_history = _recent_records_before(daily_records, tomorrow)
    latest_recent = recent_history[-1] if recent_history else {}

    temp_max = _feature_value(tomorrow_record, recent_history, "temperature_2m_max", current.get("temperature_2m", 10.0))
    temp_min = _feature_value(tomorrow_record, recent_history, "temperature_2m_min", current.get("temperature_2m", 3.0))
    temp_mean = _feature_value(tomorrow_record, recent_history, "temperature_2m_mean", None)
    if temp_mean is None:
        temp_mean = round((float(temp_max) + float(temp_min)) / 2, 2)
    precipitation = _feature_value(tomorrow_record, recent_history, "precipitation_sum", current.get("precipitation", 0.0))
    rain = _feature_value(tomorrow_record, recent_history, "rain_sum", current.get("rain", 0.0))
    recent_precipitation = _record_float(latest_recent, "precipitation_sum")
    if recent_precipitation is None:
        recent_precipitation = _safe_float(current.get("precipitation"), 0.0)

    prediction_input = {
        "city": location["city"],
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "date": tomorrow.isoformat(),
        "temperature_2m_max": temp_max,
        "temperature_2m_min": temp_min,
        "temperature_2m_mean": temp_mean,
        "relative_humidity_2m_mean": _feature_value(
            tomorrow_record,
            recent_history,
            "relative_humidity_2m_mean",
            current.get("relative_humidity_2m", 80.0),
        ),
        "precipitation_sum": precipitation,
        "rain_sum": rain,
        "wind_speed_10m_max": _feature_value(tomorrow_record, recent_history, "wind_speed_10m_max", current.get("wind_speed_10m", 15.0)),
        "wind_gusts_10m_max": _feature_value(tomorrow_record, recent_history, "wind_gusts_10m_max", current.get("wind_gusts_10m", 25.0)),
        "pressure_msl_mean": _feature_value(tomorrow_record, recent_history, "pressure_msl_mean", current.get("pressure_msl", 1013.0)),
        "RainToday": int(recent_precipitation >= PARAMS["target"]["precipitation_threshold_mm"]),
    }

    return {
        "city": location["city"],
        "source": "Open-Meteo Forecast API",
        "basis": "Tomorrow forecast features with recent weather history for context.",
        "forecast_date": tomorrow.isoformat(),
        "history_days": len(recent_history),
        "current": current,
        "daily": daily,
        "recent_history": recent_history,
        "tomorrow_forecast": tomorrow_record,
        "prediction_input": prediction_input,
    }


def _daily_records(daily: dict[str, Any]) -> list[dict[str, Any]]:
    list_lengths = [len(value) for value in daily.values() if isinstance(value, list)]
    record_count = max(list_lengths, default=0)
    records = []
    for index in range(record_count):
        record = {}
        for key, values in daily.items():
            if isinstance(values, list) and index < len(values):
                record[key] = values[index]
        records.append(record)
    return records


def _record_date(record: dict[str, Any]) -> Date | None:
    value = record.get("time")
    if not value:
        return None
    try:
        return Date.fromisoformat(str(value))
    except ValueError:
        return None


def _record_for_date(records: list[dict[str, Any]], target_date: Date) -> dict[str, Any] | None:
    for record in records:
        if _record_date(record) == target_date:
            return record
    return None


def _recent_records_before(records: list[dict[str, Any]], target_date: Date, limit: int = 3) -> list[dict[str, Any]]:
    dated_records = [
        record
        for record in records
        if (record_date := _record_date(record)) is not None and record_date < target_date
    ]
    if dated_records:
        return dated_records[-limit:]
    return records[:-1][-limit:]


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _record_float(record: dict[str, Any], key: str) -> float | None:
    return _safe_float(record.get(key))


def _mean_recent(records: list[dict[str, Any]], key: str) -> float | None:
    values = [_record_float(record, key) for record in records]
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _feature_value(
    forecast_record: dict[str, Any],
    recent_records: list[dict[str, Any]],
    key: str,
    default: Any,
) -> Any:
    forecast_value = _record_float(forecast_record, key)
    if forecast_value is not None:
        return round(forecast_value, 2)

    recent_value = _mean_recent(recent_records, key)
    if recent_value is not None:
        return round(recent_value, 2)

    return default


@app.get("/monitoring/recent")
def monitoring_recent(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    entries = recent_predictions(limit=limit)
    return {"count": len(entries), "predictions": entries}
