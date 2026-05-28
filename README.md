# Rain Tomorrow Prediction System for Estonia Using MLOps

An end-to-end MLOps project that predicts whether it is likely to rain tomorrow for selected Estonian cities. The project covers data collection, preprocessing, target creation, model training, experiment tracking, evaluation, API deployment, Docker containerization, monitoring, and CI.

Team:

- Debotush Dam
- Calvin Saragih
- Md Parves Shikder

## Project Overview

The application estimates tomorrow's rain chance for a selected Estonian city. It uses weather features such as temperature, humidity, precipitation, rain amount, wind speed, wind gusts, pressure, location, and calendar fields.

Target output:

```text
RainTomorrow: Yes or No
```

This is an MLOps demonstration project. The model gives an estimated rain chance, not a professional weather forecast.

## Key Features

- Estonia-focused weather prediction for multiple cities
- Open-Meteo historical weather data pipeline
- Deterministic sample data mode for reliable local and CI runs
- `RainToday` and `RainTomorrow` target creation during preprocessing
- Scikit-learn Logistic Regression model
- MLflow experiment tracking and Automated Model Registry promotion
- DVC pipeline orchestration
- FastAPI prediction service with browser frontend
- Forecast input loader using recent weather plus tomorrow forecast data
- Prediction logging for monitoring
- Live API monitoring with Prometheus and Grafana
- Docker Compose services for pipeline, API, MLflow, Prometheus, and Grafana
- GitHub Actions CI workflow

## Tech Stack

| Area | Tool |
| --- | --- |
| Language | Python |
| ML | scikit-learn |
| Data pipeline | pandas, requests |
| Experiment tracking | MLflow |
| Pipeline orchestration | DVC |
| API | FastAPI, Uvicorn |
| Containerization | Docker, Docker Compose |
| Monitoring | Prometheus, Grafana |
| Testing | pytest |
| CI | GitHub Actions |

## Project Structure

```text
.
|-- data/
|   |-- raw/
|   `-- processed/
|-- logs/
|-- models/
|-- reports/
|   |-- final_report.md
|   `-- metrics.json
|-- src/
|   |-- api.py
|   |-- config.py
|   |-- evaluate.py
|   |-- fetch_data.py
|   |-- monitor.py
|   |-- preprocess.py
|   `-- train.py
|-- tests/
|-- .github/workflows/ci.yml
|-- docker-compose.yml
|-- Dockerfile
|-- dvc.yaml
|-- params.yaml
`-- requirements.txt
```

Generated files such as raw data, processed data, models, MLflow runs, prediction logs, and local presentation outputs are ignored by git.

## Data and Target

The project uses daily weather observations for selected Estonian cities. The planned live source is:

```text
Open-Meteo Historical Weather API
```

The target is created inside the project:

```text
RainToday = 1 if today's precipitation_sum >= 1.0 mm
RainToday = 0 otherwise

RainTomorrow = 1 if the next day's precipitation_sum >= 1.0 mm
RainTomorrow = 0 otherwise
```

Important detail: `RainTomorrow` is shifted within each city group, so one city's last row is not accidentally matched with another city's first row.

## Model

The model is a scikit-learn pipeline:

- numeric imputation with median
- categorical imputation with most frequent value
- numeric scaling with `StandardScaler`
- city encoding with `OneHotEncoder`
- Logistic Regression classifier
- chronological train/test split

Current sample-data metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.5393 |
| F1 score | 0.6017 |
| Precision | 0.7658 |
| Recall | 0.5393 |
| ROC-AUC | 0.5755 |

These metrics are saved in:

```text
reports/metrics.json
```

## Correct Run Order

Use this order after cloning the project:

1. Run the training pipeline first.
2. Start the API and MLflow services.
3. Open the browser UI and make predictions.

The API needs the trained model file from the pipeline:

```text
models/estonia_rain_model.joblib
```

## Run with Docker Compose

Clone and enter the project:

```bash
git clone git@github.com:parvesshikder/rain-tomorrow-prediction-system-for-estonia-using-MLOps.git
cd rain-tomorrow-prediction-system-for-estonia-using-MLOps
```

Build and run the full ML pipeline:

```bash
docker compose run --rm --build rain-pipeline
```

By default, the pipeline uses deterministic sample data:

```text
USE_SAMPLE_DATA=1
```

This is intentional. It makes the project run reliably even if the Open-Meteo historical API is rate-limited.

The pipeline creates:

```text
data/raw/estonia_weather.csv
data/processed/estonia_weather_modeling.csv
models/estonia_rain_model.joblib
reports/metrics.json
mlruns/
```

Start the API, MLflow, Prometheus, and Grafana:

```bash
docker compose up -d
```

Open the frontend:

```text
http://localhost:8000
```

Open MLflow (Experiment Tracking & Model Registry):

```text
http://localhost:5001
```

Open Grafana (API Monitoring):

```text
http://localhost:3001
```

Stop services:

```bash
docker compose down
```

## Run with Live Open-Meteo Data

The default sample mode is recommended for demos and CI. To try live historical data:

```bash
USE_SAMPLE_DATA=0 docker compose run --rm --build rain-pipeline
```

If you want the command to fail instead of falling back to sample data when the API is rate-limited:

```bash
USE_SAMPLE_DATA=0 FALLBACK_TO_SAMPLE_DATA=0 docker compose run --rm --build rain-pipeline
```

Open-Meteo can return `429 Too Many Requests`. If that happens, wait and try again, or use the default sample data mode.

## Port Conflicts

If port `8000` is already in use, run the API on another host port:

```bash
API_HOST_PORT=8001 docker compose up -d api
```

Then open:

```text
http://localhost:8001
```

If port `5001` is already in use, run MLflow on another host port:

```bash
MLFLOW_HOST_PORT=5002 docker compose up -d mlflow
```

Then open:

```text
http://localhost:5002
```

## API Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Browser frontend |
| `/health` | GET | Service and model status |
| `/predict` | POST | Rain-tomorrow prediction |
| `/weather/forecast-input` | GET | Recent weather plus tomorrow forecast inputs |
| `/weather/current` | GET | Backward-compatible alias for forecast inputs |
| `/monitoring/recent` | GET | Recent prediction logs |

Example prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Tallinn",
    "temperature_2m_max": 12,
    "temperature_2m_min": 4,
    "temperature_2m_mean": 8,
    "relative_humidity_2m_mean": 82,
    "precipitation_sum": 1.4,
    "rain_sum": 1.2,
    "wind_speed_10m_max": 18,
    "wind_gusts_10m_max": 28,
    "pressure_msl_mean": 1009
  }'
```

Example forecast input request:

```bash
curl "http://localhost:8000/weather/forecast-input?city=Tallinn"
```

## Browser Frontend

The frontend is served by FastAPI:

```text
http://localhost:8000
```

Workflow:

1. Select a city.
2. Click `Load Forecast Data`.
3. Review the weather inputs.
4. Click `Estimate Rain Chance`.
5. View the estimated rain probability and prediction sentence.

The frontend uses `/weather/forecast-input`, which requests recent weather and tomorrow forecast data from Open-Meteo. Internet access is required for that button.

## Monitoring

Each prediction is logged to:

```text
logs/predictions.jsonl
```

View recent prediction logs:

```bash
curl "http://localhost:8000/monitoring/recent?limit=10"
```

## DVC Pipeline

The DVC workflow is defined in:

```text
dvc.yaml
```

Stages:

```text
fetch_data -> preprocess -> train -> evaluate
```

Run DVC through the Docker image:

```bash
docker compose run --rm -e USE_SAMPLE_DATA=1 api dvc repro
```

The simpler project command is usually enough:

```bash
docker compose run --rm --build rain-pipeline
```

## Testing

Run tests after the pipeline has created the model and metrics:

```bash
docker compose run --rm --build rain-pipeline
docker compose run --rm api python -m pytest -q
```

Expected current result:

```text
24 passed
```

## CI

GitHub Actions is configured in:

```text
.github/workflows/ci.yml
```

The workflow:

1. Installs dependencies.
2. Runs the DVC pipeline with sample data.
3. Runs tests.
4. Builds the Docker image.

## Notes and Limitations

- The project is complete as an MLOps demonstration.
- The model's current predictive accuracy is modest.
- The frontend wording uses estimated rain chance because this is not a professional meteorological forecast.
- The default sample data mode is used to make local runs and CI stable.
- Live API data can be rate-limited by Open-Meteo.

## Final Deliverables

- Reproducible ML pipeline
- Processed Estonia weather dataset
- Trained rain prediction model
- MLflow experiment logs and Model Registry
- DVC pipeline
- FastAPI prediction API and browser UI
- Prometheus & Grafana Monitoring Dashboards
- Docker Compose deployment
- GitHub Actions CI
- Final metrics report
