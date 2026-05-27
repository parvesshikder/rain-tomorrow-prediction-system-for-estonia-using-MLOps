# Weather Prediction System with MLOps Pipeline Flow

## Project Title

Rain Tomorrow Prediction System for Estonia Using MLOps

## Project Description

The goal of this project is to build a machine learning application that predicts whether it will rain tomorrow for a selected location in Estonia.

```text
Rain Tomorrow: Yes or No
```

The focus of the project is not only to build a machine learning model, but to create a complete and reproducible MLOps workflow using the tools from the course.

## Project Scope

The system will predict rain tomorrow for selected Estonian locations using historical weather observations. The raw weather data will come from a public weather API, and the machine learning model will be trained as part of this project.

## Problem Statement

The system will predict:

```text
RainTomorrow: Yes or No
```

Using historical weather data such as:

```text
temperature
humidity
wind speed
pressure
precipitation
rain amount
location
date-based features
```

## Dataset Plan

The project will use public historical weather data for Estonia.

Planned data source:

```text
Open-Meteo Historical Weather API
```

Open-Meteo provides public historical weather observations. The `RainTomorrow` target will be created during preprocessing from the next day's precipitation value.

The dataset will include multiple Estonian locations.

## Target Variable

The target column will be created manually:

```text
RainToday = 1 if today's precipitation_sum >= 1.0 mm
RainToday = 0 otherwise

RainTomorrow = 1 if tomorrow's precipitation_sum >= 1.0 mm
RainTomorrow = 0 otherwise
```

This means the API provides raw data only. The target creation, preprocessing, model training, evaluation, and deployment are done in this project.

## Project Objectives

The project objectives are:

```text
1. Machine learning model development
2. Data versioning
3. Experiment tracking
4. MLflow usage
5. Docker containerization
6. Continuous Integration
7. Workflow orchestration
8. Local deployment
9. Monitoring
```

## Machine Learning Model

Planned model:

```text
Logistic Regression
```

Library:

```text
Scikit-learn
```

Evaluation metrics:

```text
Accuracy
F1-score
Precision
Recall
ROC-AUC
```

## MLOps Components

The project will include the following MLOps components:

| Component | Planned Implementation |
| --- | --- |
| Data management and versioning | DVC |
| Experiment tracking | MLflow |
| Model artifact storage | MLflow and `models/` folder |
| Containerization | Docker |
| Continuous Integration | GitHub Actions |
| Workflow orchestration | DVC pipeline / Python scripts |
| Local deployment | FastAPI |
| Monitoring | Prediction logs |

## System Workflow

The planned system workflow is:

```text
1. Historical Estonia weather data
2. DVC data versioning
3. Data preprocessing
4. Model training
5. MLflow experiment tracking
6. Model evaluation
7. Docker container
8. GitHub Actions CI
9. FastAPI deployment
10. Monitoring logs
```

## Technologies Used

| Task | Tool |
| --- | --- |
| Programming | Python |
| ML Library | Scikit-learn |
| Tracking | MLflow |
| Versioning | DVC |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | FastAPI |
| Orchestration | DVC pipeline / scripts |
| Repository | GitHub |



## Expected Outcome

At the end of the full project:

```text
1. A working Estonia rain prediction model will be created.
2. The system will be reproducible and deployable.
3. The project will simulate a production ML workflow.
```

## How to Run

Build and run the full pipeline with Docker Compose:

```bash
docker compose run --rm --build rain-pipeline
```

The pipeline creates:

```text
data/raw/estonia_weather.csv
data/processed/estonia_weather_modeling.csv
models/estonia_rain_model.joblib
reports/metrics.json
mlruns/
```

By default, Docker Compose uses deterministic sample data so the project can run even when the Open-Meteo daily API limit is reached. To force live Open-Meteo data:

```bash
USE_SAMPLE_DATA=0 FALLBACK_TO_SAMPLE_DATA=0 docker compose run --rm --build rain-pipeline
```

## FastAPI App

Start the API:

```bash
docker compose up --build api
```

Open the browser UI:

```text
http://localhost:8000
```

API endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Browser UI for predictions |
| `/health` | GET | Service and model status |
| `/predict` | POST | Rain-tomorrow prediction |
| `/weather/forecast-input` | GET | Recent weather plus tomorrow forecast inputs for a city |
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

Predictions are logged to:

```text
logs/predictions.jsonl
```

## MLflow

Run the MLflow UI:

```bash
docker compose up -d mlflow
```

Open:

```text
http://localhost:5001
```

## Testing

Run the pipeline first, then tests:

```bash
docker compose run --rm --build rain-pipeline
docker compose run --rm api python -m pytest
```

## CI

GitHub Actions is configured in:

```text
.github/workflows/ci.yml
```

The CI workflow installs dependencies, runs the pipeline with sample data, runs `pytest`, and validates the Docker build.
