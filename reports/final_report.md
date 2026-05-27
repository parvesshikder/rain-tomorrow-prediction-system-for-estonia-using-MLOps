# Final Report

## Project

Rain Tomorrow Prediction System for Estonia Using MLOps

## Goal

The project predicts whether it will rain tomorrow for selected Estonian cities using weather features such as temperature, humidity, precipitation, wind, pressure, location, and calendar fields.

## Data

The planned live data source is the Open-Meteo Historical Weather API. The project also supports deterministic sample data for local Docker and CI runs when the public API daily limit is reached.

The preprocessing step creates:

- `RainToday`
- `RainTomorrow`
- `month`
- `day_of_year`

## Model

The model is a scikit-learn Logistic Regression pipeline. It includes:

- median imputation for numeric features
- most-frequent imputation for categorical features
- standard scaling for numeric features
- one-hot encoding for city
- balanced class weights

The model is saved to:

```text
models/estonia_rain_model.joblib
```

## Experiment Tracking

Training logs parameters, evaluation metrics, and the model artifact to MLflow. The local MLflow backend is:

```text
mlruns/
```

The MLflow UI is available through Docker Compose on:

```text
http://localhost:5001
```

## Deployment

The project includes a FastAPI application in:

```text
src/api.py
```

Implemented endpoints:

- `GET /`
- `GET /health`
- `POST /predict`
- `GET /weather/forecast-input`
- `GET /weather/current`
- `GET /monitoring/recent`

The root endpoint provides a browser UI for loading recent weather plus tomorrow forecast inputs and viewing estimated rain chance.

## Monitoring

Every prediction is logged as JSON Lines to:

```text
logs/predictions.jsonl
```

The recent logs can be viewed through:

```text
GET /monitoring/recent
```

## Containerization

The project includes:

- `Dockerfile`
- `docker-compose.yml`

Docker Compose services:

- `rain-pipeline`
- `api`
- `mlflow`

## CI

GitHub Actions is configured in:

```text
.github/workflows/ci.yml
```

The workflow installs dependencies, runs the pipeline with sample data, runs tests, and validates the Docker build.

## Completion Status

Person 3 deployment and MLOps tasks are complete:

- FastAPI app added
- Required endpoints added
- Browser UI added
- Prediction logging added
- Dockerfile completed
- GitHub Actions CI added
- API tests added
- README updated
- Final report written
