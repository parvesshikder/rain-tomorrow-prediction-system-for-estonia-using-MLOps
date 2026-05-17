# Rain Tomorrow Prediction System for Estonia Using MLOps

## 1. Project Goal

Build a machine learning system that predicts whether it will rain tomorrow for a selected location in Estonia.

The project should not predict one single value for the whole country. Instead, the user will choose or provide an Estonian location, and the model will predict:

```text
RainTomorrow: Yes or No
```

This matches the proposal requirement of building a weather prediction model while making the project more locally relevant than using a generic dataset.

## 2. Final Project Scope

The project will use historical weather data for multiple Estonian cities and train one model that can predict rain tomorrow for different locations in Estonia.

Recommended cities:

```text
Tallinn
Tartu
Narva
Pärnu
Viljandi
Rakvere
Kuressaare
Võru
Haapsalu
Jõhvi
```

Each city should include:

```text
city
latitude
longitude
date
temperature
humidity
wind speed
wind gusts
pressure
precipitation
rain amount
```

## 3. Dataset Source

Use the Open-Meteo Historical Weather API:

```text
https://open-meteo.com/en/docs/historical-weather-api
```

Reason:

Open-Meteo provides historical weather observations for specific coordinates. It does not give us a ready-made machine learning model. We only use it to collect raw weather data. The dataset preparation, target creation, model training, evaluation, deployment, and monitoring are done by us.

Justification for report:

```text
We selected Estonia as the prediction area because it makes the project locally relevant and practical. Historical weather data is collected for multiple Estonian cities, allowing the model to learn weather patterns across the country. The API provides raw observations only; the RainTomorrow target, preprocessing, training, evaluation, deployment, and monitoring are implemented by us.
```

## 4. Target Variable

The API will not provide `RainTomorrow`, so create it manually.

For each city separately:

```text
RainToday = 1 if precipitation_sum >= 1.0 mm
RainToday = 0 otherwise

RainTomorrow = 1 if next day's precipitation_sum >= 1.0 mm
RainTomorrow = 0 otherwise
```

Important:

The target must be shifted within each city group. Do not shift the full dataset at once, because the last row for Tallinn could accidentally use the first row for Tartu.

## 5. Features

Use these model features:

```text
city
latitude
longitude
temperature_2m_max
temperature_2m_min
temperature_2m_mean
relative_humidity_2m_mean
precipitation_sum
rain_sum
wind_speed_10m_max
wind_gusts_10m_max
pressure_msl_mean
RainToday
month
day_of_year
```

Encoding:

```text
city -> OneHotEncoder
numeric features -> StandardScaler
missing values -> SimpleImputer
```

## 6. Recommended Project Structure

Create this structure:

```text
Mlops Project/
  data/
    raw/
    processed/
  models/
  logs/
  reports/
  src/
    fetch_data.py
    preprocess.py
    train.py
    evaluate.py
    api.py
    monitor.py
    config.py
  tests/
    test_preprocess.py
    test_api.py
  .github/
    workflows/
      ci.yml
  params.yaml
  dvc.yaml
  requirements.txt
  Dockerfile
  README.md
  PLAN.md
```

## 7. Step-by-Step Implementation Plan

### Step 1: Initialize the Project

Create folders:

```text
data/raw
data/processed
models
logs
reports
src
tests
.github/workflows
```

Create basic files:

```text
requirements.txt
params.yaml
README.md
Dockerfile
dvc.yaml
```

### Step 2: Define Configuration

Create `params.yaml`.

Include:

```text
city names
latitude and longitude for each city
start date
end date
weather variables
rain threshold
model settings
MLflow experiment name
file paths
```

Example date range:

```text
2015-01-01 to latest completed date
```

### Step 3: Install Dependencies

Add these to `requirements.txt`:

```text
pandas
numpy
scikit-learn
requests
joblib
pyyaml
mlflow
dvc
fastapi
uvicorn
pydantic
pytest
httpx
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 4: Fetch Historical Weather Data

Create `src/fetch_data.py`.

The script should:

```text
loop through all selected Estonian cities
call Open-Meteo API for each city
download daily weather data
add city, latitude, longitude columns
combine all cities into one dataframe
save to data/raw/estonia_weather.csv
```

Expected output:

```text
data/raw/estonia_weather.csv
```

This satisfies:

```text
Data collection
Dataset creation
Local raw data storage
```

### Step 5: Preprocess the Data

Create `src/preprocess.py`.

The script should:

```text
read data/raw/estonia_weather.csv
convert date column to datetime
create month
create day_of_year
create RainToday
create RainTomorrow by grouping by city
remove the final row for each city because tomorrow is unknown
handle missing values
save modeling dataset
```

Expected output:

```text
data/processed/estonia_weather_modeling.csv
```

This satisfies:

```text
Data preprocessing
Feature engineering
Target creation
```

### Step 6: Version Data with DVC

Initialize Git and DVC:

```bash
git init
dvc init
```

Track raw data:

```bash
dvc add data/raw/estonia_weather.csv
```

Track processed data through the DVC pipeline.

This satisfies:

```text
Data versioning
Reproducibility
```

### Step 7: Build the DVC Pipeline

Create `dvc.yaml` with stages:

```text
fetch_data
preprocess
train
evaluate
```

Pipeline flow:

```text
fetch_data -> preprocess -> train -> evaluate
```

Run:

```bash
dvc repro
```

This satisfies:

```text
Workflow orchestration
Reproducible ML pipeline
```

The proposal mentions Kubeflow. For this project, use DVC pipeline or scripts first. If required by the instructor, add Kubeflow later as an advanced extension.

### Step 8: Train the ML Model

Create `src/train.py`.

Use:

```text
Scikit-learn Logistic Regression
```

The training script should:

```text
read processed dataset
split train and test data
build preprocessing pipeline
encode city
scale numeric features
train Logistic Regression
save model to models/estonia_rain_model.joblib
```

Recommended split:

```text
chronological split
80% training
20% testing
```

This satisfies:

```text
Machine learning model development
Model training
```

### Step 9: Track Experiments with MLflow

Inside `src/train.py`, add MLflow logging.

Track:

```text
model type
rain threshold
features used
train/test split
accuracy
F1-score
precision
recall
ROC-AUC
model artifact
```

Run MLflow UI:

```bash
mlflow ui
```

This satisfies:

```text
Experiment tracking
MLflow usage
Model artifact tracking
```

### Step 10: Evaluate the Model

Create `src/evaluate.py`.

Calculate:

```text
Accuracy
F1-score
Precision
Recall
ROC-AUC
Confusion matrix
```

Save metrics to:

```text
reports/metrics.json
```

This satisfies:

```text
Model evaluation
Metrics reporting
```

### Step 11: Create FastAPI Deployment

Create `src/api.py`.

Endpoints:

```text
GET /health
POST /predict
GET /monitoring/recent
```

Prediction input should include:

```text
city
latitude
longitude
temperature values
humidity
wind speed
pressure
precipitation
rain today
month
day_of_year
```

Prediction output:

```json
{
  "prediction": "Yes",
  "rain_probability": 0.72
}
```

Run locally:

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

This satisfies:

```text
Local deployment
FastAPI deployment
Prediction service
```

### Step 12: Add Monitoring

Create `src/monitor.py`.

For every prediction, log:

```text
timestamp
city
input features
prediction
rain probability
```

Save logs to:

```text
logs/predictions.jsonl
```

This satisfies:

```text
Monitoring
Prediction logs
Input/output logging
```

Optional:

Later, compare predictions with actual next-day rain to estimate live accuracy.

### Step 13: Add Tests

Create tests for:

```text
RainTomorrow target creation
city grouping logic
preprocessing output columns
model prediction shape
FastAPI health endpoint
FastAPI predict endpoint
```

Run:

```bash
pytest
```

This satisfies:

```text
Validation
CI readiness
```

### Step 14: Containerize with Docker

Create `Dockerfile`.

The Docker image should:

```text
install dependencies
copy source code
copy trained model
start FastAPI with Uvicorn
```

Build:

```bash
docker build -t estonia-rain-api .
```

Run:

```bash
docker run -p 8000:8000 estonia-rain-api
```

This satisfies:

```text
Docker containerization
Local deployment
```

### Step 15: Add GitHub Actions CI

Create:

```text
.github/workflows/ci.yml
```

CI should:

```text
checkout code
install Python
install dependencies
run tests
validate DVC pipeline
build Docker image
```

This satisfies:

```text
Continuous Integration
Pipeline validation
Docker build validation
```

### Step 16: Write README

The README should include:

```text
project overview
why Estonia was selected
data source
how RainTomorrow is created
tools used
setup instructions
pipeline commands
MLflow instructions
API examples
Docker commands
monitoring explanation
results and metrics
```

This satisfies:

```text
Documentation
Reproducibility
Project presentation
```

### Step 17: Prepare Final Report or Presentation

Include:

```text
problem statement
dataset source and justification
architecture diagram
pipeline flow
model choice
evaluation metrics
MLOps tools
API demo
monitoring logs
limitations
future improvements
```

## 8. Mapping to Proposal Requirements

| Proposal Requirement | Project Implementation |
| --- | --- |
| Machine learning model development | Scikit-learn Logistic Regression |
| Data versioning | DVC tracks raw and processed data |
| Experiment tracking | MLflow logs parameters, metrics, and model |
| MLflow usage | Local MLflow experiment tracking |
| Docker containerization | Dockerfile for FastAPI service |
| Continuous Integration | GitHub Actions workflow |
| Workflow orchestration | DVC pipeline with fetch, preprocess, train, evaluate |
| Local deployment | FastAPI app running locally |
| Monitoring | Prediction logs in JSONL format |
| Weather prediction | RainTomorrow prediction for selected Estonian location |

## 9. Recommended Work Order

Complete the project in this exact order:

```text
1. Create folder structure
2. Create requirements.txt
3. Create params.yaml
4. Write data fetching script
5. Download Estonia weather dataset
6. Write preprocessing script
7. Create RainTomorrow target
8. Train Logistic Regression model
9. Evaluate model
10. Add MLflow tracking
11. Add DVC data versioning
12. Add DVC pipeline
13. Add FastAPI app
14. Add monitoring logs
15. Add tests
16. Add Dockerfile
17. Add GitHub Actions CI
18. Write README
19. Run full project from scratch
20. Prepare final report/demo
```

## 10. Minimum Final Deliverables

The completed project should contain:

```text
working dataset pipeline
processed Estonia weather dataset
trained rain prediction model
MLflow experiment logs
DVC pipeline
FastAPI prediction API
prediction monitoring logs
Docker image
GitHub Actions CI workflow
README documentation
final metrics report
```

## 11. Future Improvements

Possible improvements after the main project works:

```text
try Random Forest or XGBoost
add more Estonian locations
use more weather features
compare city-specific models vs one Estonia model
add live weather input from forecast API
add dashboard for monitoring
deploy to cloud
convert DVC pipeline to Kubeflow pipeline
```
