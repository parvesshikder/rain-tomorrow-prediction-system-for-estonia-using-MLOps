from fastapi.testclient import TestClient

from src import api


class DummyModel:
    def predict(self, frame):
        return [1]

    def predict_proba(self, frame):
        return [[0.25, 0.75]]


class DummyWeatherResponse:
    def raise_for_status(self):
        return None

    def json(self):
        today = api.Date.today()
        dates = [today - api.timedelta(days=2), today - api.timedelta(days=1), today, today + api.timedelta(days=1)]
        return {
            "current": {
                "temperature_2m": 8,
                "relative_humidity_2m": 82,
                "precipitation": 1.4,
                "rain": 1.2,
                "wind_speed_10m": 18,
                "wind_gusts_10m": 28,
                "pressure_msl": 1009,
            },
            "daily": {
                "time": [day.isoformat() for day in dates],
                "temperature_2m_max": [8, 9, 10, 12],
                "temperature_2m_min": [2, 3, 4, 4],
                "temperature_2m_mean": [5, 6, 7, 8],
                "relative_humidity_2m_mean": [76, 78, 80, 82],
                "precipitation_sum": [0.0, 0.4, 1.4, 2.2],
                "rain_sum": [0.0, 0.3, 1.2, 2.0],
                "wind_speed_10m_max": [14, 15, 16, 18],
                "wind_gusts_10m_max": [24, 25, 26, 28],
                "pressure_msl_mean": [1012, 1011, 1010, 1009],
            },
        }


def test_health_endpoint_reports_status(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: DummyModel())
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_endpoint_returns_prediction(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "load_model", lambda: DummyModel())
    monkeypatch.setitem(api.PARAMS["monitoring"], "prediction_log_path", str(tmp_path / "predictions.jsonl"))
    client = TestClient(api.app)

    response = client.post(
        "/predict",
        json={
            "city": "Tallinn",
            "temperature_2m_max": 12,
            "temperature_2m_min": 4,
            "temperature_2m_mean": 8,
            "relative_humidity_2m_mean": 82,
            "precipitation_sum": 1.4,
            "rain_sum": 1.2,
            "wind_speed_10m_max": 18,
            "wind_gusts_10m_max": 28,
            "pressure_msl_mean": 1009,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["city"] == "Tallinn"
    assert body["prediction"] == "Yes"
    assert body["rain_probability"] == 0.75
    assert body["logged"] is True


def test_monitoring_recent_returns_logged_predictions(monkeypatch, tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setitem(api.PARAMS["monitoring"], "prediction_log_path", str(log_path))
    client = TestClient(api.app)

    api.log_prediction(
        city="Tallinn",
        input_features={"city": "Tallinn"},
        prediction="Yes",
        rain_probability=0.75,
    )

    response = client.get("/monitoring/recent?limit=1")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["predictions"][0]["city"] == "Tallinn"


def test_browser_ui_loads():
    client = TestClient(api.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Estonia Rain Tomorrow Predictor" in response.text
    assert "Load Forecast Data" in response.text
    assert "Estimate Rain Chance" in response.text
    assert "No weather data loaded" in response.text
    assert "Load forecast data to enable prediction." in response.text
    assert "probability-number" in response.text
    assert "Estimated rain chance" in response.text


def test_weather_current_builds_prediction_input(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(kwargs["params"])
        return DummyWeatherResponse()

    monkeypatch.setattr(api.requests, "get", fake_get)
    client = TestClient(api.app)

    response = client.get("/weather/forecast-input?city=Tallinn")

    assert response.status_code == 200
    body = response.json()
    prediction_input = body["prediction_input"]
    tomorrow = api.Date.today() + api.timedelta(days=1)
    assert body["city"] == "Tallinn"
    assert body["forecast_date"] == tomorrow.isoformat()
    assert body["history_days"] == 3
    assert calls[0]["past_days"] == 3
    assert calls[0]["forecast_days"] == 2
    assert prediction_input["temperature_2m_max"] == 12
    assert prediction_input["temperature_2m_min"] == 4
    assert prediction_input["temperature_2m_mean"] == 8.0
    assert prediction_input["precipitation_sum"] == 2.2
    assert prediction_input["RainToday"] == 1
