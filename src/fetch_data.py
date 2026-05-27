import os
import time
import math
import random
import requests
import pandas as pd
from src.config import PARAMS


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def generate_sample_weather_data():
    """Create deterministic sample weather data for Docker/local demos."""
    start_date = os.getenv("SAMPLE_START_DATE", PARAMS["data"]["start_date"])
    end_date = os.getenv("SAMPLE_END_DATE", PARAMS["data"]["end_date"])
    locations = PARAMS["data"]["locations"]

    rows = []
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    for loc in locations:
        city = loc["city"]
        lat = loc["latitude"]
        lon = loc["longitude"]
        rng = random.Random(f"{city}-{start_date}-{end_date}")
        location_offset = (lat - 58.5) * -1.3 + (lon - 25.0) * 0.12

        for date in dates:
            day = date.timetuple().tm_yday
            seasonal = math.sin(2 * math.pi * (day - 105) / 365.25)
            temp_mean = 6.0 + 14.0 * seasonal + location_offset + rng.uniform(-3.0, 3.0)
            temp_max = temp_mean + rng.uniform(2.0, 7.0)
            temp_min = temp_mean - rng.uniform(2.0, 7.0)
            humidity = max(45.0, min(100.0, 78.0 - temp_mean * 0.45 + rng.uniform(-12.0, 12.0)))
            rain_chance = max(0.08, min(0.75, 0.22 + (humidity - 70.0) * 0.006 - seasonal * 0.06))

            if rng.random() < rain_chance:
                precipitation = max(0.1, rng.expovariate(0.45))
            else:
                precipitation = 0.0

            wind_speed = max(1.0, rng.gauss(16.0, 5.0))
            wind_gusts = wind_speed + rng.uniform(6.0, 18.0)
            pressure = 1013.0 + 7.0 * math.cos(2 * math.pi * day / 365.25) + rng.uniform(-10.0, 10.0)

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "temperature_2m_max": round(temp_max, 2),
                "temperature_2m_min": round(temp_min, 2),
                "temperature_2m_mean": round(temp_mean, 2),
                "relative_humidity_2m_mean": round(humidity, 2),
                "precipitation_sum": round(precipitation, 2),
                "rain_sum": round(precipitation * rng.uniform(0.75, 1.0), 2),
                "wind_speed_10m_max": round(wind_speed, 2),
                "wind_gusts_10m_max": round(wind_gusts, 2),
                "pressure_msl_mean": round(pressure, 2),
                "city": city,
                "latitude": lat,
                "longitude": lon,
            })

    return pd.DataFrame(rows)


def save_raw_data(df, raw_path):
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path}")


def save_sample_weather_data(raw_path, reason):
    print(f"{reason} Using deterministic sample weather data instead.")
    sample_df = generate_sample_weather_data()
    save_raw_data(sample_df, raw_path)


def fetch_weather_data():
    base_url = PARAMS["data"]["source_url"]
    start_date = PARAMS["data"]["start_date"]
    end_date = PARAMS["data"]["end_date"]
    timezone = PARAMS["data"]["timezone"]
    variables = PARAMS["data"]["daily_variables"]
    locations = PARAMS["data"]["locations"]
    raw_path = PARAMS["data"]["raw_path"]

    if _env_flag("USE_SAMPLE_DATA"):
        save_sample_weather_data(raw_path, "USE_SAMPLE_DATA is enabled.")
        return

    all_data = []
    max_attempts = _env_int("FETCH_MAX_ATTEMPTS", 5)
    rate_limit_backoff_seconds = _env_int("FETCH_RATE_LIMIT_BACKOFF_SECONDS", 30)
    error_backoff_seconds = _env_int("FETCH_ERROR_BACKOFF_SECONDS", 10)
    sleep_between_locations = _env_float("FETCH_SLEEP_BETWEEN_LOCATIONS_SECONDS", 5.0)

    for loc in locations:
        city = loc["city"]
        lat = loc["latitude"]
        lon = loc["longitude"]
        
        print(f"Fetching data for {city}...")
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(variables),
            "timezone": timezone
        }
        
        success = False
        for attempt in range(max_attempts):
            try:
                response = requests.get(base_url, params=params, timeout=15)
                
                # Check for rate limit explicitly
                if response.status_code == 429:
                    try:
                        header_retry = int(response.headers.get("Retry-After", 0))
                    except ValueError:
                        header_retry = 0
                    wait_time = max(rate_limit_backoff_seconds * (attempt + 1), header_retry)
                    print(f"Rate limited (429) for {city}. Retrying in {wait_time}s (Attempt {attempt+1}/{max_attempts})...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                
                if "daily" in data:
                    df = pd.DataFrame(data["daily"])
                    df.rename(columns={"time": "date"}, inplace=True)
                    df["city"] = city
                    df["latitude"] = lat
                    df["longitude"] = lon
                    all_data.append(df)
                    success = True
                    break
                else:
                    print(f"Warning: No daily data returned for {city}")
                    break
            except Exception as e:
                wait_time = (attempt + 1) * error_backoff_seconds
                print(f"Attempt {attempt+1} failed for {city}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        if not success:
            if _env_flag("FALLBACK_TO_SAMPLE_DATA"):
                save_sample_weather_data(raw_path, f"Failed to fetch live data for {city}.")
                return
            raise RuntimeError(f"Failed to fetch data for {city} after {max_attempts} attempts.")
            
        time.sleep(sleep_between_locations)  # Sleep between locations to avoid triggering rate limits

    if not all_data:
        raise ValueError("No data was fetched.")

    combined_df = pd.concat(all_data, ignore_index=True)
    save_raw_data(combined_df, raw_path)

if __name__ == "__main__":
    fetch_weather_data()
