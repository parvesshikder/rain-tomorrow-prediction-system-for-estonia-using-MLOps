import os
import time
import requests
import pandas as pd
from src.config import PARAMS

def fetch_weather_data():
    base_url = PARAMS["data"]["source_url"]
    start_date = PARAMS["data"]["start_date"]
    end_date = PARAMS["data"]["end_date"]
    timezone = PARAMS["data"]["timezone"]
    variables = PARAMS["data"]["daily_variables"]
    locations = PARAMS["data"]["locations"]
    raw_path = PARAMS["data"]["raw_path"]

    all_data = []

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
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = requests.get(base_url, params=params, timeout=15)
                
                # Check for rate limit explicitly
                if response.status_code == 429:
                    header_retry = int(response.headers.get("Retry-After", 0))
                    wait_time = max(30 * (attempt + 1), header_retry)
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
                wait_time = (attempt + 1) * 10
                print(f"Attempt {attempt+1} failed for {city}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        if not success:
            raise RuntimeError(f"Failed to fetch data for {city} after {max_attempts} attempts.")
            
        time.sleep(5.0)  # Sleep 5 seconds between locations to avoid triggering rate limits

    if not all_data:
        raise ValueError("No data was fetched.")

    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure raw output directory exists
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    combined_df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path}")

if __name__ == "__main__":
    fetch_weather_data()
