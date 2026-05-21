import pandas as pd
import numpy as np
import pytest
from src.preprocess import preprocess_df

def test_rain_today_calculation():
    # Verify that precipitation >= threshold results in RainToday=1, and < threshold is 0
    df = pd.DataFrame({
        "date": ["2026-05-20", "2026-05-21", "2026-05-22"],
        "precipitation_sum": [1.5, 0.5, 1.0],
        "city": ["Tallinn", "Tallinn", "Tallinn"],
        "latitude": [59.437, 59.437, 59.437],
        "longitude": [24.753, 24.753, 24.753]
    })
    
    threshold = 1.0
    precip_col = "precipitation_sum"
    numeric_features = ["latitude", "longitude", "precipitation_sum", "RainToday"]
    
    processed = preprocess_df(df, threshold, precip_col, numeric_features)
    
    # After preprocessing, the last day is dropped because RainTomorrow is NaN.
    # So we check the first two rows.
    assert len(processed) == 2
    assert processed["RainToday"].iloc[0] == 1  # 1.5 >= 1.0
    assert processed["RainToday"].iloc[1] == 0  # 0.5 < 1.0

def test_rain_tomorrow_shift_by_city():
    # Verify shift grouped by city (leak prevention)
    df = pd.DataFrame({
        "date": ["2026-05-20", "2026-05-21", "2026-05-20", "2026-05-21"],
        "precipitation_sum": [1.5, 0.0, 0.0, 2.0],
        "city": ["Tallinn", "Tallinn", "Tartu", "Tartu"],
        "latitude": [59.437, 59.437, 58.378, 58.378],
        "longitude": [24.753, 24.753, 26.729, 26.729]
    })
    
    threshold = 1.0
    precip_col = "precipitation_sum"
    numeric_features = ["latitude", "longitude", "precipitation_sum", "RainToday"]
    
    processed = preprocess_df(df, threshold, precip_col, numeric_features)
    
    # We should have 2 rows left: the first day for Tallinn and first day for Tartu.
    # The last days of each city must be dropped because their RainTomorrow target is NaN.
    assert len(processed) == 2
    
    # Check Tallinn's tomorrow value (which should be Tallinn's second day RainToday = 0)
    tallinn_row = processed[processed["city"] == "Tallinn"]
    assert len(tallinn_row) == 1
    assert tallinn_row["RainTomorrow"].iloc[0] == 0
    
    # Check Tartu's tomorrow value (which should be Tartu's second day RainToday = 1)
    tartu_row = processed[processed["city"] == "Tartu"]
    assert len(tartu_row) == 1
    assert tartu_row["RainTomorrow"].iloc[0] == 1

def test_calendar_features_extraction():
    # 2026-05-20 should yield month=5, day_of_year=140
    # 2026-05-21 should yield month=5, day_of_year=141
    df = pd.DataFrame({
        "date": ["2026-05-20", "2026-05-21"],
        "precipitation_sum": [0.0, 0.0],
        "city": ["Tallinn", "Tallinn"],
        "latitude": [59.437, 59.437],
        "longitude": [24.753, 24.753]
    })
    
    threshold = 1.0
    precip_col = "precipitation_sum"
    numeric_features = ["latitude", "longitude", "precipitation_sum", "RainToday"]
    
    processed = preprocess_df(df, threshold, precip_col, numeric_features)
    
    assert processed["month"].iloc[0] == 5
    assert processed["day_of_year"].iloc[0] == 140

def test_missing_imputation():
    # Verify imputation fills NaN values using the median of the column
    # After dropping the last row, the remaining non-NaN values are 10.0 and 20.0, yielding a median of 15.0.
    df = pd.DataFrame({
        "date": ["2026-05-20", "2026-05-21", "2026-05-22", "2026-05-23"],
        "precipitation_sum": [0.0, 0.0, 0.0, 0.0],
        "temp": [10.0, np.nan, 20.0, 30.0],
        "city": ["Tallinn", "Tallinn", "Tallinn", "Tallinn"],
        "latitude": [59.437, 59.437, 59.437, 59.437],
        "longitude": [24.753, 24.753, 24.753, 24.753]
    })
    
    threshold = 1.0
    precip_col = "precipitation_sum"
    numeric_features = ["latitude", "longitude", "precipitation_sum", "RainToday", "temp"]
    
    processed = preprocess_df(df, threshold, precip_col, numeric_features)
    
    # We should have 3 rows remaining (the last is dropped)
    assert len(processed) == 3
    assert processed["temp"].iloc[0] == 10.0
    assert processed["temp"].iloc[1] == 15.0  # NaN imputed with median [10.0, 20.0] -> 15.0
    assert processed["temp"].iloc[2] == 20.0

