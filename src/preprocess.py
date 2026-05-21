import os
import pandas as pd
from src.config import PARAMS

def preprocess_df(df, threshold, precip_col, numeric_features):
    """
    Cleans raw DataFrame, extracts calendar features, calculates targets,
    and imputes missing values.
    """
    df = df.copy()
    
    # 1. Parse date column
    df["date"] = pd.to_datetime(df["date"])
    
    # 2. Extract calendar features
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    
    # 3. Create RainToday Target (1 if precipitation_sum >= threshold, else 0)
    df["RainToday"] = (df[precip_col] >= threshold).astype(int)
    
    # 4. Create RainTomorrow Target (next day's RainToday value grouped by city)
    df["RainTomorrow"] = df.groupby("city")["RainToday"].shift(-1)
    
    # 5. Drop rows where RainTomorrow is NaN (the last day of history for each city)
    df.dropna(subset=["RainTomorrow"], inplace=True)
    df["RainTomorrow"] = df["RainTomorrow"].astype(int)
    
    # 6. Impute missing numeric values using the median
    for col in numeric_features:
        if col in df.columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Imputed missing values in column '{col}' with median: {median_val}")
                
    return df

def preprocess_data():
    raw_path = PARAMS["data"]["raw_path"]
    processed_path = PARAMS["data"]["processed_path"]
    threshold = PARAMS["target"]["precipitation_threshold_mm"]
    precip_col = PARAMS["target"]["precipitation_column"]
    numeric_features = PARAMS["features"]["numeric"]
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")
        
    df = pd.read_csv(raw_path)
    
    df_processed = preprocess_df(df, threshold, precip_col, numeric_features)
                
    # Ensure processed output directory exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_processed.to_csv(processed_path, index=False)
    print(f"Saved preprocessed data to {processed_path}")

if __name__ == "__main__":
    preprocess_data()
