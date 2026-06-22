"""
preprocess.py
Cleans raw AQI CSV and engineers features for ML training.
"""

import pandas as pd
import numpy as np
import os

RAW_FILE       = "data/raw_aqi.csv"
PROCESSED_FILE = "data/processed_aqi.csv"


def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw CSV, drop duplicates, handle missing values."""
    print("Loading raw data ...")
    df = pd.read_csv(path, parse_dates=["datetime"])

    # Remove negative / unrealistic readings
    df = df[df["value"] > 0].copy()
    df = df[df["value"] < 1000].copy()

    # Keep only the AQI value column + datetime
    df = df[["datetime", "value"]].rename(columns={"value": "pm25"})

    # Sort and deduplicate
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    # Resample to hourly frequency, interpolate small gaps (≤3 hrs)
    df = df.set_index("datetime")
    df = df.resample("h").mean()
    df["pm25"] = df["pm25"].interpolate(method="time", limit=3)
    df = df.dropna().reset_index()

    print(f"Clean records: {len(df)}  |  Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    return df


def compute_aqi_us(pm25: float) -> int:
    """
    Convert PM2.5 (µg/m³) to US EPA AQI.
    Breakpoints: https://www.airnow.gov/aqi/aqi-basics/
    """
    breakpoints = [
        (0.0,   12.0,    0,  50),
        (12.1,  35.4,   51, 100),
        (35.5,  55.4,  101, 150),
        (55.5, 150.4,  151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo)
    return 500 if pm25 > 500 else 0


def aqi_category(aqi: int) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and datetime features."""
    print("Engineering features ...")
    df = df.copy()

    # --- AQI conversion ---
    df["aqi"] = df["pm25"].apply(compute_aqi_us)
    df["category"] = df["aqi"].apply(aqi_category)

    # --- Lag features (past AQI values) ---
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}h"] = df["aqi"].shift(lag)

    # --- Rolling statistics ---
    for window in [3, 6, 12, 24]:
        df[f"aqi_roll_mean_{window}h"] = df["aqi"].shift(1).rolling(window).mean()
        df[f"aqi_roll_std_{window}h"]  = df["aqi"].shift(1).rolling(window).std()

    # --- Datetime features ---
    df["hour"]        = df["datetime"].dt.hour
    df["dayofweek"]   = df["datetime"].dt.dayofweek
    df["month"]       = df["datetime"].dt.month
    df["is_weekend"]  = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]     = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

    # Drop rows with NaN (from lags/rolling)
    df = df.dropna().reset_index(drop=True)
    print(f"Feature matrix shape: {df.shape}")
    return df


def main():
    os.makedirs("data", exist_ok=True)
    df = load_and_clean(RAW_FILE)
    df = engineer_features(df)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"\nSaved processed data → {PROCESSED_FILE}")
    print(df[["datetime", "pm25", "aqi", "category"]].tail())


if __name__ == "__main__":
    main()
