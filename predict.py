"""
predict.py
Loads the trained model and generates AQI forecasts for the next N hours.
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import timedelta

MODEL_FILE     = "models/xgb_aqi_model.pkl"
SCALER_FILE    = "models/scaler.pkl"
PROCESSED_FILE = "data/processed_aqi.csv"
FORECAST_HOURS = 24

FEATURE_COLS = [
    "aqi_lag_1h", "aqi_lag_2h", "aqi_lag_3h",
    "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h",
    "aqi_roll_mean_3h", "aqi_roll_mean_6h",
    "aqi_roll_mean_12h", "aqi_roll_mean_24h",
    "aqi_roll_std_3h", "aqi_roll_std_6h",
    "aqi_roll_std_12h", "aqi_roll_std_24h",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
    "is_weekend",
]


def aqi_category(aqi: float) -> str:
    aqi = int(aqi)
    if aqi <= 50:   return "🟢 Good"
    if aqi <= 100:  return "🟡 Moderate"
    if aqi <= 150:  return "🟠 Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "🔴 Unhealthy"
    if aqi <= 300:  return "🟣 Very Unhealthy"
    return "🔴 Hazardous"


def health_advice(aqi: float) -> str:
    aqi = int(aqi)
    if aqi <= 50:
        return "Air quality is satisfactory. Enjoy outdoor activities."
    if aqi <= 100:
        return "Acceptable. Sensitive individuals should limit prolonged outdoor exertion."
    if aqi <= 150:
        return "Sensitive groups should reduce outdoor activity."
    if aqi <= 200:
        return "Everyone should reduce prolonged outdoor exertion."
    if aqi <= 300:
        return "Avoid outdoor activities. Wear N95 mask if going outside."
    return "Health emergency. Stay indoors with air purifier if possible."


def build_feature_row(dt: pd.Timestamp, history: list) -> dict:
    """
    Build a single feature row for a given datetime, using
    the most recent AQI history.
    history: list of recent AQI values (newest last), at least 24 items.
    """
    hist = np.array(history[-24:], dtype=float)  # last 24 hours

    lags = {
        "aqi_lag_1h":  hist[-1]  if len(hist) >= 1  else np.nan,
        "aqi_lag_2h":  hist[-2]  if len(hist) >= 2  else np.nan,
        "aqi_lag_3h":  hist[-3]  if len(hist) >= 3  else np.nan,
        "aqi_lag_6h":  hist[-6]  if len(hist) >= 6  else np.nan,
        "aqi_lag_12h": hist[-12] if len(hist) >= 12 else np.nan,
        "aqi_lag_24h": hist[-24] if len(hist) >= 24 else np.nan,
    }

    rolls = {}
    for w in [3, 6, 12, 24]:
        window = hist[-w:] if len(hist) >= w else hist
        rolls[f"aqi_roll_mean_{w}h"] = window.mean()
        rolls[f"aqi_roll_std_{w}h"]  = window.std() if len(window) > 1 else 0.0

    time_feats = {
        "hour_sin":   np.sin(2 * np.pi * dt.hour / 24),
        "hour_cos":   np.cos(2 * np.pi * dt.hour / 24),
        "dow_sin":    np.sin(2 * np.pi * dt.dayofweek / 7),
        "dow_cos":    np.cos(2 * np.pi * dt.dayofweek / 7),
        "month_sin":  np.sin(2 * np.pi * dt.month / 12),
        "month_cos":  np.cos(2 * np.pi * dt.month / 12),
        "is_weekend": int(dt.dayofweek >= 5),
    }

    return {**lags, **rolls, **time_feats}


def forecast(hours: int = FORECAST_HOURS) -> pd.DataFrame:
    # Load model & scaler
    with open(MODEL_FILE,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_FILE, "rb") as f: scaler = pickle.load(f)

    # Load recent data to seed the forecast
    df = pd.read_csv(PROCESSED_FILE, parse_dates=["datetime"])
    recent_aqi = df["aqi"].tolist()

    last_dt = df["datetime"].max()
    print(f"Last known data point: {last_dt}  (AQI={recent_aqi[-1]:.0f})")
    print(f"Generating {hours}-hour forecast ...\n")

    results = []
    history = list(recent_aqi)  # rolling history updated with each prediction

    for h in range(1, hours + 1):
        future_dt = last_dt + timedelta(hours=h)
        row = build_feature_row(future_dt, history)

        X = np.array([[row[c] for c in FEATURE_COLS]])
        X_s = scaler.transform(X)
        pred_aqi = float(model.predict(X_s)[0])
        pred_aqi = max(0, min(500, pred_aqi))   # clamp to valid range

        results.append({
            "datetime":      future_dt,
            "forecast_hour": h,
            "predicted_aqi": round(pred_aqi, 1),
            "category":      aqi_category(pred_aqi),
            "advice":        health_advice(pred_aqi),
        })

        # Feed prediction back into history for next step
        history.append(pred_aqi)

    return pd.DataFrame(results)


def main():
    forecast_df = forecast(FORECAST_HOURS)

    # Print summary table
    print("=" * 72)
    print(f"{'Hour':>5}  {'Date/Time':<20}  {'AQI':>6}  {'Category':<35}")
    print("-" * 72)
    for _, row in forecast_df.iterrows():
        dt_str = row["datetime"].strftime("%Y-%m-%d %H:%M")
        print(f"  +{row['forecast_hour']:02d}h  {dt_str:<20}  {row['predicted_aqi']:>6.1f}  {row['category']}")
    print("=" * 72)

    # Print min/max/avg summary
    aqi_vals = forecast_df["predicted_aqi"]
    print(f"\nForecast summary ({FORECAST_HOURS}h window):")
    print(f"  Min AQI : {aqi_vals.min():.1f}")
    print(f"  Max AQI : {aqi_vals.max():.1f}")
    print(f"  Avg AQI : {aqi_vals.mean():.1f}")
    worst_idx = aqi_vals.idxmax()
    print(f"  Worst at: {forecast_df.loc[worst_idx, 'datetime'].strftime('%Y-%m-%d %H:00')}")
    print(f"  Advice  : {forecast_df.loc[worst_idx, 'advice']}")

    # Save to CSV
    forecast_df.to_csv("outputs/forecast.csv", index=False)
    print("\nFull forecast saved → outputs/forecast.csv")

    return forecast_df


if __name__ == "__main__":
    main()
