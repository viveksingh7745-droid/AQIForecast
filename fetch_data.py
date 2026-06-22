"""
fetch_data.py
Fetches historical AQI data from OpenAQ API and saves to CSV.
Run this first to collect your dataset.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# ── Config ──────────────────────────────────────────────────────────────────
CITY        = "Hyderabad"
COUNTRY     = "IN"
PARAMETER   = "pm25"          # pm25 | pm10 | no2 | o3 | co | so2
DAYS_BACK   = 90              # how many days of history to fetch
OUTPUT_FILE = "data/raw_aqi.csv"
# ────────────────────────────────────────────────────────────────────────────


def fetch_openaq(city: str, country: str, parameter: str, days_back: int) -> pd.DataFrame:
    """Fetch measurements from OpenAQ v2 API."""
    base_url = "https://api.openaq.org/v2/measurements"
    date_to   = datetime.utcnow()
    date_from = date_to - timedelta(days=days_back)

    all_records = []
    page = 1
    limit = 1000

    print(f"Fetching {parameter.upper()} data for {city}, {country} ...")

    while True:
        params = {
            "city":        city,
            "country":     country,
            "parameter":   parameter,
            "date_from":   date_from.isoformat() + "Z",
            "date_to":     date_to.isoformat() + "Z",
            "limit":       limit,
            "page":        page,
            "sort":        "asc",
            "order_by":    "datetime",
        }
        try:
            resp = requests.get(base_url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  API error on page {page}: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        for r in results:
            all_records.append({
                "datetime":  r["date"]["utc"],
                "value":     r["value"],
                "unit":      r["unit"],
                "location":  r["location"],
                "city":      r["city"],
                "country":   r["country"],
                "parameter": r["parameter"],
            })

        print(f"  Page {page}: fetched {len(results)} records (total so far: {len(all_records)})")

        if len(results) < limit:
            break
        page += 1
        time.sleep(0.5)   # be polite to the API

    if not all_records:
        print("No records found. Try a different city or parameter.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"Total records fetched: {len(df)}")
    return df


def use_sample_data() -> pd.DataFrame:
    """
    Generate synthetic AQI data for offline testing when the API
    is unavailable or returns no results.
    """
    import numpy as np
    print("Generating synthetic sample data for testing ...")
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=90 * 24, freq="h", tz="UTC")
    # Simulate realistic AQI patterns (diurnal + weekly + noise)
    hours  = dates.hour
    days   = dates.dayofweek
    trend  = 80 + 20 * np.sin(2 * np.pi * dates.dayofyear / 365)
    diurnal = 15 * np.sin(2 * np.pi * (hours - 6) / 24)
    weekly  = 10 * (days >= 5).astype(float) * -1   # cleaner on weekends
    noise   = np.random.normal(0, 10, len(dates))
    values  = np.clip(trend + diurnal + weekly + noise, 5, 400)

    df = pd.DataFrame({
        "datetime":  dates,
        "value":     values.round(1),
        "unit":      "µg/m³",
        "location":  "Sample Station",
        "city":      CITY,
        "country":   COUNTRY,
        "parameter": PARAMETER,
    })
    return df


def main():
    os.makedirs("data", exist_ok=True)

    df = fetch_openaq(CITY, COUNTRY, PARAMETER, DAYS_BACK)

    if df.empty:
        print("Falling back to synthetic sample data.")
        df = use_sample_data()

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} rows → {OUTPUT_FILE}")
    print(df.head())


if __name__ == "__main__":
    main()
