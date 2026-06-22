"""
run_pipeline.py
Runs the entire AQI Forecast pipeline in one command:
  1. Fetch data  →  2. Preprocess  →  3. Train  →  4. Predict  →  5. Visualize
"""

import subprocess
import sys
import os

STEPS = [
    ("fetch_data.py",  "Step 1/5 — Fetching data"),
    ("preprocess.py",  "Step 2/5 — Preprocessing & feature engineering"),
    ("train_model.py", "Step 3/5 — Training XGBoost model"),
    ("predict.py",     "Step 4/5 — Generating 24-hour forecast"),
    ("visualize.py",   "Step 5/5 — Creating visualizations"),
]

def banner(msg: str):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)

def run_step(script: str, label: str) -> bool:
    banner(label)
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n❌  {script} failed with exit code {result.returncode}")
        return False
    return True

def main():
    banner("AQI Forecast Pipeline — Starting")
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    for script, label in STEPS:
        if not run_step(script, label):
            sys.exit(1)

    banner("Pipeline complete!")
    print("\nOutputs:")
    print("  data/raw_aqi.csv              ← raw fetched data")
    print("  data/processed_aqi.csv        ← cleaned + features")
    print("  data/test_predictions.csv     ← model test results")
    print("  models/xgb_aqi_model.pkl      ← trained XGBoost model")
    print("  models/metrics.json           ← evaluation metrics")
    print("  outputs/forecast.csv          ← 24-hour AQI forecast")
    print("  outputs/1_eda_overview.png    ← EDA charts")
    print("  outputs/2_actual_vs_predicted.png")
    print("  outputs/3_forecast.png        ← forecast chart")
    print("  outputs/4_category_distribution.png")

if __name__ == "__main__":
    main()
