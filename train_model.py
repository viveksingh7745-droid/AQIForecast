"""
train_model.py
Trains an XGBoost model to forecast AQI 24 hours ahead.
Saves the trained model and evaluation metrics.
"""

import pandas as pd
import numpy as np
import pickle
import os
import json

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

PROCESSED_FILE = "data/processed_aqi.csv"
MODEL_FILE     = "models/xgb_aqi_model.pkl"
SCALER_FILE    = "models/scaler.pkl"
METRICS_FILE   = "models/metrics.json"
FORECAST_HOURS = 24   # predict this many hours ahead

# Features used for training (no datetime, no target leakage)
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
TARGET_COL = "aqi"


def load_data(path: str):
    df = pd.read_csv(path, parse_dates=["datetime"])
    print(f"Loaded {len(df)} rows from {path}")

    # Create target: AQI value N hours in the future
    df["target"] = df[TARGET_COL].shift(-FORECAST_HOURS)
    df = df.dropna(subset=["target"] + FEATURE_COLS).reset_index(drop=True)
    return df


def evaluate(y_true, y_pred, split_name: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # Category accuracy
    cats_true = pd.Series(y_true).apply(cat_from_aqi)
    cats_pred = pd.Series(y_pred).apply(cat_from_aqi)
    cat_acc = (cats_true.values == cats_pred.values).mean() * 100

    print(f"\n  [{split_name}]")
    print(f"    MAE            : {mae:.2f} AQI points")
    print(f"    RMSE           : {rmse:.2f} AQI points")
    print(f"    R²             : {r2:.4f}")
    print(f"    Category acc.  : {cat_acc:.1f}%")
    return {"mae": round(mae, 2), "rmse": round(rmse, 2),
            "r2": round(r2, 4), "category_accuracy": round(cat_acc, 1)}


def cat_from_aqi(aqi):
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "USG"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"


def main():
    os.makedirs("models", exist_ok=True)

    df = load_data(PROCESSED_FILE)
    X  = df[FEATURE_COLS].values
    y  = df["target"].values

    # --- Time-series train/test split (last 20% as test) ---
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

    # --- Scale features ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # --- Cross-validation on training set ---
    print("\nRunning 5-fold time-series cross-validation ...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_s), 1):
        m = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )
        m.fit(X_train_s[tr_idx], y_train[tr_idx],
              eval_set=[(X_train_s[val_idx], y_train[val_idx])],
              verbose=False)
        preds = m.predict(X_train_s[val_idx])
        mae = mean_absolute_error(y_train[val_idx], preds)
        cv_maes.append(mae)
        print(f"  Fold {fold} MAE: {mae:.2f}")
    print(f"  Mean CV MAE: {np.mean(cv_maes):.2f} ± {np.std(cv_maes):.2f}")

    # --- Final model on full training set ---
    print("\nTraining final model on full training split ...")
    model = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train_s, y_train, verbose=False)

    # --- Evaluate ---
    train_preds = model.predict(X_train_s)
    test_preds  = model.predict(X_test_s)
    print("\nEvaluation Results:")
    train_metrics = evaluate(y_train, train_preds, "Train")
    test_metrics  = evaluate(y_test,  test_preds,  "Test")

    # --- Feature importance (top 10) ---
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print("\nTop 10 feature importances:")
    print(importances.sort_values(ascending=False).head(10).round(4).to_string())

    # --- Save model, scaler, metrics ---
    with open(MODEL_FILE,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_FILE, "wb") as f: pickle.dump(scaler, f)

    metrics = {
        "forecast_hours": FORECAST_HOURS,
        "train_rows": len(X_train),
        "test_rows":  len(X_test),
        "cv_mae_mean": round(float(np.mean(cv_maes)), 2),
        "cv_mae_std":  round(float(np.std(cv_maes)), 2),
        "train": train_metrics,
        "test":  test_metrics,
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved → {MODEL_FILE}")
    print(f"Scaler saved → {SCALER_FILE}")
    print(f"Metrics saved → {METRICS_FILE}")

    # Save test predictions for plotting
    test_df = df.iloc[split:].copy()
    test_df["predicted_aqi"] = test_preds
    test_df[["datetime", "aqi", "predicted_aqi"]].to_csv(
        "data/test_predictions.csv", index=False
    )
    print("Test predictions saved → data/test_predictions.csv")


if __name__ == "__main__":
    main()
