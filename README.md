# AQI Forecast — CodTech Internship Project. Intership ID-CITS3857

Predicts Air Quality Index (AQI) 24 hours ahead using XGBoost and real-time
data from the OpenAQ API.

## Project Structure

```
aqi_forecast/
├── fetch_data.py       ← downloads AQI data from OpenAQ API
├── preprocess.py       ← cleans data, computes AQI, engineers features
├── train_model.py      ← trains XGBoost model with time-series CV
├── predict.py          ← generates 24-hour rolling forecast
├── visualize.py        ← creates EDA, evaluation, and forecast plots
├── run_pipeline.py     ← runs all 5 steps in sequence
├── requirements.txt
├── data/               ← generated CSVs
├── models/             ← saved model + scaler + metrics
└── outputs/            ← forecast CSV + PNG charts
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (recommended)
python run_pipeline.py

# Or run each step manually:
python fetch_data.py      # fetch data
python preprocess.py      # clean + feature engineering
python train_model.py     # train model
python predict.py         # generate forecast
python visualize.py       # create plots
```

## How It Works

1. **Data collection** — fetches hourly PM2.5 readings from OpenAQ for your
   configured city (default: Hyderabad, IN). Falls back to synthetic data if
   API returns no results.

2. **Preprocessing** — resamples to hourly, interpolates small gaps, converts
   PM2.5 to US EPA AQI using official breakpoints.

3. **Feature engineering** — creates lag features (1h, 2h, 3h, 6h, 12h, 24h),
   rolling mean/std (3h, 6h, 12h, 24h), cyclical time encodings (sin/cos for
   hour, day-of-week, month).

4. **Model** — XGBoost regressor, trained on 80% of data, evaluated on the
   final 20% using 5-fold time-series cross-validation.

5. **Prediction** — iterative 24-step-ahead forecast using the model's own
   predictions as input for subsequent steps.

## Configuration

Edit the top of `fetch_data.py` to change city, country, or pollutant:

```python
CITY      = "Hyderabad"
COUNTRY   = "IN"
PARAMETER = "pm25"   # pm25 | pm10 | no2 | o3 | co | so2
DAYS_BACK = 90
```

## Outputs

| File | Description |
|---|---|
| `outputs/1_eda_overview.png` | Time-series, hourly pattern, distribution |
| `outputs/2_actual_vs_predicted.png` | Test set evaluation with metrics |
| `outputs/3_forecast.png` | 24-hour color-coded AQI forecast |
| `outputs/4_category_distribution.png` | AQI category breakdown (pie chart) |
| `outputs/forecast.csv` | Tabular forecast with category + health advice |
| `models/metrics.json` | MAE, RMSE, R², category accuracy |

## AQI Scale

| AQI Range | Category | Color |
|---|---|---|
| 0–50 | Good | 🟢 |
| 51–100 | Moderate | 🟡 |
| 101–150 | Unhealthy for Sensitive Groups | 🟠 |
| 151–200 | Unhealthy | 🔴 |
| 201–300 | Very Unhealthy | 🟣 |
| 301–500 | Hazardous | ⚫ |
