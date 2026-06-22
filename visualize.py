"""
visualize.py
Generates plots for EDA, model evaluation, and forecast results.
Saves all charts to outputs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os

os.makedirs("outputs", exist_ok=True)

PALETTE = {
    "Good":                         "#00C853",
    "Moderate":                     "#FFD600",
    "Unhealthy for Sensitive":      "#FF6D00",
    "Unhealthy":                    "#D50000",
    "Very Unhealthy":               "#7B1FA2",
    "Hazardous":                    "#4E342E",
}

def aqi_color(aqi: float) -> str:
    if aqi <= 50:   return "#00C853"
    if aqi <= 100:  return "#FFD600"
    if aqi <= 150:  return "#FF6D00"
    if aqi <= 200:  return "#D50000"
    if aqi <= 300:  return "#7B1FA2"
    return "#4E342E"


# ── 1. EDA: AQI time-series overview ────────────────────────────────────────
def plot_timeseries():
    df = pd.read_csv("data/processed_aqi.csv", parse_dates=["datetime"])
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor="#F5F5F5")
    fig.suptitle("AQI Data Overview", fontsize=16, fontweight="bold", y=1.01)

    # Full time-series
    ax = axes[0]
    ax.plot(df["datetime"], df["aqi"], linewidth=0.8, color="#1565C0", alpha=0.8)
    ax.axhline(50,  color="#00C853", linestyle="--", linewidth=0.8, label="Good/Moderate (50)")
    ax.axhline(100, color="#FFD600", linestyle="--", linewidth=0.8, label="Moderate/USG (100)")
    ax.axhline(150, color="#FF6D00", linestyle="--", linewidth=0.8, label="USG/Unhealthy (150)")
    ax.set_title("AQI Over Time", fontweight="bold")
    ax.set_ylabel("AQI")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_facecolor("white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Hourly average pattern
    ax = axes[1]
    hourly = df.groupby("hour")["aqi"].mean()
    bars = ax.bar(hourly.index, hourly.values, color=[aqi_color(v) for v in hourly.values])
    ax.set_title("Average AQI by Hour of Day", fontweight="bold")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean AQI")
    ax.set_facecolor("white")
    ax.set_xticks(range(0, 24, 2))

    # Distribution
    ax = axes[2]
    ax.hist(df["aqi"], bins=50, color="#1565C0", alpha=0.75, edgecolor="white")
    ax.axvline(df["aqi"].mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {df['aqi'].mean():.1f}")
    ax.axvline(df["aqi"].median(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median = {df['aqi'].median():.1f}")
    ax.set_title("AQI Distribution", fontweight="bold")
    ax.set_xlabel("AQI")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.set_facecolor("white")

    plt.tight_layout()
    plt.savefig("outputs/1_eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved outputs/1_eda_overview.png")


# ── 2. Actual vs Predicted (test set) ───────────────────────────────────────
def plot_actual_vs_predicted():
    df = pd.read_csv("data/test_predictions.csv", parse_dates=["datetime"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="#F5F5F5")
    fig.suptitle("Model Evaluation: Actual vs Predicted AQI", fontsize=16, fontweight="bold")

    # Time-series comparison
    ax = axes[0]
    ax.plot(df["datetime"], df["aqi"],           label="Actual",    color="#1565C0", linewidth=1)
    ax.plot(df["datetime"], df["predicted_aqi"], label="Predicted", color="#E53935",
            linewidth=1, linestyle="--", alpha=0.8)
    ax.set_title("Actual vs Predicted AQI (Test Set)", fontweight="bold")
    ax.set_ylabel("AQI")
    ax.legend()
    ax.set_facecolor("white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # Scatter plot
    ax = axes[1]
    ax.scatter(df["aqi"], df["predicted_aqi"],
               alpha=0.3, s=10, color="#1565C0", edgecolors="none")
    lims = [min(df["aqi"].min(), df["predicted_aqi"].min()),
            max(df["aqi"].max(), df["predicted_aqi"].max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    ax.set_title("Scatter: Actual vs Predicted", fontweight="bold")
    ax.legend()
    ax.set_facecolor("white")

    # Annotate with metrics
    try:
        with open("models/metrics.json") as f:
            m = json.load(f)
        txt = (f"MAE  = {m['test']['mae']}\n"
               f"RMSE = {m['test']['rmse']}\n"
               f"R²   = {m['test']['r2']}\n"
               f"Cat. accuracy = {m['test']['category_accuracy']}%")
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    except FileNotFoundError:
        pass

    plt.tight_layout()
    plt.savefig("outputs/2_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved outputs/2_actual_vs_predicted.png")


# ── 3. 24-hour Forecast ──────────────────────────────────────────────────────
def plot_forecast():
    try:
        df = pd.read_csv("outputs/forecast.csv", parse_dates=["datetime"])
    except FileNotFoundError:
        print("No forecast file found. Run predict.py first.")
        return

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#F5F5F5")

    # Color-coded bars by AQI level
    colors = [aqi_color(v) for v in df["predicted_aqi"]]
    bars = ax.bar(df["datetime"], df["predicted_aqi"], color=colors, width=0.035, alpha=0.85)

    # AQI threshold bands
    for level, color, label in [
        (50,  "#00C853", "Good"),
        (100, "#FFD600", "Moderate"),
        (150, "#FF6D00", "USG"),
        (200, "#D50000", "Unhealthy"),
    ]:
        ax.axhline(level, color=color, linewidth=1, linestyle=":", alpha=0.7)
        ax.text(df["datetime"].iloc[-1], level + 2, label,
                color=color, fontsize=8, ha="right")

    # Line overlay
    ax.plot(df["datetime"], df["predicted_aqi"], "k-", linewidth=1.2, alpha=0.6)

    ax.set_title("24-Hour AQI Forecast", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date / Time")
    ax.set_ylabel("Predicted AQI")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.set_facecolor("white")

    # Summary box
    txt = (f"Min: {df['predicted_aqi'].min():.0f}  "
           f"Max: {df['predicted_aqi'].max():.0f}  "
           f"Avg: {df['predicted_aqi'].mean():.0f}")
    ax.text(0.5, 0.96, txt, transform=ax.transAxes, ha="center",
            fontsize=10, fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig("outputs/3_forecast.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved outputs/3_forecast.png")


# ── 4. AQI Category pie chart ────────────────────────────────────────────────
def plot_category_distribution():
    df = pd.read_csv("data/processed_aqi.csv")

    def cat(aqi):
        if aqi <= 50:   return "Good"
        if aqi <= 100:  return "Moderate"
        if aqi <= 150:  return "Unhealthy for Sensitive"
        if aqi <= 200:  return "Unhealthy"
        if aqi <= 300:  return "Very Unhealthy"
        return "Hazardous"

    counts = df["aqi"].apply(cat).value_counts()
    colors = [PALETTE.get(c, "#999") for c in counts.index]

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#F5F5F5")
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("AQI Category Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/4_category_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved outputs/4_category_distribution.png")


def main():
    print("Generating plots ...\n")
    plot_timeseries()
    plot_actual_vs_predicted()
    plot_forecast()
    plot_category_distribution()
    print("\nAll plots saved to outputs/")


if __name__ == "__main__":
    main()
