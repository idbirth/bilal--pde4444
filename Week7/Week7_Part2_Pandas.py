#!/usr/bin/env python3
"""Week 7 Part 2: Pandas and visualization examples converted from notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "smart_sensors.csv"


def main() -> None:
    # Load dataset.
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("First 5 rows:")
    print(df.head())

    # Summary of dataset.
    print("\nDataset info:")
    df.info()

    # Convert columns to numeric; invalid values become NaN.
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["humid"] = pd.to_numeric(df["humid"], errors="coerce")
    df["dist"] = pd.to_numeric(df["dist"], errors="coerce")
    df["bright"] = pd.to_numeric(df["bright"], errors="coerce")
    df["soundlevel"] = pd.to_numeric(df["soundlevel"], errors="coerce")

    # Remove invalid distance rows.
    df = df[df["dist"] > 0].copy()

    # Check missing values.
    print("\nAny missing values by column:")
    print(df.isna().any())

    # Fill missing humidity/temperature with mean.
    df["humid"] = df["humid"].fillna(df["humid"].mean())
    df["temp"] = df["temp"].fillna(df["temp"].mean())

    print("\nTotal missing-value columns after fill:")
    print(df.isna().any().sum())

    # Descriptive stats.
    print("\nOverall describe():")
    print(df.describe(include="all"))

    print("\nNumeric summary (temp/humid/bright/dist):")
    print(df[["temp", "humid", "bright", "dist"]].describe())

    # Filter rows where temperature > 30 C.
    high_temp = df[df["temp"] > 30]
    print(f"\nRows with temp > 30 C: {len(high_temp)}")
    print(high_temp.head())

    # Group by node id and average temperature.
    avg_temp_by_node = df.groupby("nodeid")["temp"].mean()
    print("\nAverage temperature by nodeid:")
    print(avg_temp_by_node)

    # Convert result_time to datetime for time-series plotting.
    df["result_time"] = pd.to_datetime(df["result_time"], errors="coerce")
    df = df.dropna(subset=["result_time"]).copy()

    # Plot temperature and humidity over time.
    plt.figure(figsize=(12, 5))
    plt.plot(df["result_time"], df["temp"], label="Temperature (°C)", alpha=0.7)
    plt.plot(df["result_time"], df["humid"], label="Humidity (%)", alpha=0.7)
    plt.title("Temperature and Humidity Trends Over Time")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Temperature distribution.
    plt.figure(figsize=(8, 5))
    sns.histplot(df["temp"], kde=True, bins=30, color="blue")
    plt.title("Distribution of Temperature Readings")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Temperature vs humidity.
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="temp", y="humid", data=df, alpha=0.6)
    plt.title("Temperature vs Humidity")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Humidity (%)")
    plt.tight_layout()
    plt.show()

    # Pair plot for key sensor readings.
    pair_df = df[["temp", "humid", "bright", "dist"]].dropna()
    sns.pairplot(pair_df)
    plt.suptitle("Pair Plot of Key Sensor Readings", y=1.02)
    plt.show()

    # Correlation heatmap.
    correlation_matrix = df[["temp", "humid", "bright", "dist", "soundlevel"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, cmap="coolwarm")
    plt.title("Correlation Heatmap of Sensor Readings")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
