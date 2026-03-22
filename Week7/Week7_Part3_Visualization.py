#!/usr/bin/env python3
"""Week 7 Part 3: Visualization examples converted from notebook."""

import random

import matplotlib.pyplot as plt


def pie_chart_sales() -> None:
    sales = [100, 400, 300, 600]
    slice_labels = ["1st Qtr", "2nd Qtr", "3rd Qtr", "4th Qtr"]
    colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99"]
    plt.figure()
    plt.pie(sales, labels=slice_labels, autopct="%1.1f%%", colors=colors)
    plt.title("Sales by Quarter")
    plt.legend()
    plt.show()


def pie_chart_energy() -> None:
    energy_consumption = [120, 450, 200, 80, 50]
    slice_labels = [
        "Smart Light",
        "HVAC",
        "Smart Plugs",
        "Security Cameras",
        "Others",
    ]
    plt.figure()
    plt.pie(energy_consumption, labels=slice_labels)
    plt.title("Energy Consumption in a Smart Home")
    plt.legend()
    plt.show()


def bar_chart_numeric_x() -> None:
    left_edges = [0, 10, 20, 30, 40]
    heights = [100, 200, 300, 400, 500]
    bar_width = 5
    colors = ["#FFD700", "#FF6347", "#87CEFA", "#32CD32", "#8A2BE2"]
    plt.figure()
    plt.bar(left_edges, heights, bar_width, color=colors)
    plt.title("This is the Title of the Bar Chart")
    plt.xlabel("X Axis Label")
    plt.ylabel("Y Axis Label")
    plt.show()


def bar_chart_categorical_x() -> None:
    left_edges = [
        "Smart Lights",
        "Security Cameras",
        "Smart Thermostat",
        "Smart Plugs",
        "Smart Door Lock",
    ]
    heights = [5, 50, 10, 8, 3]
    colors = ["#FFD700", "#FF6347", "#87CEFA", "#32CD32", "#8A2BE2"]
    plt.figure()
    plt.bar(left_edges, heights, color=colors)
    plt.title("Data Usage")
    plt.xticks(rotation=90)
    plt.xlabel("Devices")
    plt.ylabel("Data Usage in GB")
    plt.tight_layout()
    plt.show()


def scatter_time_temperature() -> None:
    time = [i for i in range(1, 25)]
    temperature = [random.uniform(18, 30) for _ in range(24)]
    plt.figure()
    plt.scatter(time, temperature, color="blue", marker="o")
    plt.title("Temperature Readings Over 24 Hours", fontsize=14)
    plt.xlabel("Time (hours)", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.grid(True, linestyle="--")
    plt.show()


def scatter_temperature_humidity() -> None:
    temperature = [random.uniform(15, 35) for _ in range(50)]
    humidity = [random.uniform(30, 90) for _ in range(50)]
    plt.figure()
    plt.scatter(temperature, humidity, color="blue", marker="o")
    plt.title("Temperature and humidity Readings")
    plt.xlabel("Temperature (°C)", fontsize=12)
    plt.ylabel("humidity (%)", fontsize=12)
    plt.grid(True, linestyle="--")
    plt.show()


def line_temperature_over_time() -> None:
    time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    temperature = [22, 23, 21, 25, 26, 24, 27, 28, 29, 30]
    plt.figure()
    plt.plot(time, temperature, marker="o", linestyle="-", color="blue", label="Temperature")
    plt.title("Temperature Trends Over Time", fontsize=14)
    plt.xlabel("Time (Hours)", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.xlim(0, 11)
    plt.ylim(20, 32)
    plt.xticks([1, 3, 5, 7, 9], ["1 AM", "3 AM", "5 AM", "7 AM", "9 AM"])
    plt.yticks([20, 22, 24, 26, 28, 30], ["20°C", "22°C", "24°C", "26°C", "28°C", "30°C"])
    plt.legend()
    plt.grid(True, linestyle="--")
    plt.show()


def line_energy_over_day() -> None:
    time = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    energy_consumption = [random.uniform(0.5, 2.5) for _ in range(12)]
    plt.figure()
    plt.plot(time, energy_consumption, marker="o", linestyle="-", color="green", label="Energy Usage")
    plt.title("Energy Consumption Trend Over 24 Hours", fontsize=14)
    plt.xlabel("Time (24-Hour Format)", fontsize=12)
    plt.ylabel("Energy Consumption (kWh)", fontsize=12)
    plt.xticks(
        time,
        ["00:00", "2:00", "4:00", "6:00 ", "8:00 ", "10:00 ", "12:00 ", "14:00", "16:00", "18:00 ", "20:00 ", "22:00"],
    )
    plt.yticks([0.5, 1, 1.5, 2, 2.5], ["0.5 kWh", "1 kWh", "1.5 kWh", "2 kWh", "2.5 kWh"])
    plt.xlim(-1, 23)
    plt.ylim(0, 3)
    plt.legend(title="Electricity Usage")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def histogram_temperature() -> None:
    temperatures = [random.uniform(18, 28) for _ in range(100)]
    plt.figure()
    plt.hist(temperatures, bins=10, color="skyblue", edgecolor="black")
    plt.title("Temperature Distribution in a Smart Building", fontsize=14)
    plt.xlabel("Temperature (°C)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def histogram_pm25() -> None:
    pm25_levels = [random.uniform(10, 150) for _ in range(100)]
    plt.figure()
    plt.hist(pm25_levels, bins=10, color="teal", edgecolor="black", alpha=0.7)
    plt.title("Distribution of PM2.5 Levels in a City", fontsize=14)
    plt.xlabel("PM2.5 Level (µg/m³)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(range(10, 161, 20))
    plt.yticks(range(0, 26, 5))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def main() -> None:
    random.seed(42)
    pie_chart_sales()
    pie_chart_energy()
    bar_chart_numeric_x()
    bar_chart_categorical_x()
    scatter_time_temperature()
    scatter_temperature_humidity()
    line_temperature_over_time()
    line_energy_over_day()
    histogram_temperature()
    histogram_pm25()


if __name__ == "__main__":
    main()
