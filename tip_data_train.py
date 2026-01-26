import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # Load the provided dataset
    # Expected columns from lab instructions
    columns = [
        "avg. Speed",
        "Morning",
        "Afternoon",
        "Evening",
        "weekend",
        "rain",
        "fog",
        "distance to travel",
        "travel time",
    ]
    data = pd.read_excel("DatasetLab1.xls", names=columns)

    # Show heads and shape
    print("Dataset head (default):")
    print(data.head())
    print("\nDataset head (first 5):")
    print(data.head(5))
    print("\nDataset shape:")
    print(data.shape)

    # Quick analysis
    print("\nSummary statistics:")
    print(data.describe(numeric_only=True))

    # Model development: predict travel time
    X = data[columns[:-1]].values
    y = data["travel time"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("\nLinear Regression results:")
    print(f"Intercept: {model.intercept_:.4f}")
    print("Slopes:", np.round(model.coef_, 4))
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2: {r2_score(y_test, y_pred):.4f}")

    # Plots
    plt.figure(figsize=(8, 5))
    sns.histplot(data["travel time"], bins=20, kde=True)
    plt.title("Distribution of Travel Time")
    plt.xlabel("Travel Time")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.title("Predicted vs Actual (Test Set)")
    plt.xlabel("Actual Travel Time")
    plt.ylabel("Predicted Travel Time")
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
