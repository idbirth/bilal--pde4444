import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # Load the tips dataset
    data = sns.load_dataset("tips")

    # Show heads and shape
    print("Dataset head (default):")
    print(data.head())
    print("\nDataset head (first 5):")
    print(data.head(5))
    print("\nDataset shape:")
    print(data.shape)

    # Quick analysis
    print("\nSummary statistics:")
    print(data[["total_bill", "tip"]].describe())

    # Model development: predict tip from total_bill
    X = data[["total_bill"]].values
    y = data["tip"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nLinear Regression results:")
    print(f"Intercept: {model.intercept_:.4f}")
    print("Slopes:", np.round(model.coef_, 4))
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2: {r2_score(y_test, y_pred):.4f}")

    # Plots
    plt.figure(figsize=(8, 5))
    sns.histplot(data["tip"], bins=20, kde=True)
    plt.title("Distribution of Tip")
    plt.xlabel("Tip")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x="total_bill", y="tip")
    plt.title("Tip vs Total Bill")
    plt.xlabel("Total Bill")
    plt.ylabel("Tip")
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.title("Predicted vs Actual (Test Set)")
    plt.xlabel("Actual Tip")
    plt.ylabel("Predicted Tip")
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
