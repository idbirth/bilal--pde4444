import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    # Load provided travel time dataset (CSV text with .xls extension)
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
    data = pd.read_csv("DatasetLab1.xls", names=columns)

    # Show heads and shape
    print("Dataset head (default):")
    print(data.head())
    print("\nDataset head (first 5):")
    print(data.head(5))
    print("\nDataset shape:")
    print(data.shape)
    print(f"Total rows: {len(data)}")

    # Quick analysis
    print("\nSummary statistics:")
    print(data.describe())

    # Use all available predictors to fit the small dataset better
    features = [
        "Morning",
        "Afternoon",
        "Evening",
        "avg. Speed",
        "rain",
        "fog",
        "weekend",
        "distance to travel",
    ]
    target = "travel time"

    X = data[features].values
    y = data[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Optimize training parameters using cross-validated search
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ]
    )
    param_grid = [
        {"model": [LinearRegression()]},
        {"model": [Ridge()], "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        {"model": [Lasso(max_iter=10000)], "model__alpha": [0.001, 0.01, 0.1, 1.0]},
    ]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipeline, param_grid=param_grid, scoring="r2", cv=cv, n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_all = best_model.predict(X)

    model = best_model.named_steps["model"]
    print("\nLinear Regression results (optimized):")
    print("Best model:", model.__class__.__name__)
    if hasattr(model, "alpha"):
        print(f"Best alpha: {model.alpha}")
    print(f"Intercept: {model.intercept_:.4f}")
    print("Slopes:", np.round(model.coef_, 4))
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2: {r2_score(y_test, y_pred):.4f}")

    # Plots
    plt.figure(figsize=(8, 5))
    sns.histplot(data[target], bins=20, kde=True)
    plt.title("Distribution of Travel Time")
    plt.xlabel("Travel Time")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y, y=y_pred_all)
    min_val = min(y.min(), y_pred_all.min())
    max_val = max(y.max(), y_pred_all.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
    plt.title("Predicted vs Actual (All Rows)")
    plt.xlabel("Actual Travel Time")
    plt.ylabel("Predicted Travel Time")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
