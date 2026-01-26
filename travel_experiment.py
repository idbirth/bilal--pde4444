import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(path):
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
    return pd.read_csv(path, names=columns)


def run_experiment(data, features, target, test_size, random_state, cv_folds):
    X = data[features].values
    y = data[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

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
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        pipeline, param_grid=param_grid, scoring="r2", cv=cv, n_jobs=1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_all = best_model.predict(X)

    model = best_model.named_steps["model"]
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "features": features,
        "target": target,
        "test_size": test_size,
        "random_state": random_state,
        "cv_folds": cv_folds,
        "best_model": model.__class__.__name__,
        "best_alpha": getattr(model, "alpha", None),
        "intercept": float(model.intercept_),
        "coefficients": [float(x) for x in np.ravel(model.coef_)],
        "metrics": {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mse": float(mean_squared_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        },
    }

    return results, y, y_pred_all


def save_results(path, results):
    Path(path).write_text(json.dumps(results, indent=2))


def save_plots(output_dir, y, y_pred_all, target):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.histplot(y, bins=20, kde=True)
    plt.title(f"Distribution of {target}")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "distribution.png", dpi=150)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y, y=y_pred_all)
    min_val = min(y.min(), y_pred_all.min())
    max_val = max(y.max(), y_pred_all.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
    plt.title("Predicted vs Actual (All Rows)")
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "predicted_vs_actual.png", dpi=150)


def run_batch(data, target):
    batches = [
        {
            "features": [
                "Morning",
                "Afternoon",
                "Evening",
                "avg. Speed",
                "rain",
                "fog",
                "weekend",
                "distance to travel",
            ],
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5,
            "reason": "Baseline with all predictors and standard 80/20 split.",
        },
        {
            "features": [
                "Morning",
                "Afternoon",
                "Evening",
                "avg. Speed",
                "rain",
                "fog",
                "distance to travel",
            ],
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5,
            "reason": "Remove 'weekend' to reduce redundancy with time-of-day flags.",
        },
        {
            "features": [
                "avg. Speed",
                "rain",
                "fog",
                "distance to travel",
            ],
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5,
            "reason": "Focus on continuous factors + weather; drop time-of-day dummies.",
        },
        {
            "features": [
                "Morning",
                "Afternoon",
                "Evening",
                "avg. Speed",
                "rain",
                "fog",
                "weekend",
            ],
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5,
            "reason": "Exclude 'distance to travel' to test over-reliance.",
        },
        {
            "features": [
                "Morning",
                "Afternoon",
                "Evening",
                "avg. Speed",
                "rain",
                "fog",
                "weekend",
                "distance to travel",
            ],
            "test_size": 0.3,
            "random_state": 42,
            "cv_folds": 5,
            "reason": "Larger test split to check generalization.",
        },
        {
            "features": [
                "Morning",
                "Afternoon",
                "Evening",
                "avg. Speed",
                "rain",
                "fog",
                "weekend",
                "distance to travel",
            ],
            "test_size": 0.2,
            "random_state": 7,
            "cv_folds": 5,
            "reason": "Different random seed to check split sensitivity.",
        },
        {
            "features": [
                "Morning",
                "Afternoon",
                "Evening",
                "avg. Speed",
                "rain",
                "fog",
                "weekend",
                "distance to travel",
            ],
            "test_size": 0.2,
            "random_state": 99,
            "cv_folds": 5,
            "reason": "Another seed to gauge variability.",
        },
        {
            "features": [
                "Morning",
                "Afternoon",
                "Evening",
                "avg. Speed",
                "rain",
                "fog",
                "weekend",
                "distance to travel",
            ],
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 4,
            "reason": "Fewer CV folds to reduce variance with tiny dataset.",
        },
        {
            "features": [
                "Morning",
                "Afternoon",
                "Evening",
                "avg. Speed",
                "rain",
                "fog",
                "weekend",
                "distance to travel",
            ],
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 3,
            "reason": "Even fewer folds to test stability.",
        },
        {
            "features": [
                "Morning",
                "Afternoon",
                "Evening",
                "avg. Speed",
                "rain",
                "fog",
                "weekend",
                "distance to travel",
            ],
            "test_size": 0.1,
            "random_state": 42,
            "cv_folds": 5,
            "reason": "Smaller test split to maximize training data.",
        },
    ]

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.txt"
    lines = []

    for idx, cfg in enumerate(batches, start=1):
        results, y, y_pred_all = run_experiment(
            data,
            cfg["features"],
            target,
            cfg["test_size"],
            cfg["random_state"],
            cfg["cv_folds"],
        )
        results["reason"] = cfg["reason"]

        result_path = results_dir / f"iter_{idx:02d}.json"
        save_results(result_path, results)

        plot_dir = results_dir / f"iter_{idx:02d}_plots"
        save_plots(plot_dir, y, y_pred_all, target)

        lines.append(
            f"iter_{idx:02d}: r2={results['metrics']['r2']:.4f}, "
            f"mae={results['metrics']['mae']:.4f}, "
            f"mse={results['metrics']['mse']:.4f} | {cfg['reason']}"
        )

    summary_path.write_text("\n".join(lines))
    print(f"Saved {len(batches)} iterations to {results_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run travel time regression experiment and save results."
    )
    parser.add_argument("--data", default="DatasetLab1.xls")
    parser.add_argument(
        "--features",
        default="Morning,Afternoon,Evening,avg. Speed,rain,fog,weekend,distance to travel",
        help="Comma-separated feature list",
    )
    parser.add_argument("--target", default="travel time")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--plots-dir", default="plots")
    return parser.parse_args()


def main():
    args = parse_args()
    features = [f.strip() for f in args.features.split(",") if f.strip()]

    data = load_data(args.data)
    if args.results == "batch":
        run_batch(data, args.target)
    else:
        results, y, y_pred_all = run_experiment(
            data,
            features,
            args.target,
            args.test_size,
            args.random_state,
            args.cv_folds,
        )

        save_results(args.results, results)
        save_plots(args.plots_dir, y, y_pred_all, args.target)

        print("Saved:", args.results)
        print("Plots in:", args.plots_dir)


if __name__ == "__main__":
    main()
