#!/usr/bin/env python3
"""
Week 3 Lab (single script):
1) Perceptron on Health dataset (Age, BloodPressure, Cholesterol, BMI, SmokingStatus -> Disease)
2) SVM on Iris dataset (4 features -> target), multi-class One-vs-Rest (OVR)

Run:
  python week3_lab.py --health-path health_data.xls
Optional:
  python week3_lab.py --health-path health_data.xls --save-plots --output-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt


HEALTH_FEATURES = ["Age", "BloodPressure", "Cholesterol", "BMI", "SmokingStatus"]
HEALTH_TARGET = "Disease"

IRIS_UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
IRIS_COLS = ["sepal length", "sepal width", "petal length", "petal width", "target"]


def load_health_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    The provided 'health_data.xls' is actually CSV text.
    We read it with read_csv (not read_excel).
    """
    df = pd.read_csv(path)

    missing = [c for c in (HEALTH_FEATURES + [HEALTH_TARGET]) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Health dataset is missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    X = df[HEALTH_FEATURES].copy()
    y = df[HEALTH_TARGET].copy()

    # Safety: ensure numeric
    X = X.apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(y, errors="raise").astype(int)

    return X, y


def load_iris_dataset(uci_url: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Try loading Iris from the UCI URL as requested.
    If that fails (e.g., no internet), fall back to sklearn's built-in iris dataset.
    """
    try:
        iris = pd.read_csv(uci_url, header=None, names=IRIS_COLS)
        iris = iris.dropna()
        X = iris[IRIS_COLS[:-1]].copy()
        y = iris["target"].copy()
        # ensure numeric features
        X = X.apply(pd.to_numeric, errors="raise")
        return X, y
    except Exception as e:
        print(f"[WARN] Could not load Iris from URL ({uci_url}). Reason: {e}")
        print("[WARN] Falling back to sklearn.datasets.load_iris()")

        from sklearn.datasets import load_iris

        data = load_iris(as_frame=True)
        X = data.data.copy()
        y = data.target.map(lambda i: data.target_names[i]).copy()
        return X, y


def evaluate_and_print(name: str, y_true, y_pred) -> None:
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


def save_confusion_matrix_plot(cm, title: str, out_path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # annotate values
    for (i, j), v in __import__("numpy").ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_health_perceptron(args) -> None:
    X, y = load_health_dataset(args.health_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("perceptron", Perceptron(eta0=0.3, max_iter=1000, tol=1e-3, random_state=args.random_state)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    evaluate_and_print("Task 1: Perceptron (Health)", y_test, y_pred)

    if args.save_plots:
        cm = confusion_matrix(y_test, y_pred)
        save_confusion_matrix_plot(
            cm,
            "Perceptron Confusion Matrix (Health)",
            args.output_dir / "health_perceptron_confusion_matrix.png",
        )
        print(f"[OK] Saved: {args.output_dir / 'health_perceptron_confusion_matrix.png'}")


def run_iris_svm(args) -> None:
    X, y = load_iris_dataset(args.iris_url)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    svm_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(C=1, decision_function_shape="ovr")),  # One-vs-Rest
        ]
    )

    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    evaluate_and_print("Task 2: SVM (Iris, OVR)", y_test, y_pred)

    if args.save_plots:
        cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
        save_confusion_matrix_plot(
            cm,
            "SVM Confusion Matrix (Iris)",
            args.output_dir / "iris_svm_confusion_matrix.png",
        )
        print(f"[OK] Saved: {args.output_dir / 'iris_svm_confusion_matrix.png'}")


def parse_args():
    p = argparse.ArgumentParser(description="PDE4444 Week 3 Lab - Single Script (Perceptron + SVM)")
    p.add_argument(
        "--health-path",
        type=Path,
        default=Path("health_data.xls"),
        help="Path to health dataset (CSV text file; can be .xls or .csv). Default: health_data.xls",
    )
    p.add_argument(
        "--iris-url",
        type=str,
        default=IRIS_UCI_URL,
        help="UCI Iris dataset URL. Default is the standard UCI iris.data URL.",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Test split ratio. Default: 0.2")
    p.add_argument("--random-state", type=int, default=42, help="Random seed. Default: 42")
    p.add_argument("--save-plots", action="store_true", help="Save confusion matrix plots as PNG")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Where to save plots if --save-plots is set. Default: outputs/",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.health_path.exists():
        raise FileNotFoundError(
            f"Health dataset not found at: {args.health_path.resolve()}\n"
            f"Put the file in the same folder as this script, or pass --health-path."
        )

    run_health_perceptron(args)
    run_iris_svm(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
