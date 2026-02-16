import os
from pathlib import Path

# Keep Matplotlib cache inside the project to avoid permission warnings.
_MPL_CACHE_DIR = Path(__file__).resolve().parent / ".mplconfig"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics, svm
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split


def load_iris_dataframe():
    iris = load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=["sepal length", "sepal width", "petal length", "petal width"],
    )
    target_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    df["target"] = [target_names[i] for i in iris.target]
    return df


def main():
    print("Task 1: Read Iris dataset")
    df = load_iris_dataframe()
    print(df.head(), "\n")

    print("Task 2: Extract features (X) and target (y)")
    features = ["sepal length", "sepal width", "petal length", "petal width"]
    X = df[features].values
    y = df["target"].values
    print("X shape:", X.shape)
    print("y shape:", y.shape, "\n")

    print("Task 3: Train/test split (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42, stratify=y
    )
    print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0], "\n")

    print("Task 4-7: Train and evaluate SVM with kernels: poly, sigmoid, linear")
    kernels = ["poly", "sigmoid", "linear"]
    accuracies = {}

    for kernel in kernels:
        model = svm.SVC(kernel=kernel, C=1, decision_function_shape="ovr", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        accuracies[kernel] = acc
        print(f"{kernel} accuracy: {acc:.4f}")
    print()

    print("Task 8: Save kernel accuracy bar plot")
    plt.figure(figsize=(6, 4))
    plt.bar(list(accuracies.keys()), list(accuracies.values()))
    plt.ylim(0, 1.05)
    plt.xlabel("Kernel")
    plt.ylabel("Accuracy")
    plt.title("SVM Kernel Accuracy")
    output_plot = Path(__file__).resolve().parent / "svm_kernel_accuracy.png"
    plt.savefig(output_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_plot}\n")

    print("Additional: GridSearchCV hyperparameter tuning")
    base_model = svm.SVC()
    hyperparameter_space = [
        {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001], "kernel": ["rbf", "sigmoid"]},
        {"C": [0.1, 1, 10, 100], "kernel": ["linear"]},
    ]

    optimizer = GridSearchCV(
        base_model,
        param_grid=hyperparameter_space,
        scoring="accuracy",
        cv=2,
        return_train_score=True,
    )
    optimizer.fit(X_train, y_train)

    print("Optimal hyperparameter combination:", optimizer.best_params_)
    print("Mean cross-validated training accuracy score:", f"{optimizer.best_score_:.4f}")

    best_model = optimizer.best_estimator_
    best_model.fit(X_train, y_train)
    print("Train accuracy:", f"{best_model.score(X_train, y_train):.4f}")

    y_pred_best = best_model.predict(X_test)
    print("Test accuracy:", f"{metrics.accuracy_score(y_test, y_pred_best):.4f}")


if __name__ == "__main__":
    main()
