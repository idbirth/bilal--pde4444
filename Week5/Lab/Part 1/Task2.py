import warnings

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier


warnings.filterwarnings("ignore", category=ConvergenceWarning)


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

    print("Task 2: Split into X and y")
    X = df[["sepal length", "sepal width", "petal length", "petal width"]].values
    y = df["target"].values
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Classes:", sorted(df["target"].unique()), "\n")

    print("Task 3: Train/test split (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0], "\n")

    print("Task 4: Create MLP model")
    model = MLPClassifier(
        hidden_layer_sizes=(10,),
        activation="logistic",
        solver="sgd",
        learning_rate="constant",
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
    )

    print("Task 5: Train and test model")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", f"{accuracy_score(y_test, y_pred):.4f}", "\n")

    print("Task 6: Confusion matrix")
    print(confusion_matrix(y_test, y_pred), "\n")
    print("Task 6: Classification report")
    print(classification_report(y_test, y_pred))

    print("Additional: Hyperparameter tuning with GridSearchCV")
    param_grid = {
        "hidden_layer_sizes": [(10,), (20,), (20, 10)],
        "activation": ["logistic", "tanh", "relu"],
        "solver": ["sgd", "adam"],
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [1000],
    }
    tuned = GridSearchCV(
        MLPClassifier(random_state=42),
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=1,
    )
    tuned.fit(X_train, y_train)
    tuned_pred = tuned.best_estimator_.predict(X_test)
    print("Best params:", tuned.best_params_)
    print("Best cross-validation score:", f"{tuned.best_score_:.4f}")
    print("Tuned test accuracy:", f"{accuracy_score(y_test, tuned_pred):.4f}")


if __name__ == "__main__":
    main()
