import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris


def main():
    # Task-1: Read Iris dataset (local sklearn copy to avoid network dependency)
    iris = load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=["sepal length", "sepal width", "petal length", "petal width"],
    )
    target_names = np.array(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    df["target"] = target_names[iris.target]
    print("Task-1: Iris head:\n", df.head(), "\n")

    # Task-2: extract features
    features = ["sepal length", "sepal width", "petal length", "petal width"]
    x = df[features].values
    y = df["target"].values
    print("Task-2: x shape:", x.shape)
    print("Task-2: y shape:", y.shape, "\n")

    # Task-3: standardize data
    scaled_x = StandardScaler().fit_transform(x)
    print("Task-3: scaled_x mean (approx 0):", scaled_x.mean(axis=0))
    print("Task-3: scaled_x std (approx 1):", scaled_x.std(axis=0), "\n")

    # Task-4: covariance matrix
    C = np.cov(scaled_x.T)
    print("Task-4: Covariance matrix:\n", C, "\n")

    # Task-6: Eigenvalues and Eigenvectors
    eValue, eVector = np.linalg.eig(C)
    print("Task-6: Eigenvalues:\n", eValue)
    print("Task-6: Eigenvectors (columns):\n", eVector, "\n")

    # Sort by descending eigenvalues
    sort_idx = np.argsort(eValue)[::-1]
    eValue_sorted = eValue[sort_idx]
    eVector_sorted = eVector[:, sort_idx]

    # Task-7: percentage of information
    data_information = eValue_sorted / np.sum(eValue_sorted)
    print("Task-7: percentage of information (variance) preserved by each component:")
    print(data_information)
    print("total information is:")
    print(np.sum(data_information), "\n")

    # Task-8: reduce to 1D
    pc1 = eVector_sorted[:, 0]
    x_1d = scaled_x.dot(pc1)
    info_preserved_1d = data_information[0]
    info_loss_1d = 1.0 - info_preserved_1d
    print("Task-8: 1D reduced data shape:", x_1d.shape)
    print("Task-8: percentage of information loss (1D):", info_loss_1d, "\n")

    # Task-9: reduce to 2D
    pc2 = eVector_sorted[:, :2]
    x_2d = scaled_x.dot(pc2)
    info_preserved_2d = np.sum(data_information[:2])
    info_loss_2d = 1.0 - info_preserved_2d
    print("Task-9: 2D reduced data shape:", x_2d.shape)
    print("Task-9: percentage of information loss (2D):", info_loss_2d, "\n")

    # Task-10: sklearn PCA comparison
    pca_1 = PCA(n_components=1)
    pca_2 = PCA(n_components=2)
    pca_1.fit(scaled_x)
    pca_2.fit(scaled_x)
    print("Task-10: sklearn PCA (1 component) explained_variance_ratio_:")
    print(pca_1.explained_variance_ratio_)
    print("Task-10: sklearn PCA (2 components) explained_variance_ratio_:")
    print(pca_2.explained_variance_ratio_)
    print("Task-10: sklearn PCA components (2 components):\n", pca_2.components_)


if __name__ == "__main__":
    main()
