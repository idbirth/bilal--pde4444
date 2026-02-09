import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path


def load_data():
    # Use the dataset inside this folder
    xls_path = Path(__file__).resolve().parent / "Clustering.xls"
    try:
        # Some environments lack Excel engines and this .xls is CSV text.
        # Try Excel first, then fall back to CSV.
        try:
            df = pd.read_excel(xls_path)
        except ValueError:
            df = pd.read_csv(xls_path)
        return df
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read {xls_path}. Ensure the file exists and a suitable Excel engine is installed."
        ) from exc


def main():
    # Task 1: read dataset and plot it
    data = load_data()
    print("Data head:\n", data.head(), "\n")

    plt.figure(figsize=(5, 5))
    plt.scatter(data["Weight"], data["Height"])
    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.title("Data Distribution")
    plt.savefig("data_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Task 2: elbow method
    X = data[["Weight", "Height"]]
    variance = []
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X)
        variance.append(kmeans.inertia_)

    print("Elbow inertias (k=2..19):")
    print(variance, "\n")

    plt.figure(figsize=(15, 5))
    plt.plot(range(2, 20), variance)
    plt.grid(True)
    plt.title("Elbow curve")
    plt.savefig("elbow_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Task 3: KMeans clustering
    num_clusters = 3
    kmeans_model = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans_model.fit(X)

    pred = kmeans_model.predict(X)
    print("Cluster labels:")
    print(pred)

    # Plot clustering results
    frame = data.copy()
    frame["cluster"] = pred

    colors = ["blue", "green", "cyan", "black"]
    for k in range(num_clusters):
        cluster_data = frame[frame["cluster"] == k]
        plt.scatter(cluster_data["Weight"], cluster_data["Height"], c=colors[k])
    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.title("K-means Clusters")
    plt.savefig("kmeans_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
