# K-Means Lab (Week 4)

This folder contains a Python script that reproduces the K-means lab workflow on the provided dataset.

## Files
- `Clustering.xls`: The dataset. Despite the `.xls` extension, it is CSV text.
- `Week 4 Lab K_Means.ipynb`: Original notebook instructions.
- `Week 4 Lab K_Means.html`: HTML export.
- `kmeans_tasks.py`: Python script implementation.

## What `kmeans_tasks.py` does

1. **Load the dataset**
   - `load_data()` reads `Clustering.xls` from the same folder as the script.
   - It first tries `pandas.read_excel`. If that fails (common when no Excel engine is installed and because the file is CSV text), it falls back to `pandas.read_csv`.

2. **Show a preview of the data**
   - Prints the first 5 rows with `data.head()`.

3. **Plot the raw data distribution**
   - Uses `Weight` on the x-axis and `Height` on the y-axis.
   - Saves the plot to `data_distribution.png`.

4. **Elbow method for choosing k**
   - Builds K-means models for `k = 2..19`.
   - Collects `inertia_` (within-cluster sum of squares).
   - Prints the list of inertias and saves the elbow plot to `elbow_curve.png`.

5. **Train K-means with 3 clusters**
   - Uses `num_clusters = 3` (from the elbow method).
   - Fits the model on the two features `Weight` and `Height`.
   - Prints the cluster label for each sample.

6. **Plot clustered points**
   - Colors each cluster differently.
   - Saves the result to `kmeans_clusters.png`.

## How to run

From the repository root:

```bash
python Week4/Lab/K-means/kmeans_tasks.py
```

## Outputs
- `data_distribution.png`
- `elbow_curve.png`
- `kmeans_clusters.png`
- Console output showing the data head, elbow inertias, and cluster labels.

## Notes
- If you run with a different Python environment, ensure `pandas`, `matplotlib`, and `scikit-learn` are installed.
- The script uses `random_state=42` and `n_init=10` for stable results.
