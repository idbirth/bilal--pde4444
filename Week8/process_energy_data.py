#!/usr/bin/env python3
"""Week 8 energy data processing pipeline.

This script converts the Week 8 notebook workflow into a standalone Python file and
produces cleaned outputs, including `processed_energy_data.csv`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "Energy_Data.csv"


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate missing values and percentages by column."""
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: "Missing Values", 1: "% of Total Values"}
    )
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0
    ].sort_values("% of Total Values", ascending=False).round(1)

    print(
        f"Your selected dataframe has {df.shape[1]} columns.\n"
        f"There are {mis_val_table_ren_columns.shape[0]} columns that have missing values."
    )
    return mis_val_table_ren_columns


def remove_collinear_features(x: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Remove highly collinear features using correlation threshold."""
    y = x["score"].copy()
    x = x.drop(columns=["score"])

    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j : (j + 1), (i + 1) : (i + 2)]
            col = item.columns
            val = abs(item.values)
            if val >= threshold:
                drop_cols.append(col.values[0])

    drops = set(drop_cols)
    x = x.drop(columns=list(drops), errors="ignore")

    # Match explicit removals from the notebook.
    x = x.drop(
        columns=[
            "Weather Normalized Site EUI (kBtu/ft²)",
            "Water Use (All Water Sources) (kgal)",
            "log_Water Use (All Water Sources) (kgal)",
            "Largest Property Use Type - Gross Floor Area (ft²)",
        ],
        errors="ignore",
    )

    x["score"] = y
    return x


def main() -> None:
    pd.set_option("display.max_columns", 60)

    # 1) Load data.
    data = pd.read_csv(INPUT_CSV, low_memory=False)

    # 2) Replace explicit missing token and coerce selected columns to numeric.
    data = data.replace({"Not Available": np.nan})
    for col in list(data.columns):
        if (
            "ft²" in col
            or "kBtu" in col
            or "Metric Tons CO2e" in col
            or "kWh" in col
            or "therms" in col
            or "gal" in col
            or "Score" in col
        ):
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # 3) Drop columns with > 50% missing values.
    missing_df = missing_values_table(data)
    missing_columns = list(missing_df[missing_df["% of Total Values"] > 50].index)
    print(f"We will remove {len(missing_columns)} columns.")
    data = data.drop(columns=missing_columns)

    # 4) Rename target score column.
    data = data.rename(columns={"ENERGY STAR Score": "score"})

    # 5) Remove outliers from Site EUI using IQR rule (3 * IQR).
    first_quartile = data["Site EUI (kBtu/ft²)"].describe()["25%"]
    third_quartile = data["Site EUI (kBtu/ft²)"].describe()["75%"]
    iqr = third_quartile - first_quartile
    data = data[
        (data["Site EUI (kBtu/ft²)"] > (first_quartile - 3 * iqr))
        & (data["Site EUI (kBtu/ft²)"] < (third_quartile + 3 * iqr))
    ].copy()

    # 6) Build feature table (numeric + log transforms + one-hot categorical).
    numeric_subset = data.select_dtypes("number").copy()
    for col in numeric_subset.columns:
        if col == "score":
            continue
        numeric_subset[f"log_{col}"] = np.log(numeric_subset[col].where(numeric_subset[col] > 0))

    categorical_subset = data[["Borough", "Largest Property Use Type"]].copy()
    categorical_subset = pd.get_dummies(categorical_subset)

    features = pd.concat([numeric_subset, categorical_subset], axis=1)

    # 7) Remove collinear features and all-NaN columns.
    features = remove_collinear_features(features, 0.6)
    features = features.dropna(axis=1, how="all")

    # Save full processed table (includes rows with and without score).
    features.to_csv(BASE_DIR / "processed_energy_data.csv", index=False)

    # 8) Split scored/no-score rows and save train/test files from notebook flow.
    no_score = features[features["score"].isna()].copy()
    score = features[features["score"].notnull()].copy()

    ml_features = score.drop(columns="score").replace({np.inf: np.nan, -np.inf: np.nan})
    targets = pd.DataFrame(score["score"])

    X, X_test, y, y_test = train_test_split(
        ml_features,
        targets,
        test_size=0.3,
        random_state=42,
    )

    no_score.to_csv(BASE_DIR / "no_score.csv", index=False)
    X.to_csv(BASE_DIR / "training_features.csv", index=False)
    X_test.to_csv(BASE_DIR / "testing_features.csv", index=False)
    y.to_csv(BASE_DIR / "training_labels.csv", index=False)
    y_test.to_csv(BASE_DIR / "testing_labels.csv", index=False)

    print("\nSaved files:")
    print("- processed_energy_data.csv")
    print("- no_score.csv")
    print("- training_features.csv")
    print("- testing_features.csv")
    print("- training_labels.csv")
    print("- testing_labels.csv")
    print(f"Processed data shape: {features.shape}")
    print(f"Rows with score: {score.shape}")
    print(f"Rows without score: {no_score.shape}")


if __name__ == "__main__":
    main()
