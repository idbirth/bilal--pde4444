#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABELS = {"defective": 1, "non_defective": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a quick HOG + LogisticRegression baseline.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    return parser.parse_args()


def list_images(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in path.iterdir() if p.suffix.lower() in exts])


def extract_features(image_path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read {image_path}")
    img = cv2.resize(img, (size, size))
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features.astype(np.float32)


def load_split(data_root: Path, split: str, size: int):
    X, y = [], []
    for class_name, label in LABELS.items():
        class_dir = data_root / split / class_name
        for img_path in list_images(class_dir):
            X.append(extract_features(img_path, size))
            y.append(label)
    return np.stack(X), np.array(y)


def main() -> None:
    args = parse_args()
    X_train, y_train = load_split(args.data_root, "train", args.size)
    eval_split = "val" if (args.data_root / "val").exists() else "test"
    X_eval, y_eval = load_split(args.data_root, eval_split, args.size)

    clf = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_eval)
    print(classification_report(y_eval, y_pred, target_names=["non_defective", "defective"]))


if __name__ == "__main__":
    main()
