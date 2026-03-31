#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABELS = {"defective": 1, "non_defective": 0}
INDEX_TO_LABEL = {value: key for key, value in LABELS.items()}
DECISIONS = {"defective": "FAIL", "non_defective": "PASS"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "Sklearn_basle_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a quick HOG + LogisticRegression baseline.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
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
    X, y, paths = [], [], []
    for class_name, label in LABELS.items():
        class_dir = data_root / split / class_name
        for img_path in list_images(class_dir):
            X.append(extract_features(img_path, size))
            y.append(label)
            paths.append(img_path)
    return np.stack(X), np.array(y), paths


def create_run_dir(run_root: Path, run_name: str | None) -> Path:
    if run_name:
        run_dir = run_root / run_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = run_root / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def class_counts(y: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_name, label in LABELS.items():
        counts[class_name] = int(np.sum(y == label))
    return counts


def save_model(path: Path, model) -> None:
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_predictions_csv(
    path: Path,
    image_paths: list[Path],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_path",
                "true_label",
                "true_decision",
                "pred_label",
                "pred_decision",
                "prob_non_defective",
                "prob_defective",
                "confidence",
                "status",
            ]
        )
        for image_path, true_idx, pred_idx, probs in zip(image_paths, y_true, y_pred, y_proba):
            true_label = INDEX_TO_LABEL[int(true_idx)]
            pred_label = INDEX_TO_LABEL[int(pred_idx)]
            writer.writerow(
                [
                    image_path,
                    true_label,
                    DECISIONS[true_label],
                    pred_label,
                    DECISIONS[pred_label],
                    float(probs[0]),
                    float(probs[1]),
                    float(np.max(probs)),
                    "correct" if int(true_idx) == int(pred_idx) else "wrong",
                ]
            )


def save_confusion_matrix_csv(path: Path, cm: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", "non_defective", "defective"])
        writer.writerow(["non_defective", int(cm[0, 0]), int(cm[0, 1])])
        writer.writerow(["defective", int(cm[1, 0]), int(cm[1, 1])])


def put_text(img: np.ndarray, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int = 2):
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def save_confusion_matrix_image(path: Path, cm: np.ndarray) -> None:
    labels = ["non_defective", "defective"]
    cell = 170
    margin_left = 180
    margin_top = 110
    canvas = np.full((margin_top + cell * 2 + 40, margin_left + cell * 2 + 40, 3), 255, dtype=np.uint8)
    max_value = max(int(cm.max()), 1)

    put_text(canvas, "Confusion Matrix", (20, 40), 1.0, (0, 0, 0), 2)
    put_text(canvas, "Predicted", (margin_left + 65, 85), 0.8, (0, 0, 0), 2)
    put_text(canvas, "True", (40, margin_top + 110), 0.8, (0, 0, 0), 2)

    for idx, label in enumerate(labels):
        put_text(canvas, label, (margin_left + idx * cell + 12, 105), 0.55, (0, 0, 0), 2)
        put_text(canvas, label, (20, margin_top + idx * cell + 95), 0.55, (0, 0, 0), 2)

    row_sums = cm.sum(axis=1, keepdims=True)
    for row in range(2):
        for col in range(2):
            x0 = margin_left + col * cell
            y0 = margin_top + row * cell
            intensity = int(round(255 * (cm[row, col] / max_value)))
            color = cv2.applyColorMap(np.array([[intensity]], dtype=np.uint8), cv2.COLORMAP_BONE)[0, 0]
            cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell), tuple(int(v) for v in color), -1)
            cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell), (40, 40, 40), 2)

            percent = 0.0 if row_sums[row, 0] == 0 else (cm[row, col] / row_sums[row, 0]) * 100.0
            text_color = (255, 255, 255) if intensity > 140 else (0, 0, 0)
            put_text(canvas, str(int(cm[row, col])), (x0 + 62, y0 + 80), 1.2, text_color, 3)
            put_text(canvas, f"{percent:.1f}%", (x0 + 42, y0 + 118), 0.8, text_color, 2)

    cv2.imwrite(str(path), canvas)


def save_misclassified_grid(path: Path, image_paths: list[Path], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    mistakes = [(img_path, int(true_idx), int(pred_idx)) for img_path, true_idx, pred_idx in zip(image_paths, y_true, y_pred) if int(true_idx) != int(pred_idx)]
    if not mistakes:
        canvas = np.full((220, 640, 3), 255, dtype=np.uint8)
        put_text(canvas, "No misclassified evaluation images", (30, 110), 0.9, (0, 0, 0), 2)
        cv2.imwrite(str(path), canvas)
        return

    mistakes = mistakes[:12]
    cols = 3
    thumb = 220
    text_h = 70
    rows = math.ceil(len(mistakes) / cols)
    canvas = np.full((rows * (thumb + text_h) + 20, cols * thumb + 20, 3), 245, dtype=np.uint8)

    for idx, (img_path, true_idx, pred_idx) in enumerate(mistakes):
        row = idx // cols
        col = idx % cols
        x0 = 10 + col * thumb
        y0 = 10 + row * (thumb + text_h)

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            image = np.full((thumb - text_h, thumb, 3), 220, dtype=np.uint8)
        else:
            image = cv2.resize(image, (thumb, thumb - text_h))

        canvas[y0 : y0 + thumb - text_h, x0 : x0 + thumb] = image
        cv2.rectangle(canvas, (x0, y0), (x0 + thumb, y0 + thumb - text_h), (60, 60, 60), 2)

        true_label = INDEX_TO_LABEL[true_idx]
        pred_label = INDEX_TO_LABEL[pred_idx]
        put_text(canvas, f"true: {true_label}", (x0 + 8, y0 + thumb - 45), 0.45, (0, 120, 0), 1)
        put_text(canvas, f"pred: {pred_label}", (x0 + 8, y0 + thumb - 20), 0.45, (0, 0, 180), 1)

    cv2.imwrite(str(path), canvas)


def main() -> None:
    args = parse_args()
    run_dir = create_run_dir(args.run_root, args.run_name)

    X_train, y_train, train_paths = load_split(args.data_root, "train", args.size)
    eval_split = "val" if (args.data_root / "val").exists() else "test"
    X_eval, y_eval, eval_paths = load_split(args.data_root, eval_split, args.size)

    clf = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_eval)
    y_proba = clf.predict_proba(X_eval)

    report_text = classification_report(y_eval, y_pred, target_names=["non_defective", "defective"], digits=4)
    report_dict = classification_report(y_eval, y_pred, target_names=["non_defective", "defective"], output_dict=True, digits=4)
    cm = confusion_matrix(y_eval, y_pred, labels=[0, 1])
    accuracy = float(accuracy_score(y_eval, y_pred))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_eval, y_pred, average="macro", zero_division=0
    )

    save_model(run_dir / "model.pkl", clf)
    save_text(run_dir / "classification_report.txt", report_text)
    save_json(
        run_dir / "metrics.json",
        {
            "model_type": "HOG + LogisticRegression",
            "feature_size": args.size,
            "eval_split": eval_split,
            "train_counts": class_counts(y_train),
            "eval_counts": class_counts(y_eval),
            "accuracy": accuracy,
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "classification_report": report_dict,
        },
    )
    save_predictions_csv(run_dir / "predictions.csv", eval_paths, y_eval, y_pred, y_proba)
    save_confusion_matrix_csv(run_dir / "confusion_matrix.csv", cm)
    save_confusion_matrix_image(run_dir / "confusion_matrix.png", cm)
    save_misclassified_grid(run_dir / "misclassified_grid.jpg", eval_paths, y_eval, y_pred)

    print(report_text)
    print(f"[OK] sklearn baseline run saved to {run_dir}")
    print(f"[OK] model: {run_dir / 'model.pkl'}")
    print(f"[OK] metrics: {run_dir / 'metrics.json'}")
    print(f"[OK] visualization: {run_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
