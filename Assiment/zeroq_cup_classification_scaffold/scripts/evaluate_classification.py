#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

LABEL_ORDER = ["non_defective", "defective"]
PASS_FAIL_MAP = {
    "defective": "FAIL",
    "non_defective": "PASS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier on a split and print report-friendly metrics.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="Path to trained weights, e.g. best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def list_images(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in path.iterdir() if p.suffix.lower() in exts])


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.predictions))

    y_true: list[str] = []
    image_paths: list[Path] = []
    for label in LABEL_ORDER:
        class_dir = args.data_root / args.split / label
        if not class_dir.exists():
            continue
        for img in list_images(class_dir):
            image_paths.append(img)
            y_true.append(label)

    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.data_root / args.split}")

    results = model.predict(source=[str(p) for p in image_paths], verbose=False)
    y_pred = []
    for result in results:
        top1_index = int(result.probs.top1)
        y_pred.append(result.names[top1_index])

    print("Classification report")
    print(classification_report(y_true, y_pred, labels=LABEL_ORDER, digits=4))
    print("Confusion matrix (rows=true, cols=pred)")
    print(confusion_matrix(y_true, y_pred, labels=LABEL_ORDER))
    print("\nPASS/FAIL mapping")
    for cls in LABEL_ORDER:
        print(f"{cls} -> {PASS_FAIL_MAP[cls]}")


if __name__ == "__main__":
    main()
