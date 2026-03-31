#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

LABEL_ORDER = ["non_defective", "defective"]


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    project_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description="Evaluate classifier and save report files under output/.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "Assiment/zeroq_cup_classification_scaffold/data/processed",
    )
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, default=script_root / "output" / "reports")
    return parser.parse_args()


def list_images(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in path.iterdir() if p.suffix.lower() in exts])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))

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
    y_pred: list[str] = []
    confs: list[float] = []
    for result in results:
        top1_idx = int(result.probs.top1)
        y_pred.append(result.names[top1_idx])
        confs.append(float(result.probs.top1conf))

    report = classification_report(y_true, y_pred, labels=LABEL_ORDER, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    report_file = args.output_dir / f"classification_report_{args.split}.txt"
    cm_file = args.output_dir / f"confusion_matrix_{args.split}.csv"
    preds_file = args.output_dir / f"predictions_{args.split}.csv"

    report_file.write_text(report)
    pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER).to_csv(cm_file)
    pd.DataFrame(
        {
            "image": [p.name for p in image_paths],
            "true_label": y_true,
            "pred_label": y_pred,
            "confidence": confs,
        }
    ).to_csv(preds_file, index=False)

    print(f"[DONE] report: {report_file}")
    print(f"[DONE] confusion matrix: {cm_file}")
    print(f"[DONE] predictions: {preds_file}")


if __name__ == "__main__":
    main()

