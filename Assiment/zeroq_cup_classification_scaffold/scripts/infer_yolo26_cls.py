#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

PASS_FAIL_MAP = {
    "defective": "FAIL",
    "non_defective": "PASS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO classification inference and print PASS/FAIL decisions.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Image file or folder of images")
    return parser.parse_args()


def iter_images(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.is_file():
        return [path]
    return sorted([p for p in path.iterdir() if p.suffix.lower() in exts])


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    images = iter_images(args.input)
    if not images:
        raise FileNotFoundError(f"No images found in {args.input}")

    results = model.predict(source=[str(p) for p in images], verbose=False)
    for img_path, result in zip(images, results):
        top1_index = int(result.probs.top1)
        pred_class = result.names[top1_index]
        confidence = float(result.probs.top1conf)
        decision = PASS_FAIL_MAP.get(pred_class, pred_class.upper())
        print(f"{img_path.name}	class={pred_class}	decision={decision}	confidence={confidence:.4f}")


if __name__ == "__main__":
    main()
