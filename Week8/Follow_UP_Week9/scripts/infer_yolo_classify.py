#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

PASS_FAIL_MAP = {
    "defective": "FAIL",
    "non_defective": "PASS",
}


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Run YOLO classification inference and save CSV output.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Image file or folder")
    parser.add_argument("--output-dir", type=Path, default=script_root / "output" / "inference")
    parser.add_argument("--output-name", type=str, default="inference_predictions.csv")
    return parser.parse_args()


def iter_images(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.is_file():
        return [path]
    return sorted([p for p in path.iterdir() if p.suffix.lower() in exts])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    images = iter_images(args.input)
    if not images:
        raise FileNotFoundError(f"No images found in {args.input}")

    results = model.predict(source=[str(p) for p in images], verbose=False)

    rows = []
    for img_path, result in zip(images, results):
        top1_idx = int(result.probs.top1)
        pred_class = result.names[top1_idx]
        confidence = float(result.probs.top1conf)
        decision = PASS_FAIL_MAP.get(pred_class, pred_class.upper())
        rows.append(
            {
                "image": img_path.name,
                "pred_class": pred_class,
                "decision": decision,
                "confidence": confidence,
            }
        )
        print(
            f"{img_path.name}\tclass={pred_class}\tdecision={decision}\tconfidence={confidence:.4f}"
        )

    out_file = args.output_dir / args.output_name
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"[DONE] saved predictions to: {out_file}")


if __name__ == "__main__":
    main()

