#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer


class NoAugClassificationTrainer(ClassificationTrainer):
    """Force train split through non-augmented transform path."""

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=mode)


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    project_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description="Train YOLO classifier and save all runs under output/.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "Assiment/zeroq_cup_classification_scaffold/data/processed",
    )
    parser.add_argument("--model", type=Path, default=Path("yolo26s-cls.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="week9_cls_train")
    parser.add_argument("--output-dir", type=Path, default=script_root / "output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.data_root}")

    model = YOLO(str(args.model))
    if model.task != "classify":
        raise SystemExit(
            f"Expected classification checkpoint, but got task={model.task!r} from {args.model}"
        )

    print(f"[INFO] model={args.model}")
    print(f"[INFO] data_root={args.data_root}")
    print(f"[INFO] output_dir={args.output_dir}")
    print(f"[INFO] run_name={args.run_name}")

    model.train(
        trainer=NoAugClassificationTrainer,
        data=str(args.data_root),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
        project=str(args.output_dir),
        name=args.run_name,
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()

