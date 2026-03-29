#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly undersample the cleaned dataset so each class matches the minority-class image count."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reference-class",
        type=str,
        default=None,
        help="Optional class name whose cleaned image count should be matched by all classes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible undersampling.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional CSV path for a keep/remove manifest. Defaults to <output-root>/balance_report.csv.",
    )
    return parser.parse_args()


def list_images(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in exts)


def choose_target_count(class_images: dict[str, list[Path]], reference_class: str | None) -> int:
    if reference_class is not None:
        if reference_class not in class_images:
            raise ValueError(f"Reference class not found in dataset: {reference_class}")
        return len(class_images[reference_class])
    return min(len(images) for images in class_images.values())


def write_report(
    report_path: Path,
    decisions: list[tuple[str, str, str, str]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_name", "path", "status", "reason"])
        writer.writerows(decisions)


def main() -> None:
    args = parse_args()
    report_path = args.report_path or (args.output_root / "balance_report.csv")

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    class_dirs = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found in: {args.input_root}")

    class_images = {class_dir.name: list_images(class_dir) for class_dir in class_dirs}
    empty_classes = [name for name, images in class_images.items() if not images]
    if empty_classes:
        raise ValueError(f"Cannot balance dataset with empty classes: {', '.join(empty_classes)}")

    target_count = choose_target_count(class_images, args.reference_class)
    rng = random.Random(args.seed)

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    decisions: list[tuple[str, str, str, str]] = []

    for class_name, images in class_images.items():
        output_dir = args.output_root / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if len(images) <= target_count:
            kept = set(images)
            reason = "kept_at_or_below_target"
        else:
            kept = set(rng.sample(images, target_count))
            reason = "random_undersample_keep"

        kept_count = 0
        removed_count = 0
        for image in images:
            relative_path = image.relative_to(args.input_root)
            if image in kept:
                shutil.copy2(image, args.output_root / relative_path)
                decisions.append((class_name, str(relative_path), "kept", reason))
                kept_count += 1
            else:
                decisions.append((class_name, str(relative_path), "removed", "random_undersample_remove"))
                removed_count += 1

        print(
            f"[OK] {class_name}: kept {kept_count}/{len(images)} images "
            f"(removed {removed_count}, target={target_count})"
        )

    write_report(report_path, decisions)
    print(f"[OK] balance report: {report_path}")


if __name__ == "__main__":
    main()
