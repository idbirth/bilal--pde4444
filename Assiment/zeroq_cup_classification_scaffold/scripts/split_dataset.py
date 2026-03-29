#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test using source groups.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_images(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in path.iterdir() if p.suffix.lower() in exts])


def source_group_name(path: Path) -> str:
    return path.stem.split("__")[0]


def assign_groups(groups: list[str], train: float, val: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    groups = groups[:]
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    n_test = max(0, n - n_train - n_val)

    if n > 0 and n_test == 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1

    splits = {}
    for i, g in enumerate(groups):
        if i < n_train:
            splits[g] = "train"
        elif i < n_train + n_val:
            splits[g] = "val"
        else:
            splits[g] = "test"
    return splits


def main() -> None:
    args = parse_args()
    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise ValueError("train + val + test must sum to 1.0")

    if args.output_root.exists():
        shutil.rmtree(args.output_root)

    for split in ["train", "val", "test"]:
        for class_dir in sorted([p for p in args.input_root.iterdir() if p.is_dir()]):
            (args.output_root / split / class_dir.name).mkdir(parents=True, exist_ok=True)

    for class_dir in sorted([p for p in args.input_root.iterdir() if p.is_dir()]):
        images = list_images(class_dir)
        grouped: dict[str, list[Path]] = defaultdict(list)
        for img in images:
            grouped[source_group_name(img)].append(img)

        split_map = assign_groups(sorted(grouped.keys()), args.train, args.val, args.seed)

        for group_name, files in grouped.items():
            split = split_map[group_name]
            for src in files:
                dst = args.output_root / split / class_dir.name / src.name
                shutil.copy2(src, dst)

        print(f"[OK] {class_dir.name}: {len(grouped)} source groups, {len(images)} images")


if __name__ == "__main__":
    main()
