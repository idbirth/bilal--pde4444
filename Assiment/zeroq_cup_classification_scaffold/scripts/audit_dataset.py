#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cleanvision import Imagelab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dataset images with CleanVision.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_root = args.data_root / args.split
    if not split_root.exists():
        raise FileNotFoundError(f"Split path not found: {split_root}")

    imagelab = Imagelab(data_path=str(split_root))
    imagelab.find_issues()
    imagelab.report()


if __name__ == "__main__":
    main()
