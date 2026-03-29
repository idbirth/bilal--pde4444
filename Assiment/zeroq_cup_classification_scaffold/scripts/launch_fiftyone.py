#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import fiftyone as fo
import fiftyone.types as fot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch FiftyOne on an image classification dataset.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--name", type=str, default="zeroq-cups")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = fo.Dataset.from_dir(
        dataset_dir=str(args.data_root),
        dataset_type=fot.ImageClassificationDirectoryTree,
        name=args.name,
    )
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
