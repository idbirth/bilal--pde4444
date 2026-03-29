#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import yaml
from tqdm import tqdm


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_transform(target_size: tuple[int, int], class_cfg: dict[str, Any]) -> A.Compose:
    tfs: list[Any] = []
    if class_cfg["pipeline"].get("resize_first", True):
        tfs.append(A.Resize(height=target_size[0], width=target_size[1]))

    for item in class_cfg["pipeline"]["transforms"]:
        name = item["name"]
        p = item.get("p", 1.0)

        if name == "RotationSweep":
            continue

        if name == "Affine":
            tfs.append(
                A.Affine(
                    scale=tuple(item.get("scale", [1.0, 1.0])),
                    translate_percent=tuple(item.get("translate_percent", [0.0, 0.0])),
                    rotate=tuple(item.get("rotate", [0, 0])),
                    shear=tuple(item.get("shear", [0, 0])),
                    p=p,
                    border_mode=cv2.BORDER_REPLICATE,
                )
            )
        elif name == "HorizontalFlip":
            tfs.append(A.HorizontalFlip(p=p))
        elif name == "RandomBrightnessContrast":
            tfs.append(
                A.RandomBrightnessContrast(
                    brightness_limit=tuple(item.get("brightness_limit", [-0.1, 0.1])),
                    contrast_limit=tuple(item.get("contrast_limit", [-0.1, 0.1])),
                    p=p,
                )
            )
        elif name == "HueSaturationValue":
            tfs.append(
                A.HueSaturationValue(
                    hue_shift_limit=item.get("hue_shift_limit", 0),
                    sat_shift_limit=item.get("sat_shift_limit", 0),
                    val_shift_limit=item.get("val_shift_limit", 0),
                    p=p,
                )
            )
        elif name == "GaussianBlur":
            blur_limit = tuple(item.get("blur_limit", [3, 5]))
            tfs.append(A.GaussianBlur(blur_limit=blur_limit, p=p))
        elif name == "MotionBlur":
            blur_limit = tuple(item.get("blur_limit", [3, 5]))
            tfs.append(A.MotionBlur(blur_limit=blur_limit, p=p))
        elif name == "GaussNoise":
            std_range = tuple(item.get("std_range", [0.01, 0.03]))
            tfs.append(A.GaussNoise(std_range=std_range, p=p))
        elif name == "CoarseDropout":
            tfs.append(
                A.CoarseDropout(
                    num_holes_range=tuple(item.get("num_holes_range", [1, 2])),
                    hole_height_range=tuple(item.get("hole_height_range", [0.02, 0.05])),
                    hole_width_range=tuple(item.get("hole_width_range", [0.02, 0.05])),
                    p=p,
                )
            )
        elif name == "Perspective":
            tfs.append(A.Perspective(scale=tuple(item.get("scale", [0.02, 0.05])), p=p))
        elif name == "Sharpen":
            tfs.append(
                A.Sharpen(
                    alpha=tuple(item.get("alpha", [0.1, 0.2])),
                    lightness=tuple(item.get("lightness", [0.8, 1.1])),
                    p=p,
                )
            )
        else:
            raise ValueError(f"Unsupported transform in YAML: {name}")

    return A.Compose(tfs)


def get_rotation_sweep(class_cfg: dict[str, Any]) -> list[int]:
    for item in class_cfg["pipeline"]["transforms"]:
        if item["name"] != "RotationSweep":
            continue

        if "angles" in item:
            return [int(angle) % 360 for angle in item["angles"]]

        start_deg = int(item.get("start_deg", 0))
        end_deg = int(item.get("end_deg", 359))
        step_deg = int(item.get("step_deg", 1))
        if step_deg <= 0:
            raise ValueError("RotationSweep step_deg must be > 0")
        if end_deg < start_deg:
            raise ValueError("RotationSweep end_deg must be >= start_deg")
        return [angle % 360 for angle in range(start_deg, end_deg + 1, step_deg)]

    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline augmented dataset files.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copy-originals", action="store_true", default=True)
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_image(path: Path, image_rgb) -> None:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def list_images(class_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in class_dir.iterdir() if p.suffix.lower() in exts])


def apply_rotation_sweep(image_rgb, angle_deg: int, target_size: tuple[int, int], resize_first: bool):
    if resize_first:
        working = cv2.resize(image_rgb, (target_size[1], target_size[0]))
    else:
        working = image_rgb

    height, width = working.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    rotated = cv2.warpAffine(
        working,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if rotated.shape[:2] != target_size:
        rotated = cv2.resize(rotated, (target_size[1], target_size[0]))

    return rotated


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)

    target_size = tuple(cfg.get("target_size", [224, 224]))
    image_ext = cfg.get("image_ext", ".jpg")

    ensure_clean_dir(args.output_root)

    for class_name, class_cfg in cfg["classes"].items():
        input_dir = args.input_root / class_name
        output_dir = args.output_root / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        images = list_images(input_dir)
        if not images:
            print(f"[WARN] No images found in {input_dir}")
            continue

        transform = build_transform(target_size, class_cfg)
        rotation_sweep = get_rotation_sweep(class_cfg)
        resize_first = bool(class_cfg["pipeline"].get("resize_first", True))
        per_source_variants = int(class_cfg.get("per_source_variants", 4))
        min_output_images = int(class_cfg.get("min_output_images", len(images)))

        base_count = len(images)
        copies_per_image = max(per_source_variants, math.ceil(max(0, min_output_images - base_count) / max(1, base_count)))

        if rotation_sweep:
            print(
                f"[INFO] {class_name}: using deterministic RotationSweep with "
                f"{len(rotation_sweep)} angles ({rotation_sweep[0]}..{rotation_sweep[-1]})"
            )

        for img_path in tqdm(images, desc=f"augment:{class_name}"):
            image = read_image(img_path)
            stem = img_path.stem

            original_out = output_dir / f"{stem}__orig{image_ext}"
            resized = cv2.resize(image, (target_size[1], target_size[0]))
            write_image(original_out, resized)

            if rotation_sweep:
                for angle in rotation_sweep:
                    if angle == 0:
                        continue
                    aug = apply_rotation_sweep(image, angle, target_size, resize_first)
                    out_path = output_dir / f"{stem}__rot_{angle:03d}{image_ext}"
                    write_image(out_path, aug)
                continue

            for idx in range(copies_per_image):
                aug = transform(image=image)["image"]
                out_path = output_dir / f"{stem}__aug_{idx:03d}{image_ext}"
                write_image(out_path, aug)

        print(f"[OK] {class_name}: wrote {len(list_images(output_dir))} files to {output_dir}")


if __name__ == "__main__":
    main()
