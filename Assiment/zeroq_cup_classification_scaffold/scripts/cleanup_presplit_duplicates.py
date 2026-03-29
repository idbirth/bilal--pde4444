#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    class_name: str
    source_group: str
    exact_hash: str
    d_hash: int


@dataclass(frozen=True)
class Decision:
    image: ImageInfo
    keep: bool
    reason: str
    matched_to: str
    near_distance: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove exact and near-duplicate images from an augmented dataset before splitting."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--near-threshold",
        type=int,
        default=4,
        help="Maximum dHash Hamming distance for near-duplicate removal. Use -1 to disable near-duplicate removal.",
    )
    parser.add_argument(
        "--near-scope",
        type=str,
        default="group",
        choices=["group", "class"],
        help="Compare near duplicates within a source group or across the whole class.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional CSV path for a keep/remove manifest. Defaults to <output-root>/cleanup_report.csv.",
    )
    return parser.parse_args()


def list_images(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in exts)


def source_group_name(path: Path) -> str:
    return path.stem.split("__")[0]


def read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def exact_hash(image_bgr) -> str:
    h = hashlib.sha1()
    h.update(str(image_bgr.shape).encode("utf-8"))
    h.update(image_bgr.tobytes())
    return h.hexdigest()


def dhash(image_bgr) -> int:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]

    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def keep_priority(path: Path) -> tuple[int, str]:
    is_original = 0 if "__orig" in path.stem else 1
    return (is_original, path.name)


def matched_label(image: ImageInfo) -> str:
    return f"{image.class_name}/{image.path.name}"


def load_class_images(class_dir: Path) -> list[ImageInfo]:
    infos: list[ImageInfo] = []
    for path in list_images(class_dir):
        image = read_image(path)
        infos.append(
            ImageInfo(
                path=path,
                class_name=class_dir.name,
                source_group=source_group_name(path),
                exact_hash=exact_hash(image),
                d_hash=dhash(image),
            )
        )
    return sorted(infos, key=lambda info: keep_priority(info.path))


def remove_exact_duplicates(images: list[ImageInfo]) -> tuple[list[ImageInfo], list[Decision]]:
    kept: list[ImageInfo] = []
    decisions: list[Decision] = []
    seen_hashes: dict[str, ImageInfo] = {}

    for image in images:
        keeper = seen_hashes.get(image.exact_hash)
        if keeper is None:
            seen_hashes[image.exact_hash] = image
            kept.append(image)
            decisions.append(Decision(image=image, keep=True, reason="kept", matched_to=""))
            continue

        decisions.append(
            Decision(
                image=image,
                keep=False,
                reason="exact_duplicate",
                matched_to=matched_label(keeper),
            )
        )

    return kept, decisions


def near_duplicate_buckets(images: list[ImageInfo], near_scope: str) -> dict[str, list[ImageInfo]]:
    buckets: dict[str, list[ImageInfo]] = {}
    for image in images:
        bucket_key = image.class_name if near_scope == "class" else image.source_group
        buckets.setdefault(bucket_key, []).append(image)
    return buckets


def remove_near_duplicates(
    images: list[ImageInfo], near_threshold: int, near_scope: str
) -> tuple[list[ImageInfo], list[Decision]]:
    if near_threshold < 0:
        return images, [Decision(image=image, keep=True, reason="kept", matched_to="") for image in images]

    final_kept: list[ImageInfo] = []
    decisions: list[Decision] = []

    for bucket_images in near_duplicate_buckets(images, near_scope).values():
        kept_in_bucket: list[ImageInfo] = []
        for image in bucket_images:
            matched_keeper: ImageInfo | None = None
            matched_distance: int | None = None

            for keeper in kept_in_bucket:
                distance = hamming_distance(image.d_hash, keeper.d_hash)
                if distance <= near_threshold:
                    matched_keeper = keeper
                    matched_distance = distance
                    break

            if matched_keeper is None:
                kept_in_bucket.append(image)
                final_kept.append(image)
                decisions.append(Decision(image=image, keep=True, reason="kept", matched_to=""))
                continue

            decisions.append(
                Decision(
                    image=image,
                    keep=False,
                    reason="near_duplicate",
                    matched_to=matched_label(matched_keeper),
                    near_distance=matched_distance,
                )
            )

    return final_kept, decisions


def combine_decisions(exact_decisions: list[Decision], near_decisions: list[Decision]) -> list[Decision]:
    near_map = {decision.image.path: decision for decision in near_decisions}
    combined: list[Decision] = []
    for decision in exact_decisions:
        if not decision.keep:
            combined.append(decision)
            continue
        combined.append(near_map[decision.image.path])
    return combined


def write_manifest(path: Path, decisions: list[Decision], input_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_name", "source_group", "path", "status", "reason", "matched_to", "near_distance"])
        for decision in decisions:
            writer.writerow(
                [
                    decision.image.class_name,
                    decision.image.source_group,
                    decision.image.path.relative_to(input_root),
                    "kept" if decision.keep else "removed",
                    decision.reason,
                    decision.matched_to,
                    "" if decision.near_distance is None else decision.near_distance,
                ]
            )


def copy_retained_images(decisions: list[Decision], input_root: Path, output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for decision in decisions:
        if not decision.keep:
            continue
        relative_path = decision.image.path.relative_to(input_root)
        destination = output_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(decision.image.path, destination)


def main() -> None:
    args = parse_args()
    report_path = args.report_path or (args.output_root / "cleanup_report.csv")

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    all_decisions: list[Decision] = []
    for class_dir in sorted(path for path in args.input_root.iterdir() if path.is_dir()):
        images = load_class_images(class_dir)
        exact_kept, exact_decisions = remove_exact_duplicates(images)
        _, near_decisions = remove_near_duplicates(exact_kept, args.near_threshold, args.near_scope)
        decisions = combine_decisions(exact_decisions, near_decisions)
        all_decisions.extend(decisions)

        exact_removed = sum(1 for decision in decisions if decision.reason == "exact_duplicate")
        near_removed = sum(1 for decision in decisions if decision.reason == "near_duplicate")
        kept = sum(1 for decision in decisions if decision.keep)
        print(
            f"[OK] {class_dir.name}: kept {kept}/{len(images)} images, "
            f"removed {exact_removed} exact and {near_removed} near duplicates"
        )

    copy_retained_images(all_decisions, args.input_root, args.output_root)
    write_manifest(report_path, all_decisions, args.input_root)
    print(f"[OK] wrote cleaned dataset to {args.output_root}")
    print(f"[OK] wrote cleanup manifest to {report_path}")


if __name__ == "__main__":
    main()
