"""Build a YOLO dataset from label review decisions."""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_dataset(yaml_path: Path) -> Tuple[Path, dict]:
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(config["path"])
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root, config


def resolve_split_dir(root: Path, config: dict, split: str) -> Path:
    split_path = Path(config[split])
    return split_path if split_path.is_absolute() else root / split_path


def load_reviews(path: Path) -> Dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("reviews", data)


def image_key(split: str, image_path: Path) -> str:
    return f"{split}/{image_path.name}"


def should_keep(decision: str, keep_unsure: bool, keep_pending: bool) -> bool:
    if decision == "keep":
        return True
    if decision == "unsure":
        return keep_unsure
    if decision == "pending":
        return keep_pending
    return False


def brighten_image(image: np.ndarray, factor: float) -> np.ndarray:
    factor = max(float(factor), 1.0)
    if np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        return np.clip(image.astype(np.float32) * factor, 0, max_value).astype(image.dtype)
    return np.clip(image.astype(np.float32) * factor, 0, 1.0)


def copy_or_brighten_image(source: Path, target: Path, enhance_brightness: bool, brightness_factor: float) -> None:
    if not enhance_brightness:
        shutil.copy2(source, target)
        return
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image for brightness preprocessing: {source}")
    brightened = brighten_image(image, brightness_factor)
    if not cv2.imwrite(str(target), brightened):
        raise ValueError(f"Cannot write brightened image: {target}")


def copy_split(
    source_root: Path,
    config: dict,
    output_dir: Path,
    reviews: Dict[str, dict],
    split: str,
    reviewed_splits: set[str],
    keep_unsure: bool,
    keep_pending: bool,
) -> dict:
    source_images = resolve_split_dir(source_root, config, split)
    target_images = output_dir / "images" / split
    target_labels = output_dir / "labels" / split
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    kept = []
    dropped = []
    for image_path in sorted(path for path in source_images.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        label_path = source_root / "labels" / split / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label: {label_path}")
        key = image_key(split, image_path)
        if split in reviewed_splits:
            decision = reviews.get(key, {}).get("decision", "pending")
        else:
            decision = "keep"
        if should_keep(decision, keep_unsure, keep_pending):
            enhance_brightness = bool(reviews.get(key, {}).get("enhance_brightness", False))
            brightness_factor = float(reviews.get(key, {}).get("brightness_factor", 1.0))
            copy_or_brighten_image(
                image_path,
                target_images / image_path.name,
                enhance_brightness,
                brightness_factor,
            )
            shutil.copy2(label_path, target_labels / label_path.name)
            kept.append(
                {
                    "image": image_path.name,
                    "decision": decision,
                    "enhance_brightness": enhance_brightness,
                    "brightness_factor": brightness_factor if enhance_brightness else 1.0,
                }
            )
        else:
            dropped.append({"image": image_path.name, "decision": decision})
    return {"split": split, "kept": kept, "dropped": dropped}


def build_dataset(
    source_yaml: Path,
    review_json: Path,
    output_dir: Path,
    reviewed_splits: List[str],
    keep_unsure: bool,
    keep_pending: bool,
    overwrite: bool,
) -> dict:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Add --overwrite to rebuild.")
        shutil.rmtree(output_dir)

    source_root, config = load_dataset(source_yaml)
    reviews = load_reviews(review_json)
    reviewed_split_set = set(reviewed_splits)
    summaries = []
    for split in ("train", "val"):
        summaries.append(
            copy_split(
                source_root,
                config,
                output_dir,
                reviews,
                split,
                reviewed_split_set,
                keep_unsure,
                keep_pending,
            )
        )

    names = config.get("names", {0: "worm"})
    yaml.safe_dump(
        {
            "path": output_dir.resolve().as_posix(),
            "train": "images/train",
            "val": "images/val",
            "test": None,
            "names": names,
        },
        (output_dir / "dataset.yaml").open("w", encoding="utf-8"),
        allow_unicode=True,
        sort_keys=False,
    )
    summary = {
        "source_dataset": str(source_yaml),
        "review_json": str(review_json),
        "reviewed_splits": reviewed_splits,
        "keep_unsure": keep_unsure,
        "keep_pending": keep_pending,
        "splits": summaries,
    }
    (output_dir / "reviewed_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    brightness_rows = []
    for split in summaries:
        for row in split["kept"]:
            brightness_rows.append(
                {
                    "split": split["split"],
                    "image_file": row["image"],
                    "enhance_brightness": row.get("enhance_brightness", False),
                    "brightness_factor": row.get("brightness_factor", 1.0),
                }
            )
    if brightness_rows:
        with (output_dir / "brightness_decisions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(brightness_rows[0]))
            writer.writeheader()
            writer.writerows(brightness_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a filtered YOLO dataset from label review decisions.")
    parser.add_argument("--source-data", default="label/combined_yolo_dataset/dataset.yaml", help="Original dataset.yaml")
    parser.add_argument("--review-json", default="label/label_review.json", help="Review JSON from review_dataset_labels.py")
    parser.add_argument("--output-dir", default="label/combined_yolo_dataset_reviewed", help="Filtered dataset output")
    parser.add_argument(
        "--reviewed-splits",
        nargs="+",
        default=["train"],
        choices=["train", "val"],
        help="Splits that were reviewed and should be filtered. Unreviewed splits are copied as-is.",
    )
    parser.add_argument("--keep-unsure", action="store_true", help="Keep images marked unsure")
    parser.add_argument("--keep-pending", action="store_true", help="Keep unreviewed images in reviewed splits")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory")
    args = parser.parse_args()

    summary = build_dataset(
        Path(args.source_data),
        Path(args.review_json),
        Path(args.output_dir),
        args.reviewed_splits,
        args.keep_unsure,
        args.keep_pending,
        args.overwrite,
    )
    for split in summary["splits"]:
        print(f"{split['split']}: kept={len(split['kept'])}, dropped={len(split['dropped'])}")
    print(f"Dataset written to: {Path(args.output_dir) / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
