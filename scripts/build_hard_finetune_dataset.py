"""Build a fine-tuning dataset by oversampling and augmenting hard training images."""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def dataset_root(dataset_yaml: Path) -> Tuple[Path, dict]:
    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = Path(config["path"])
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    return root, config


def resolve_image_dir(root: Path, config: dict, split: str) -> Path:
    path = Path(config[split])
    return path if path.is_absolute() else root / path


def copy_base_dataset(dataset_yaml: Path, output_dir: Path, overwrite: bool) -> Tuple[Path, dict]:
    root, config = dataset_root(dataset_yaml)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在: {output_dir}。如需重建，请添加 --overwrite。")
        shutil.rmtree(output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image_path in resolve_image_dir(root, config, split).iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = root / "labels" / split / f"{image_path.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"缺少标签: {label}")
            shutil.copy2(image_path, output_dir / "images" / split / image_path.name)
            shutil.copy2(label, output_dir / "labels" / split / label.name)
    return root, config


def load_hard_train_images(report_path: Path) -> List[str]:
    with report_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = sorted(
        row["image_file"]
        for row in rows
        if row["split"] == "train" and str(row["is_hard"]).lower() in {"true", "1", "yes"}
    )
    if not names:
        raise ValueError("报告中没有训练集难例，不需要构建强化数据集")
    return names


def read_label_points(label_path: Path) -> List[Tuple[str, np.ndarray]]:
    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = line.split()
        points = np.asarray([float(value) for value in values[1:]], dtype=np.float32).reshape(-1, 2)
        labels.append((values[0], points))
    return labels


def write_label_points(label_path: Path, labels: List[Tuple[str, np.ndarray]]) -> None:
    lines = []
    for class_id, points in labels:
        points = np.clip(points, 0.0, 1.0)
        lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in points.flatten()))
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def transform_image_labels(
    image: np.ndarray, labels: List[Tuple[str, np.ndarray]], transform: str, rng: np.random.Generator
) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    transformed = [(class_id, points.copy()) for class_id, points in labels]
    if transform == "rot90":
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        transformed = [(cls, np.column_stack((1.0 - pts[:, 1], pts[:, 0]))) for cls, pts in transformed]
    elif transform == "rot180":
        image = cv2.rotate(image, cv2.ROTATE_180)
        transformed = [(cls, 1.0 - pts) for cls, pts in transformed]
    elif transform == "rot270":
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        transformed = [(cls, np.column_stack((pts[:, 1], 1.0 - pts[:, 0]))) for cls, pts in transformed]
    elif transform == "fliplr":
        image = cv2.flip(image, 1)
        transformed = [(cls, np.column_stack((1.0 - pts[:, 0], pts[:, 1]))) for cls, pts in transformed]
    elif transform == "flipud":
        image = cv2.flip(image, 0)
        transformed = [(cls, np.column_stack((pts[:, 0], 1.0 - pts[:, 1]))) for cls, pts in transformed]

    factor = float(rng.uniform(0.85, 1.15))
    noise = rng.normal(0, 2.5, image.shape)
    augmented = np.clip(image.astype(np.float32) * factor + noise, 0, 255).astype(np.uint8)
    return augmented, transformed


def build_dataset(
    base_yaml: Path,
    report_path: Path,
    output_dir: Path,
    repeat_copies: int,
    augment_copies: int,
    seed: int,
    overwrite: bool,
) -> dict:
    if repeat_copies < 0 or augment_copies < 0:
        raise ValueError("重复与增强副本数量不能为负数")
    root, config = copy_base_dataset(base_yaml, output_dir, overwrite)
    train_images = resolve_image_dir(root, config, "train")
    hard_names = load_hard_train_images(report_path)
    rng = np.random.default_rng(seed)
    transformations = ("rot90", "rot180", "rot270", "fliplr", "flipud")
    generated = []

    for name in hard_names:
        source_image = train_images / name
        source_label = root / "labels" / "train" / f"{source_image.stem}.txt"
        if not source_image.is_file() or not source_label.is_file():
            raise FileNotFoundError(f"报告中的训练难例不在基础数据集中: {name}")
        for copy_index in range(1, repeat_copies + 1):
            stem = f"{source_image.stem}__hardrepeat{copy_index:02d}"
            target_image = output_dir / "images" / "train" / f"{stem}{source_image.suffix}"
            target_label = output_dir / "labels" / "train" / f"{stem}.txt"
            shutil.copy2(source_image, target_image)
            shutil.copy2(source_label, target_label)
            generated.append({"source": name, "file": target_image.name, "type": "repeat"})

        image = cv2.imread(str(source_image), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"无法读取训练难例: {source_image}")
        labels = read_label_points(source_label)
        for aug_index in range(1, augment_copies + 1):
            transform = transformations[(aug_index - 1) % len(transformations)]
            aug_image, aug_labels = transform_image_labels(image, labels, transform, rng)
            stem = f"{source_image.stem}__hardaug{aug_index:02d}_{transform}"
            image_path = output_dir / "images" / "train" / f"{stem}.png"
            label_path = output_dir / "labels" / "train" / f"{stem}.txt"
            cv2.imwrite(str(image_path), aug_image)
            write_label_points(label_path, aug_labels)
            generated.append({"source": name, "file": image_path.name, "type": transform})

    base_train_count = len([path for path in resolve_image_dir(root, config, "train").iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS])
    base_val_count = len([path for path in resolve_image_dir(root, config, "val").iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS])
    summary = {
        "base_dataset": str(base_yaml),
        "hard_report": str(report_path),
        "seed": seed,
        "hard_train_images": len(hard_names),
        "repeat_copies_per_hard_image": repeat_copies,
        "augment_copies_per_hard_image": augment_copies,
        "base_train_images": base_train_count,
        "base_val_images": base_val_count,
        "generated_train_images": len(generated),
        "final_train_images": base_train_count + len(generated),
        "final_val_images": base_val_count,
        "generated": generated,
    }
    (output_dir / "dataset.yaml").write_text(
        f"path: {output_dir.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest:\n\nnames:\n  0: worm\n",
        encoding="utf-8",
    )
    (output_dir / "hard_finetune_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Oversample and augment only hard training images for fine-tuning.")
    parser.add_argument("--base-data", default="label/combined_yolo_dataset/dataset.yaml", help="原始合并数据集配置")
    parser.add_argument("--hard-report", default="hard_mining/stage2_best/hard_examples.csv", help="难例筛选 CSV")
    parser.add_argument("--output-dir", default="label/hard_finetune_dataset", help="强化训练数据集输出目录")
    parser.add_argument("--repeat-copies", type=int, default=2, help="每张训练难例原样重复次数")
    parser.add_argument("--augment-copies", type=int, default=2, help="每张训练难例生成增强图次数")
    parser.add_argument("--seed", type=int, default=42, help="亮度与噪声增强随机种子")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出目录")
    args = parser.parse_args()

    summary = build_dataset(
        Path(args.base_data),
        Path(args.hard_report),
        Path(args.output_dir),
        args.repeat_copies,
        args.augment_copies,
        args.seed,
        args.overwrite,
    )
    print(
        f"强化数据集完成: hard_train={summary['hard_train_images']}, "
        f"train={summary['final_train_images']}, val={summary['final_val_images']}"
    )


if __name__ == "__main__":
    main()
