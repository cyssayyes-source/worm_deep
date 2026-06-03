"""Merge an existing split YOLO dataset with additional labelled training images."""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def paired_items(dataset_dir: Path, split: str) -> List[Tuple[Path, Path]]:
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        return []
    items = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"图像缺少标签: {image_path} -> {label_path}")
        validate_segmentation_label(label_path)
        items.append((image_path, label_path))
    return items


def validate_segmentation_label(label_path: Path) -> None:
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if values[0] != "0" or len(values) < 7 or (len(values) - 1) % 2:
            raise ValueError(f"非单类 YOLO 分割标签格式: {label_path}:{line_number}")
        if not all(0.0 <= float(value) <= 1.0 for value in values[1:]):
            raise ValueError(f"归一化坐标越界: {label_path}:{line_number}")


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在: {output_dir}。如需重建，请添加 --overwrite。")
        shutil.rmtree(output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_items(items: Iterable[Tuple[Path, Path]], output_dir: Path, split: str, seen: Dict[str, str]) -> int:
    count = 0
    for image_path, label_path in items:
        if image_path.stem in seen:
            raise ValueError(f"数据集文件名冲突: {image_path.stem} 已来自 {seen[image_path.stem]}")
        seen[image_path.stem] = str(image_path)
        shutil.copy2(image_path, output_dir / "images" / split / image_path.name)
        shutil.copy2(label_path, output_dir / "labels" / split / label_path.name)
        count += 1
    return count


def merge_datasets(
    base_dir: Path,
    extra_dir: Path,
    output_dir: Path,
    extra_val_ratio: float,
    seed: int,
    overwrite: bool,
) -> dict:
    if not 0 < extra_val_ratio < 1:
        raise ValueError("--extra-val-ratio 必须介于 0 和 1 之间")
    base_train = paired_items(base_dir, "train")
    base_val = paired_items(base_dir, "val")
    if not base_train or not base_val:
        raise ValueError("基础数据集必须已包含非空 train 和 val 划分")

    extra_all = paired_items(extra_dir, "train") + paired_items(extra_dir, "val")
    if not extra_all:
        raise ValueError("附加数据集中没有可合并的图像/标签")
    rng = random.Random(seed)
    rng.shuffle(extra_all)
    extra_val_count = max(1, int(round(len(extra_all) * extra_val_ratio)))
    extra_val = extra_all[:extra_val_count]
    extra_train = extra_all[extra_val_count:]

    prepare_output(output_dir, overwrite)
    seen = {}
    counts = {
        "base_train": copy_items(base_train, output_dir, "train", seen),
        "base_val": copy_items(base_val, output_dir, "val", seen),
        "extra_train": copy_items(extra_train, output_dir, "train", seen),
        "extra_val": copy_items(extra_val, output_dir, "val", seen),
    }
    counts["train"] = counts["base_train"] + counts["extra_train"]
    counts["val"] = counts["base_val"] + counts["extra_val"]

    dataset_yaml = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test:\n\n"
        "names:\n"
        "  0: worm\n"
    )
    (output_dir / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    summary = {
        "base_dir": str(base_dir),
        "extra_dir": str(extra_dir),
        "output_dir": str(output_dir),
        "extra_val_ratio": extra_val_ratio,
        "seed": seed,
        "counts": counts,
        "extra_val_images": sorted(image.name for image, _ in extra_val),
    }
    (output_dir / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ROISet and Labelme YOLO segmentation datasets.")
    parser.add_argument("--base-dir", default="label/roi_yolo_dataset", help="已有 train/val 划分的基础数据集")
    parser.add_argument("--extra-dir", default="label/yolo_dataset", help="需要加入的额外单类数据集")
    parser.add_argument("--output-dir", default="label/combined_yolo_dataset", help="合并输出数据集目录")
    parser.add_argument("--extra-val-ratio", type=float, default=0.2, help="额外样本分入验证集的比例")
    parser.add_argument("--seed", type=int, default=42, help="额外样本划分随机种子")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的合并数据集目录")
    args = parser.parse_args()

    summary = merge_datasets(
        Path(args.base_dir),
        Path(args.extra_dir),
        Path(args.output_dir),
        args.extra_val_ratio,
        args.seed,
        args.overwrite,
    )
    counts = summary["counts"]
    print(
        f"合并完成: train={counts['train']} (ROISet={counts['base_train']}, Labelme={counts['extra_train']}), "
        f"val={counts['val']} (ROISet={counts['base_val']}, Labelme={counts['extra_val']})"
    )


if __name__ == "__main__":
    main()
