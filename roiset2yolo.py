"""Convert manually classified ImageJ ROI sets to a YOLO segmentation dataset."""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

from roi_utils import discover_image_roi_pairs, polygon_mask, read_gray_image, read_roi_polygons


def read_annotations(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"找不到标注文件: {path}。请先运行 annotate_rois.py。")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("version") != 1 or not isinstance(data.get("images"), dict):
        raise ValueError(f"标注文件格式无效: {path}")
    return data


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在: {output_dir}。如需重建，请添加 --overwrite。")
        shutil.rmtree(output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def validate_annotation(image_path: Path, zip_path: Path, annotations: dict, polygons: List[dict]) -> Dict[str, str]:
    entry = annotations["images"].get(image_path.name)
    if not entry or not entry.get("completed"):
        raise ValueError(f"{image_path.name} 尚未完成 blank 标注")
    if entry.get("zip_file") != zip_path.name:
        raise ValueError(f"{image_path.name} 的标注关联 zip 与当前文件不一致")

    roi_labels = {item["id"]: item.get("label") for item in entry.get("rois", [])}
    current_ids = {polygon["id"] for polygon in polygons}
    if set(roi_labels) != current_ids:
        raise ValueError(f"{image_path.name} 的 ROI 集合已改变，请重新确认标注")
    blank_names = [name for name, label in roi_labels.items() if label == "blank"]
    if len(blank_names) != 1 or entry.get("blank_roi") != blank_names[0]:
        raise ValueError(f"{image_path.name} 必须且只能有一个 blank ROI")
    if any(label not in {"worm", "blank"} for label in roi_labels.values()):
        raise ValueError(f"{image_path.name} 包含未知 ROI 标签")
    return roi_labels


def yolo_line(points: np.ndarray, width: int, height: int) -> str:
    normalized = points.copy()
    normalized[:, 0] /= width
    normalized[:, 1] /= height
    values = " ".join(f"{value:.6f}" for value in normalized.flatten())
    return f"0 {values}"


def make_dataset(
    input_dir: Path,
    annotations_path: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    overwrite: bool,
) -> dict:
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio 必须介于 0 和 1 之间")
    pairs = discover_image_roi_pairs(input_dir)
    annotations = read_annotations(annotations_path)

    records = []
    for image_path, zip_path in pairs:
        gray = read_gray_image(image_path)
        height, width = gray.shape[:2]
        polygons = read_roi_polygons(zip_path, width, height)
        roi_labels = validate_annotation(image_path, zip_path, annotations, polygons)
        records.append((image_path, zip_path, gray, polygons, roi_labels))

    rng = random.Random(seed)
    image_names = [record[0].name for record in records]
    rng.shuffle(image_names)
    val_count = max(1, int(round(len(image_names) * val_ratio)))
    val_names = set(image_names[:val_count])

    prepare_output_dir(output_dir, overwrite)
    backgrounds = {"version": 1, "source_annotations": str(annotations_path), "images": {}}
    validation_rows = []
    split_counts = {"train": 0, "val": 0}
    worm_count = 0
    duplicate_worm_count = 0

    for image_path, zip_path, gray, polygons, roi_labels in records:
        split = "val" if image_path.name in val_names else "train"
        split_counts[split] += 1
        shutil.copy2(image_path, output_dir / "images" / split / image_path.name)

        labels = []
        seen_labels = set()
        combined_worm_mask = np.zeros(gray.shape, dtype=bool)
        blank_polygon = None
        for polygon in polygons:
            if roi_labels[polygon["id"]] == "blank":
                blank_polygon = polygon
            else:
                label_line = yolo_line(polygon["points"], gray.shape[1], gray.shape[0])
                if label_line in seen_labels:
                    duplicate_worm_count += 1
                    continue
                seen_labels.add(label_line)
                labels.append(label_line)
                combined_worm_mask |= polygon_mask(gray.shape, polygon["points"])
                worm_count += 1
        with (output_dir / "labels" / split / f"{image_path.stem}.txt").open("w", encoding="utf-8") as handle:
            handle.write("\n".join(labels) + ("\n" if labels else ""))

        blank_mask = polygon_mask(gray.shape, blank_polygon["points"])
        blank_pixels = gray[blank_mask]
        background_pixels = gray[~combined_worm_mask]
        blank_mean = float(np.mean(blank_pixels)) if blank_pixels.size else None
        background_median = float(np.median(background_pixels)) if background_pixels.size else None
        difference = (
            blank_mean - background_median
            if blank_mean is not None and background_median is not None
            else None
        )
        validation_rows.append(
            {
                "image_file": image_path.name,
                "split": split,
                "worm_roi_count": len(labels),
                "blank_roi": blank_polygon["id"],
                "blank_roi_name": blank_polygon["name"],
                "blank_mean_intensity": blank_mean,
                "non_worm_background_median": background_median,
                "difference_blank_mean_minus_background_median": difference,
            }
        )
        backgrounds["images"][image_path.name] = {
            "zip_file": zip_path.name,
            "split": split,
            "blank_roi": blank_polygon["id"],
            "blank_roi_name": blank_polygon["name"],
            "points": blank_polygon["points"].tolist(),
        }

    dataset_yaml = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test:\n\n"
        "names:\n"
        "  0: worm\n"
    )
    (output_dir / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    with (output_dir / "background_rois.json").open("w", encoding="utf-8") as handle:
        json.dump(backgrounds, handle, ensure_ascii=False, indent=2)
    with (output_dir / "background_validation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation_rows[0]))
        writer.writeheader()
        writer.writerows(validation_rows)

    summary = {
        "input_dir": str(input_dir),
        "annotations": str(annotations_path),
        "output_dir": str(output_dir),
        "seed": seed,
        "val_ratio": val_ratio,
        "image_count": len(records),
        "split_counts": split_counts,
        "worm_roi_count": worm_count,
        "blank_roi_count": len(records),
        "duplicate_worm_roi_count_skipped": duplicate_worm_count,
    }
    with (output_dir / "conversion_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert classified ImageJ ROI sets to YOLO segmentation labels.")
    parser.add_argument("--input-dir", default="RoiSet", help="包含 TIFF 图像与同名 ROI zip 的目录")
    parser.add_argument("--annotations", default="RoiSet/roi_labels.json", help="annotate_rois.py 生成的标注 JSON")
    parser.add_argument("--output-dir", default="label/roi_yolo_dataset", help="YOLO 数据集输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集图片比例")
    parser.add_argument("--seed", type=int, default=42, help="训练/验证拆分随机种子")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出数据集目录")
    args = parser.parse_args()

    summary = make_dataset(
        Path(args.input_dir),
        Path(args.annotations),
        Path(args.output_dir),
        args.val_ratio,
        args.seed,
        args.overwrite,
    )
    print(
        "转换完成: "
        f"图片={summary['image_count']}, train={summary['split_counts']['train']}, "
        f"val={summary['split_counts']['val']}, worms={summary['worm_roi_count']}"
    )


if __name__ == "__main__":
    main()
