"""Mine low-quality segmentation images by comparing predictions with YOLO masks."""

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


def load_dataset_paths(yaml_path: Path) -> Tuple[Path, dict]:
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(config["path"])
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root, config


def image_paths(root: Path, config: dict, split: str) -> List[Path]:
    image_dir = Path(config[split])
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    if not image_dir.is_dir():
        raise FileNotFoundError(f"找不到 {split} 图像目录: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_path(root: Path, split: str, image_path: Path) -> Path:
    path = root / "labels" / split / f"{image_path.stem}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"图像缺少标签: {image_path.name}")
    return path


def read_shape(image_path: Path) -> Tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return image.shape[:2]


def load_ground_truth_masks(path: Path, shape: Tuple[int, int]) -> List[np.ndarray]:
    height, width = shape
    masks = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) < 7 or (len(values) - 1) % 2:
            raise ValueError(f"分割标签格式无效: {path}:{line_number}")
        points = np.asarray([float(value) for value in values[1:]], dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        masks.append(mask.astype(bool))
    return masks


def load_prediction_masks(result, shape: Tuple[int, int]) -> Tuple[List[np.ndarray], np.ndarray]:
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return [], np.array([], dtype=np.float32)
    height, width = shape
    masks = [
        cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
        for mask in result.masks.data.cpu().numpy()
    ]
    return masks, result.boxes.conf.cpu().numpy()


def mask_ious(ground_truth: List[np.ndarray], predictions: List[np.ndarray]) -> np.ndarray:
    if not ground_truth or not predictions:
        return np.zeros((len(ground_truth), len(predictions)), dtype=np.float32)
    ious = np.zeros((len(ground_truth), len(predictions)), dtype=np.float32)
    for gt_index, gt_mask in enumerate(ground_truth):
        for pred_index, pred_mask in enumerate(predictions):
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            ious[gt_index, pred_index] = intersection / union if union else 0.0
    return ious


def average_precision(tp: np.ndarray, confidence: np.ndarray, gt_count: int) -> float:
    if not len(confidence) or gt_count == 0:
        return 0.0
    order = np.argsort(-confidence)
    tp = tp[order].astype(np.float32)
    fp = 1.0 - tp
    recall = np.cumsum(tp) / gt_count
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-12)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    precision = np.flip(np.maximum.accumulate(np.flip(precision)))
    return float(np.trapz(np.interp(np.linspace(0, 1, 101), recall, precision), np.linspace(0, 1, 101)))


def match_predictions(ious: np.ndarray, confidence: np.ndarray, threshold: float) -> np.ndarray:
    matched_gt = set()
    true_positive = np.zeros(len(confidence), dtype=bool)
    for pred_index in np.argsort(-confidence):
        if not len(ious):
            continue
        candidates = np.argsort(-ious[:, pred_index])
        for gt_index in candidates:
            if ious[gt_index, pred_index] < threshold:
                break
            if int(gt_index) not in matched_gt:
                matched_gt.add(int(gt_index))
                true_positive[pred_index] = True
                break
    return true_positive


def score_image(ground_truth: List[np.ndarray], predictions: List[np.ndarray], confidence: np.ndarray) -> dict:
    ious = mask_ious(ground_truth, predictions)
    ap_values = []
    tp50 = match_predictions(ious, confidence, 0.5)
    for threshold in IOU_THRESHOLDS:
        true_positive = match_predictions(ious, confidence, float(threshold))
        ap_values.append(average_precision(true_positive, confidence, len(ground_truth)))
    matched50 = int(tp50.sum())
    precision50 = matched50 / len(predictions) if predictions else 0.0
    recall50 = matched50 / len(ground_truth) if ground_truth else 0.0
    best_gt_iou = ious.max(axis=1) if predictions else np.zeros(len(ground_truth), dtype=np.float32)
    return {
        "gt_count": len(ground_truth),
        "pred_count": len(predictions),
        "matched_mask_iou50": matched50,
        "mask_precision50": precision50,
        "mask_recall50": recall50,
        "mask_ap50": ap_values[0],
        "mask_map50_95": float(np.mean(ap_values)),
        "mean_best_gt_iou": float(best_gt_iou.mean()) if len(best_gt_iou) else 0.0,
    }


def mine_hard_examples(
    model_path: Path,
    dataset_yaml: Path,
    output_dir: Path,
    map_threshold: float,
    recall_threshold: float,
    conf: float,
    detection_conf: float,
    imgsz: int,
    device: str,
    save_previews: bool,
) -> dict:
    root, config = load_dataset_paths(dataset_yaml)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    if save_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    rows = []

    for split in ("train", "val"):
        for image_path in image_paths(root, config, split):
            shape = read_shape(image_path)
            ground_truth = load_ground_truth_masks(label_path(root, split, image_path), shape)
            result = model.predict(
                source=str(image_path), conf=conf, imgsz=imgsz, device=device, max_det=300, verbose=False
            )[0]
            predictions, confidence = load_prediction_masks(result, shape)
            metrics = score_image(ground_truth, predictions, confidence)
            detection_count = int((confidence >= detection_conf).sum())
            reasons = []
            if detection_count == 0:
                reasons.append("no_detection")
            if metrics["mask_map50_95"] < map_threshold:
                reasons.append("low_mask_map50_95")
            if metrics["mask_recall50"] < recall_threshold:
                reasons.append("low_mask_recall50")
            row = {
                "split": split,
                "image_file": image_path.name,
                **metrics,
                "pred_count_at_detection_conf": detection_count,
                "is_hard": bool(reasons),
                "reasons": ";".join(reasons),
            }
            rows.append(row)
            if reasons and save_previews:
                result.save(filename=str(preview_dir / f"{split}__{image_path.stem}.jpg"))

    fieldnames = list(rows[0])
    with (output_dir / "all_image_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    hard_rows = [row for row in rows if row["is_hard"]]
    with (output_dir / "hard_examples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(hard_rows)
    (output_dir / "hard_examples.txt").write_text(
        "\n".join(f"{row['split']}/{row['image_file']}\t{row['reasons']}" for row in hard_rows) + "\n",
        encoding="utf-8",
    )
    summary = {
        "model": str(model_path),
        "dataset": str(dataset_yaml),
        "map_threshold": map_threshold,
        "recall_threshold": recall_threshold,
        "conf": conf,
        "detection_conf": detection_conf,
        "images": len(rows),
        "hard_images": len(hard_rows),
        "hard_train_images": sum(row["split"] == "train" for row in hard_rows),
        "hard_val_images": sum(row["split"] == "val" for row in hard_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate each YOLO segmentation image and record hard examples.")
    parser.add_argument("--model", required=True, help="训练完成的分割模型 best.pt")
    parser.add_argument("--data", default="label/combined_yolo_dataset/dataset.yaml", help="YOLO dataset.yaml")
    parser.add_argument("--output-dir", default="hard_mining/stage2_best", help="难例报告输出目录")
    parser.add_argument("--map-threshold", type=float, default=0.4, help="低于此 Mask mAP50-95 视为难例")
    parser.add_argument("--recall-threshold", type=float, default=0.7, help="低于此 Mask Recall@0.5 视为难例")
    parser.add_argument("--conf", type=float, default=0.001, help="预测置信度阈值；低阈值适合 AP 计算")
    parser.add_argument("--detection-conf", type=float, default=0.1, help="用于判定实际漏检的置信度阈值")
    parser.add_argument("--imgsz", type=int, default=1376, help="推理图像尺寸")
    parser.add_argument("--device", default="0", help="推理设备")
    parser.add_argument("--save-previews", action="store_true", help="保存难例预测预览图")
    args = parser.parse_args()

    summary = mine_hard_examples(
        Path(args.model),
        Path(args.data),
        Path(args.output_dir),
        args.map_threshold,
        args.recall_threshold,
        args.conf,
        args.detection_conf,
        args.imgsz,
        args.device,
        args.save_previews,
    )
    print(
        f"难例筛选完成: images={summary['images']}, hard={summary['hard_images']} "
        f"(train={summary['hard_train_images']}, val={summary['hard_val_images']})"
    )


if __name__ == "__main__":
    main()
