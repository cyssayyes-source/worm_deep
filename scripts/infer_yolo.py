"""Run worm segmentation inference and calculate background-corrected intensity."""

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
INSTANCE_COLORS = [
    (0, 255, 0),
    (0, 180, 255),
    (255, 128, 0),
    (255, 0, 200),
    (0, 255, 255),
    (255, 80, 80),
    (160, 255, 0),
    (160, 80, 255),
    (255, 255, 0),
    (80, 220, 255),
]


def resize_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resize a model mask to original image dimensions and return a boolean mask."""
    height, width = shape
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized > 0.5


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image_f = image.astype(np.float32)
    min_value = float(np.min(image_f))
    max_value = float(np.max(image_f))
    if max_value <= min_value:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    return np.clip((image_f - min_value) * 255.0 / (max_value - min_value), 0, 255).astype(np.uint8)


def brighten_gray(gray: np.ndarray, factor: float) -> np.ndarray:
    """Enhance grayscale brightness for model detection only."""
    gray_u8 = normalize_to_uint8(gray)
    return np.clip(gray_u8.astype(np.float32) * max(float(factor), 1.0), 0, 255).astype(np.uint8)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in one predicted instance mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 2:
        return mask
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(component_areas)) + 1
    return labels == largest_label


def mean_intensity(image: np.ndarray, mask: np.ndarray) -> Optional[float]:
    pixels = image[mask]
    return float(pixels.mean()) if pixels.size else None


def read_gray_image(image_path: Path) -> Optional[np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def label_position(mask: np.ndarray) -> Tuple[int, int]:
    """Return a visible point inside the mask for an instance number marker."""
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    _, _, _, point = cv2.minMaxLoc(distance)
    return int(point[0]), int(point[1])


def draw_marker(image: np.ndarray, position: Tuple[int, int], text: str, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    radius = max(16, (text_width + 10) // 2)
    x, y = position
    cv2.circle(image, (x, y), radius, (0, 0, 0), -1)
    cv2.circle(image, (x, y), radius, color, 2)
    cv2.putText(
        image,
        text,
        (x - text_width // 2, y + text_height // 2),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def build_annotated_preview(preview: np.ndarray, rows: List[dict], background_median: Optional[float]) -> np.ndarray:
    """Append a readable instance legend beside the image."""
    panel_width = 390
    panel = np.full((preview.shape[0], panel_width, 3), 24, dtype=np.uint8)
    cv2.putText(panel, "Worm intensity summary", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    bg_text = "Background median: n/a" if background_median is None else f"Background median: {background_median:.2f}"
    cv2.putText(panel, bg_text, (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(panel, "ID    raw       corrected   conf", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
    for index, row in enumerate(rows):
        y = 132 + index * 31
        if y > panel.shape[0] - 16:
            cv2.putText(panel, "... see CSV for more rows", (20, panel.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
            break
        color = row["color"]
        corrected = row["corrected_mean_intensity"]
        corrected_text = "n/a" if corrected is None else f"{corrected:7.2f}"
        cv2.circle(panel, (30, y - 5), 9, color, -1)
        cv2.putText(panel, f"W{row['worm_id']:<3} {row['raw_mean_intensity']:7.2f}   {corrected_text}   {row['confidence']:.3f}", (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return np.hstack([preview, panel])


def overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Return intersection area divided by the smaller mask area."""
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    smaller_area = min(area_a, area_b)
    if smaller_area == 0:
        return 0.0
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    return intersection / smaller_area


def filter_overlapping_detections(
    masks: List[np.ndarray], confidences: List[float], overlap_thres: float
) -> Tuple[List[np.ndarray], List[float], int]:
    """Suppress lower-confidence masks that mostly overlap a higher-confidence mask."""
    if overlap_thres >= 1.0 or len(masks) <= 1:
        return masks, confidences, 0

    ordered_indices = sorted(range(len(masks)), key=lambda index: confidences[index], reverse=True)
    kept_masks: List[np.ndarray] = []
    kept_confidences: List[float] = []
    suppressed = 0
    for index in ordered_indices:
        mask = masks[index]
        is_duplicate = any(overlap_ratio(mask, kept_mask) >= overlap_thres for kept_mask in kept_masks)
        if is_duplicate:
            suppressed += 1
            continue
        kept_masks.append(mask)
        kept_confidences.append(confidences[index])
    return kept_masks, kept_confidences, suppressed


def process_image(
    model: YOLO,
    image_path: Path,
    output_dir: Path,
    conf: float = 0.1,
    imgsz: int = 1376,
    overlap_thres: float = 0.85,
    keep_largest: bool = True,
    enhance_brightness: bool = False,
    brightness_factor: float = 1.5,
) -> List[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gray = read_gray_image(image_path)
    if gray is None:
        print(f"Cannot read image: {image_path}")
        return []

    detection_gray = brighten_gray(gray, brightness_factor) if enhance_brightness else normalize_to_uint8(gray)
    detection_image = cv2.cvtColor(detection_gray, cv2.COLOR_GRAY2BGR)
    preview = cv2.cvtColor(normalize_to_uint8(gray), cv2.COLOR_GRAY2BGR)
    result = model(detection_image, conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes = result.boxes
    masks = result.masks

    full_masks = []
    confidences = []
    if boxes is not None and masks is not None:
        confidences = [float(value) for value in boxes.conf.cpu().numpy()]
        for mask in masks.data.cpu().numpy():
            full_mask = resize_mask(mask, gray.shape)
            if keep_largest:
                full_mask = keep_largest_component(full_mask)
            full_masks.append(full_mask)
    full_masks, confidences, suppressed_count = filter_overlapping_detections(full_masks, confidences, overlap_thres)

    combined_mask = np.zeros(gray.shape, dtype=bool)
    for mask in full_masks:
        combined_mask |= mask
    background_pixels = gray[~combined_mask]
    background_median = float(np.median(background_pixels)) if background_pixels.size else None

    rows = []
    for index, mask in enumerate(full_masks, start=1):
        raw_mean = mean_intensity(gray, mask)
        corrected = raw_mean - background_median if raw_mean is not None and background_median is not None else None
        confidence = confidences[index - 1] if index - 1 < len(confidences) else None
        color = INSTANCE_COLORS[(index - 1) % len(INSTANCE_COLORS)]
        rows.append(
            {
                "worm_id": index,
                "confidence": confidence,
                "detection_image": "brightness_enhanced" if enhance_brightness else "original",
                "intensity_source": "original",
                "brightness_factor": brightness_factor if enhance_brightness else 1.0,
                "raw_mean_intensity": raw_mean,
                "background_median": background_median,
                "corrected_mean_intensity": corrected,
                "color": color,
            }
        )

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, color, 2)
        if contours:
            draw_marker(preview, label_position(mask), f"W{index}", color)

    cv2.imwrite(str(output_dir / "segmentation_result.jpg"), build_annotated_preview(preview, rows, background_median))
    status = "detected" if rows else "no_detections"
    if background_median is None:
        status = "no_background_pixels"
    with (output_dir / "intensity_stats.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "image_file",
            "status",
            "worm_id",
            "confidence",
            "detection_image",
            "intensity_source",
            "brightness_factor",
            "raw_mean_intensity",
            "background_median",
            "corrected_mean_intensity",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            for row in rows:
                csv_row = {key: value for key, value in row.items() if key != "color"}
                writer.writerow({"image_file": image_path.name, "status": status, **csv_row})
        else:
            writer.writerow(
                {
                    "image_file": image_path.name,
                    "status": status,
                    "worm_id": "",
                    "confidence": "",
                    "detection_image": "brightness_enhanced" if enhance_brightness else "original",
                    "intensity_source": "original",
                    "brightness_factor": brightness_factor if enhance_brightness else 1.0,
                    "raw_mean_intensity": "",
                    "background_median": background_median,
                    "corrected_mean_intensity": "",
                }
            )

    with (output_dir / "intensity_stats.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Image: {image_path.name}\n")
        handle.write(f"Status: {status}\n")
        handle.write(f"Detected worms: {len(rows)}\n")
        handle.write(f"Detection image: {'brightness_enhanced' if enhance_brightness else 'original'}\n")
        handle.write(f"Brightness factor: {brightness_factor:.2f}\n")
        handle.write("Intensity source: original\n")
        handle.write(f"Filtered overlapping detections: {suppressed_count}\n")
        handle.write(f"Overlap threshold: {overlap_thres:.2f}\n")
        if background_median is None:
            handle.write("Background median: n/a\n")
        else:
            handle.write(f"Background median: {background_median:.2f}\n")
        for row in rows:
            corrected_text = (
                f"{row['corrected_mean_intensity']:.2f}"
                if row["corrected_mean_intensity"] is not None
                else "n/a"
            )
            handle.write(
                f"Worm {row['worm_id']}: raw_mean={row['raw_mean_intensity']:.2f}, "
                f"corrected_mean={corrected_text}, confidence={row['confidence']:.4f}\n"
            )

    print(f"Done: {image_path.name}, worms={len(rows)}, filtered_overlap={suppressed_count}, status={status}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO segmentation inference and calculate fluorescence stats.")
    parser.add_argument("--model", default="models/worm_yolo26s_best.pt", help="YOLO weights path")
    parser.add_argument("--input-dir", default="waiting_images", help="Input image directory")
    parser.add_argument("--output-root", default="output_root", help="Output root directory")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1376, help="Inference image size")
    parser.add_argument(
        "--enhance-brightness",
        "--enhance-contrast",
        action=argparse.BooleanOptionalAction,
        dest="enhance_brightness",
        default=False,
        help="Enhance brightness for detection only. Intensity statistics always use the original image. --enhance-contrast is a deprecated alias.",
    )
    parser.add_argument("--brightness-factor", type=float, default=1.5, help="Brightness multiplier used with --enhance-brightness.")
    parser.add_argument(
        "--overlap-thres",
        type=float,
        default=0.85,
        help="Suppress lower-confidence masks when intersection/min_area is at least this value. Use 1.0 to disable.",
    )
    parser.add_argument(
        "--keep-largest-component",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only the largest connected component for each predicted mask.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    model = YOLO(args.model)
    print(
        "Detection image:",
        "brightness_enhanced" if args.enhance_brightness else "original",
        "| Intensity source: original",
        f"| Brightness factor: {args.brightness_factor:.2f}",
    )
    image_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in input directory: {input_dir}")
    for image_path in image_paths:
        process_image(
            model,
            image_path,
            Path(args.output_root) / image_path.stem,
            args.conf,
            args.imgsz,
            args.overlap_thres,
            args.keep_largest_component,
            args.enhance_brightness,
            args.brightness_factor,
        )


if __name__ == "__main__":
    main()
