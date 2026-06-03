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


def process_image(model: YOLO, image_path: Path, output_dir: Path, conf: float = 0.1, imgsz: int = 1376) -> List[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gray = read_gray_image(image_path)
    if gray is None:
        print(f"Cannot read image: {image_path}")
        return []

    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result = model(preview, conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes = result.boxes
    masks = result.masks

    full_masks = []
    if boxes is not None and masks is not None:
        for mask in masks.data.cpu().numpy():
            full_masks.append(resize_mask(mask, gray.shape))

    combined_mask = np.zeros(gray.shape, dtype=bool)
    for mask in full_masks:
        combined_mask |= mask
    background_pixels = gray[~combined_mask]
    background_median = float(np.median(background_pixels)) if background_pixels.size else None

    rows = []
    for index, mask in enumerate(full_masks, start=1):
        raw_mean = mean_intensity(gray, mask)
        corrected = raw_mean - background_median if raw_mean is not None and background_median is not None else None
        confidence = float(boxes.conf[index - 1].cpu().item()) if boxes is not None else None
        color = INSTANCE_COLORS[(index - 1) % len(INSTANCE_COLORS)]
        rows.append(
            {
                "worm_id": index,
                "confidence": confidence,
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
                    "raw_mean_intensity": "",
                    "background_median": background_median,
                    "corrected_mean_intensity": "",
                }
            )

    with (output_dir / "intensity_stats.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Image: {image_path.name}\n")
        handle.write(f"Status: {status}\n")
        handle.write(f"Detected worms: {len(rows)}\n")
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

    print(f"Done: {image_path.name}, worms={len(rows)}, status={status}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO segmentation inference and calculate fluorescence stats.")
    parser.add_argument("--model", default="models/worm_yolo_best.pt", help="YOLO weights path")
    parser.add_argument("--input-dir", default="waiting_images", help="Input image directory")
    parser.add_argument("--output-root", default="output_root", help="Output root directory")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1376, help="Inference image size")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    model = YOLO(args.model)
    image_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in input directory: {input_dir}")
    for image_path in image_paths:
        process_image(model, image_path, Path(args.output_root) / image_path.stem, args.conf, args.imgsz)


if __name__ == "__main__":
    main()
