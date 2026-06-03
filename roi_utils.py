"""Shared helpers for reading ImageJ ROI sets paired with microscopy images."""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from roifile import roiread


IMAGE_EXTENSIONS = {".tif", ".tiff"}


def discover_image_roi_pairs(input_dir: Path) -> List[Tuple[Path, Path]]:
    """Return sorted image/ROI zip pairs and reject ambiguous or missing files."""
    input_dir = Path(input_dir)
    images: Dict[str, Path] = {}
    for image_path in input_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if image_path.stem in images:
            raise ValueError(f"同名图像不唯一: {images[image_path.stem].name}, {image_path.name}")
        images[image_path.stem] = image_path

    zips = {path.stem: path for path in input_dir.glob("*.zip")}
    images_without_zip = sorted(set(images) - set(zips))
    zips_without_image = sorted(set(zips) - set(images))
    if images_without_zip or zips_without_image:
        details = []
        if images_without_zip:
            details.append(f"缺少 zip 的图像: {', '.join(images_without_zip)}")
        if zips_without_image:
            details.append(f"缺少图像的 zip: {', '.join(zips_without_image)}")
        raise ValueError("; ".join(details))
    if not images:
        raise ValueError(f"未在 {input_dir} 中找到 .tif 或 .tiff 图像")

    return [(images[stem], zips[stem]) for stem in sorted(images)]


def read_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return image


def read_gray_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取灰度图像: {image_path}")
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def read_roi_polygons(zip_path: Path, width: int, height: int) -> List[dict]:
    """Read polygon-like ROIs and validate that points are usable on an image."""
    roi_objects = roiread(str(zip_path))
    if not isinstance(roi_objects, list):
        roi_objects = [roi_objects]
    if not roi_objects:
        raise ValueError(f"ROI zip 中没有轮廓: {zip_path}")

    polygons = []
    for index, roi in enumerate(roi_objects, start=1):
        name = str(roi.name or f"roi_{index:03d}")
        roi_id = f"{index:03d}:{name}"

        points = np.asarray(roi.coordinates(), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError(f"{zip_path.name}/{name} 不是有效多边形 ROI")
        if not np.isfinite(points).all():
            raise ValueError(f"{zip_path.name}/{name} 含有非数值坐标")
        if (
            (points[:, 0] < 0).any()
            or (points[:, 0] > width).any()
            or (points[:, 1] < 0).any()
            or (points[:, 1] > height).any()
        ):
            raise ValueError(f"{zip_path.name}/{name} 的坐标超出图像范围")

        polygons.append(
            {
                "id": roi_id,
                "name": name,
                "roi_type": str(roi.roitype).split(".")[-1].lower(),
                "points": points,
            }
        )
    return polygons


def load_pair(image_path: Path, zip_path: Path) -> Tuple[np.ndarray, List[dict]]:
    image = read_image(image_path)
    height, width = image.shape[:2]
    polygons = read_roi_polygons(zip_path, width, height)
    return image, polygons


def polygon_mask(shape: Tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    polygon = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)
