"""Convert Labelme polygon annotations to Ultralytics YOLO segmentation labels."""

import argparse
import json
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np


def labelme2yolo(labelme_dir: Path, yolo_dir: Path, class_names: List[str]) -> None:
    """Convert Labelme data using the segmentation format: class + polygon points."""
    image_output = yolo_dir / "images" / "train"
    label_output = yolo_dir / "labels" / "train"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    converted = 0
    for json_path in sorted(labelme_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        image_path = labelme_dir / data.get("imagePath", "")
        if not image_path.is_file():
            print(f"跳过 {json_path.name}: 找不到对应图像 -> {image_path}")
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"跳过 {json_path.name}: 无法读取图像 -> {image_path}")
            continue
        height, width = image.shape[:2]
        shutil.copy2(image_path, image_output / image_path.name)

        label_lines = []
        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label not in class_names:
                continue
            points = np.asarray(shape.get("points", []), dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
                print(f"跳过 {json_path.name} 中无效轮廓: {label}")
                continue
            points[:, 0] /= width
            points[:, 1] /= height
            coordinates = " ".join(f"{value:.6f}" for value in points.flatten())
            label_lines.append(f"{class_names.index(label)} {coordinates}")

        with (label_output / f"{json_path.stem}.txt").open("w", encoding="utf-8") as handle:
            handle.write("\n".join(label_lines) + ("\n" if label_lines else ""))
        converted += 1

    yaml_path = yolo_dir.resolve().as_posix()
    dataset_yaml = (
        f"path: {yaml_path}\n"
        "train: images/train\n"
        "val: images/train\n"
        "test:\n\n"
        "names:\n"
        + "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
        + "\n"
    )
    (yolo_dir / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    print(f"Labelme 转换完成: 图片={converted}, 输出目录={yolo_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Labelme annotations to YOLO segmentation format.")
    parser.add_argument("--labelme-dir", default="label/converted", help="Labelme JSON 目录")
    parser.add_argument("--yolo-dir", default="label/yolo_dataset", help="YOLO 数据集输出目录")
    parser.add_argument("--classes", nargs="*", default=["worm"], help="需要导出的类别名称")
    args = parser.parse_args()

    labelme2yolo(Path(args.labelme_dir), Path(args.yolo_dir), args.classes)


if __name__ == "__main__":
    main()
